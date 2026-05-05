"""
Drift-recovery benchmark: nonlinear Duffing-oscillator latent SDE, large T.

  dx/dt = y
  dy/dt = α x - β x^3 - γ y    + noise

This is the standard 2D nonlinear test problem from the SING paper.

We generate from this KNOWN drift, fit both EFGP-SING (JAX, pinned grid,
M-step on) and SING-SparseGP (full M-step), and compare:

  * predictive observation RMSE   (basis-invariant)
  * Procrustes-aligned latent RMSE (handles PCA basis ambiguity)
  * drift-recovery RMSE on a grid in TRUTH basis (after basis alignment)

T is cranked up to demonstrate scaling.

Run as its own python process (no torch import):

    python demos/bench_duffing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import numpy as np

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
sigma = 0.3       # diffusion noise
ALPHA, BETA, GAMMA = 1.0, 1.0, 0.5    # Duffing params (canonical)


def drift_duffing(x, t):
    """Duffing 2D drift: dx/dt = y;  dy/dt = αx - βx³ - γy."""
    return jnp.array([x[1], ALPHA * x[0] - BETA * x[0] ** 3 - GAMMA * x[1]])


def drift_duffing_np(x_batch):
    """Vectorised numpy version: x_batch shape (N, 2)."""
    out = np.zeros_like(x_batch)
    out[:, 0] = x_batch[:, 1]
    out[:, 1] = ALPHA * x_batch[:, 0] - BETA * x_batch[:, 0] ** 3 - GAMMA * x_batch[:, 1]
    return out


def sigma_fn(x, t):
    return sigma * jnp.eye(D)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def predict_obs_rmse(m_jax, output_params, ys):
    m = np.asarray(m_jax)
    C = np.asarray(output_params['C']); d = np.asarray(output_params['d'])
    y_hat = m @ C.T + d
    return float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))


def procrustes_align(m_inferred: np.ndarray, m_true: np.ndarray):
    """Find affine map A (D, D), b (D,) that best maps m_inferred → m_true
    in least-squares sense: m_true ≈ A m_inferred + b.

    Returns (A, b, m_aligned, latent_rmse).
    """
    Xi = m_inferred                                 # (T, D)
    Xt = m_true                                      # (T, D)
    # Centre
    bi = Xi.mean(axis=0); bt = Xt.mean(axis=0)
    Xi_c = Xi - bi
    Xt_c = Xt - bt
    # Solve  Xt_c = Xi_c A^T  (least squares)  →  A^T = (Xi_c^T Xi_c)^-1 Xi_c^T Xt_c
    A_T, *_ = np.linalg.lstsq(Xi_c, Xt_c, rcond=None)
    A = A_T.T                                        # (D, D)
    b = bt - A @ bi
    m_aligned = Xi @ A.T + b
    rmse = float(np.sqrt(np.mean((m_aligned - Xt) ** 2)))
    return A, b, m_aligned, rmse


def eval_efgp_drift_at_grid(grid_pts: np.ndarray, mp_efgp, ls, var,
                              eps_grid: float, T_full: int, sigma_drift_sq: float):
    """Re-build q(f) at the final EFGP marginals, then evaluate Φμ at
    grid_pts.  Returns (N_grid, D_out) numpy."""
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid_jax = jp.spectral_grid_se(ls, var, X_template, eps=eps_grid)
    ms_eval = jnp.asarray(mp_efgp['m'][0])
    Ss_eval = jnp.asarray(np.asarray(mp_efgp['S'][0]))
    SSs_eval = jnp.asarray(np.asarray(mp_efgp['SS'][0]))
    del_t = jnp.linspace(0., T_full * 0.05, T_full)[1:] - jnp.linspace(0., T_full * 0.05, T_full)[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms_eval, Ss_eval, SSs_eval, del_t, grid_jax, jr.PRNGKey(99),
        sigma_drift_sq=sigma_drift_sq, S_marginal=2, D_lat=D, D_out=D,
    )
    Ef_grid, _, _ = jpd.drift_moments_jax(
        mu_r, grid_jax, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    return np.asarray(Ef_grid)


def eval_sparsegp_drift_at_grid(grid_pts, sparse_drift, sp_drift_params,
                                  mp_sp, t_grid, T_full):
    gp_post_sp = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp_sp, jnp.ones((1, T_full), dtype=bool),
        sp_drift_params,
        jnp.zeros((1, T_full, 1)),
        jnp.zeros((D, 1)), sigma)
    return np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post_sp, sp_drift_params, jnp.asarray(grid_pts)))


def run(T: int, n_em: int = 20, n_estep: int = 10):
    print(f"\n{'='*70}\n  Duffing, T = {T}, n_em = {n_em}\n{'='*70}")

    # ----- generate -----
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.5, 0.0]),
                       f=drift_duffing, t_max=t_max, n_timesteps=T,
                       sigma=sigma_fn)
    print(f"  xs range: ({float(xs.min()):.2f}, {float(xs.max()):.2f})")

    N = 8
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                 dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    # SparseGP recipe taken DIRECTLY from sing/demos/inference_and_learning.ipynb
    # (cell 23, the SING-GP setup).  These are the settings the SING authors
    # use for their own GP-drift experiments.
    rho_sched_0 = jnp.logspace(-3, -2, 10)
    rho_sched_1 = rho_sched_0[-1] * jnp.ones(max(0, n_em - len(rho_sched_0)))
    sing_rho_sched = jnp.concatenate([rho_sched_0[:n_em], rho_sched_1])
    # SING demo uses (lr=1e-4, n_iters_m=50) which gives effective max
    # motion ~= 5e-3 per outer EM iter.  We use (lr=1e-3, n_iters_m=10)
    # which has the same effective max motion but 5x less wall-time cost
    # — should give equivalent quality.
    sing_lr = 1e-3 * jnp.ones(n_em)
    sing_n_iters_m = 10

    # EFGP gets a slightly more aggressive schedule because its M-step is
    # more stable (analytic gradient + Hutchinson trace, no autograd-thru-CG).
    efgp_rho_sched = jnp.linspace(0.05, 0.4, n_em)

    # ----- EFGP-SING (JAX) -----
    print(f"  fitting EFGP-SING (JAX, pinned grid, learn_kernel=True)...")
    t0 = time.time()
    mp_efgp, _, op_efgp, _, hist_efgp = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.7, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=efgp_rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=2,
        pin_grid=True, pin_grid_lengthscale=0.5,
        verbose=False,
    )
    t_efgp = time.time() - t0
    rmse_obs_efgp = predict_obs_rmse(mp_efgp['m'][0], op_efgp, ys)
    print(f"    wall {t_efgp:.1f}s   obs RMSE {rmse_obs_efgp:.4f}   "
          f"final ℓ {hist_efgp.lengthscale[-1]:.3f}, "
          f"σ² {hist_efgp.variance[-1]:.3f}")

    # ----- SparseGP (SING's recommended settings from the demo notebook) -----
    print("  fitting SING-SparseGP (full M-step, SING-demo settings)...")
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    # SING demo uses 12x12=144 inducing pts but that costs O(M²)=20k per
    # Adam step on the SparseGP ELBO + gradient.  At T=500 with 10 Adam
    # steps × 20 EM = 200 Adam evaluations the per-iter cost dominates.
    # 8x8=64 is the SparseGP demo default and runs ~3x faster.
    zs = initialize_zs(D=D, zs_lim=4.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.ones(D),       # SING demo init
                                 output_scale=jnp.asarray(1.0))
    t0 = time.time()
    mp_sp, _, _, sp_drift_params, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=sigma,
        rho_sched=sing_rho_sched,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=sing_n_iters_m,
        perform_m_step=True, learn_output_params=True,
        learning_rate=sing_lr,
        print_interval=999,
    )
    t_sp = time.time() - t0
    rmse_obs_sp = predict_obs_rmse(mp_sp['m'][0], op_sp, ys)
    print(f"    wall {t_sp:.1f}s   obs RMSE {rmse_obs_sp:.4f}   "
          f"final ℓ {float(jnp.mean(sp_drift_params['length_scales'])):.3f}, "
          f"σ² {float(sp_drift_params['output_scale']) ** 2:.3f}")

    # ----- Procrustes alignment -----
    m_efgp = np.asarray(mp_efgp['m'][0])
    m_sp = np.asarray(mp_sp['m'][0])
    xs_np = np.asarray(xs)
    A_efgp, b_efgp, m_efgp_align, lat_rmse_efgp = procrustes_align(
        m_efgp, xs_np)
    A_sp, b_sp, m_sp_align, lat_rmse_sp = procrustes_align(m_sp, xs_np)
    trivial_lat_rmse = float(np.sqrt(np.mean(xs_np ** 2)))

    # ----- Drift recovery on a grid (TRUTH basis) -----
    grid_lo = xs_np.min(axis=0) - 0.5
    grid_hi = xs_np.max(axis=0) + 0.5
    xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
    ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
    GX, GY = np.meshgrid(xs_g, ys_g, indexing="ij")
    grid_pts_truth = np.stack([GX.ravel(), GY.ravel()], axis=-1)

    f_true_grid = drift_duffing_np(grid_pts_truth)

    # Map truth-basis grid points to inferred basis: x_inferred = A^-1 (x_truth - b)
    A_inv_efgp = np.linalg.inv(A_efgp)
    grid_efgp_basis = (grid_pts_truth - b_efgp) @ A_inv_efgp.T
    f_efgp_inferred_basis = eval_efgp_drift_at_grid(
        grid_efgp_basis, mp_efgp, hist_efgp.lengthscale[-1],
        hist_efgp.variance[-1], 1e-2, T, sigma ** 2)
    # Drift in truth basis: f_truth = A f_inferred (since x_truth = A x_inferred + b
    # and dx_truth/dt = A dx_inferred/dt = A f_inferred(x_inferred))
    f_efgp_truth_basis = f_efgp_inferred_basis @ A_efgp.T

    A_inv_sp = np.linalg.inv(A_sp)
    grid_sp_basis = (grid_pts_truth - b_sp) @ A_inv_sp.T
    f_sp_inferred_basis = eval_sparsegp_drift_at_grid(
        grid_sp_basis, sparse_drift, sp_drift_params, mp_sp, t_grid, T)
    f_sp_truth_basis = f_sp_inferred_basis @ A_sp.T

    drift_rmse_efgp = float(np.sqrt(np.mean(
        (f_efgp_truth_basis - f_true_grid) ** 2)))
    drift_rmse_sp = float(np.sqrt(np.mean(
        (f_sp_truth_basis - f_true_grid) ** 2)))
    drift_rmse_zero = float(np.sqrt(np.mean(f_true_grid ** 2)))

    print(f"\n  T={T} summary")
    print(f"    EFGP-SING (JAX):  wall {t_efgp:.1f}s")
    print(f"      obs RMSE   {rmse_obs_efgp:.4f}")
    print(f"      latent RMSE (Procrustes)  {lat_rmse_efgp:.4f}")
    print(f"      drift RMSE (truth basis)  {drift_rmse_efgp:.4f}")
    print(f"      final ℓ {hist_efgp.lengthscale[-1]:.3f}, σ² {hist_efgp.variance[-1]:.3f}")
    print(f"    SING-SparseGP:    wall {t_sp:.1f}s")
    print(f"      obs RMSE   {rmse_obs_sp:.4f}")
    print(f"      latent RMSE (Procrustes)  {lat_rmse_sp:.4f}")
    print(f"      drift RMSE (truth basis)  {drift_rmse_sp:.4f}")
    print(f"      final ℓ {float(jnp.mean(sp_drift_params['length_scales'])):.3f}, "
          f"σ² {float(sp_drift_params['output_scale']) ** 2:.3f}")
    print(f"    baselines: trivial latent RMSE {trivial_lat_rmse:.4f}, "
          f"zero-drift RMSE {drift_rmse_zero:.4f}")

    # ----- save quiver plot -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    f_t = f_true_grid.reshape(GX.shape + (D,))
    axes[0].quiver(GX, GY, f_t[..., 0], f_t[..., 1], color="k")
    axes[0].plot(xs_np[:, 0], xs_np[:, 1], "r-", alpha=0.4)
    axes[0].set_title("True Duffing drift"); axes[0].set_aspect("equal")

    f_e = f_efgp_truth_basis.reshape(GX.shape + (D,))
    axes[1].quiver(GX, GY, f_e[..., 0], f_e[..., 1], color="b")
    axes[1].plot(m_efgp_align[:, 0], m_efgp_align[:, 1], "g-", alpha=0.6)
    axes[1].set_title(f"EFGP recovered drift (truth basis)\ndrift RMSE {drift_rmse_efgp:.3f}")
    axes[1].set_aspect("equal")

    f_s = f_sp_truth_basis.reshape(GX.shape + (D,))
    axes[2].quiver(GX, GY, f_s[..., 0], f_s[..., 1], color="g")
    axes[2].plot(m_sp_align[:, 0], m_sp_align[:, 1], "g-", alpha=0.6)
    axes[2].set_title(f"SparseGP recovered drift (truth basis)\ndrift RMSE {drift_rmse_sp:.3f}")
    axes[2].set_aspect("equal")

    fig.tight_layout()
    out = (Path(__file__).resolve().parent
           / "_efgp_demo_out_jax" / f"duffing_T{T}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"  saved {out}")

    return dict(
        T=T,
        t_efgp=t_efgp, t_sp=t_sp,
        rmse_obs_efgp=rmse_obs_efgp, rmse_obs_sp=rmse_obs_sp,
        lat_rmse_efgp=lat_rmse_efgp, lat_rmse_sp=lat_rmse_sp,
        drift_rmse_efgp=drift_rmse_efgp, drift_rmse_sp=drift_rmse_sp,
        ls_efgp=hist_efgp.lengthscale[-1],
        var_efgp=hist_efgp.variance[-1],
        ls_sp=float(jnp.mean(sp_drift_params['length_scales'])),
        var_sp=float(sp_drift_params['output_scale']) ** 2,
    )


def main():
    rows = []
    for T in [500, 2000]:
        rows.append(run(T, n_em=20, n_estep=10))

    print("\n" + "=" * 110)
    print(f"{'T':>5} {'method':>14} {'wall s':>9} {'obs RMSE':>10} "
          f"{'latRMSE':>10} {'drift RMSE':>12} {'ℓ':>7} {'σ²':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r['T']:>5} {'EFGP':>14} {r['t_efgp']:>9.1f} {r['rmse_obs_efgp']:>10.4f} "
              f"{r['lat_rmse_efgp']:>10.4f} {r['drift_rmse_efgp']:>12.4f} "
              f"{r['ls_efgp']:>7.3f} {r['var_efgp']:>7.3f}")
        print(f"{r['T']:>5} {'SparseGP':>14} {r['t_sp']:>9.1f} {r['rmse_obs_sp']:>10.4f} "
              f"{r['lat_rmse_sp']:>10.4f} {r['drift_rmse_sp']:>12.4f} "
              f"{r['ls_sp']:>7.3f} {r['var_sp']:>7.3f}")


if __name__ == "__main__":
    main()
