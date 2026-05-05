"""
Hyperparameter recovery benchmark: drift sampled directly from a GP with
KNOWN (ℓ_true, σ²_true), large T, both methods recover.

We sample one fresh draw

    f_gp(x) ~ GP(0, k_SE(·,·; ℓ_true, σ²_true))   per output dim,

add a small restoring term  f_total(x) = f_gp(x) - α x  to keep the SDE
stable, simulate a trajectory + noisy obs, and run both methods with
``learn_kernel=True``.

Reports:
  * (ℓ_inferred / ℓ_true) and (σ²_inferred / σ²_true) for both methods
  * predictive observation RMSE
  * Procrustes-aligned latent RMSE
  * truth-basis drift recovery RMSE on a grid
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import math
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
sigma = 0.3                  # SDE diffusion noise
ALPHA_RESTORE = 0.5          # mean-reversion strength so trajectory stays bounded


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    """Sample a single 2D drift draw  f(x) = f_GP(x) - α x  with the GP
    component drawn from SE(ℓ_true, σ²_true).

    Returns ``drift_jax`` and ``drift_np`` callables.
    """
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps_per_r = []
    for k in keys:
        k1, k2 = jr.split(k)
        ee = (jax.random.normal(k1, (M,))
              + 1j * jax.random.normal(k2, (M,))).astype(grid.ws.dtype)
        eps_per_r.append(ee / math.sqrt(2))
    fk_draw = jnp.stack(
        [grid.ws * eps_per_r[r] for r in range(D)],
        axis=0,
    )                                                           # (D, M)
    OUT = grid.mtot_per_dim
    xcen = grid.xcen
    h_per_dim = grid.h_per_dim

    def drift_x_jax(x):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*OUT),
                            xcen, h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    def drift_jax(x, t):
        return drift_x_jax(x) - ALPHA_RESTORE * x

    # Numpy version for grid evaluation in benchmarks
    fk_draw_np = np.asarray(fk_draw)

    def drift_np(x_batch):
        xt = jnp.asarray(x_batch, dtype=jnp.float32)
        out = np.zeros_like(x_batch)
        for r in range(D):
            fr = jp.nufft2(xt, fk_draw[r].reshape(*OUT),
                            xcen, h_per_dim, eps=6e-8).real
            out[:, r] = np.asarray(fr) - ALPHA_RESTORE * x_batch[:, r]
        return out

    return drift_jax, drift_np


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


def procrustes_align(m_inferred, m_true):
    Xi = m_inferred; Xt = m_true
    bi = Xi.mean(0); bt = Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    b = bt - A @ bi
    return A, b, Xi @ A.T + b


def run(T, ls_true, var_true, n_em=25, n_estep=10):
    print(f"\n{'='*72}")
    print(f"  GP-drift recovery: T={T}, ls_true={ls_true}, var_true={var_true}")
    print(f"{'='*72}")

    drift_jax, drift_np = make_gp_drift(ls_true, var_true, jr.PRNGKey(123))
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([0.0, 0.0]),
                       f=drift_jax, t_max=t_max, n_timesteps=T,
                       sigma=sigma_fn)
    print(f"  xs range:({float(xs.min()):.2f}, {float(xs.max()):.2f}), "
          f"std={float(jnp.std(xs)):.2f}")

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

    # ----- EFGP settings -----
    efgp_rho = jnp.linspace(0.05, 0.4, n_em)
    # ----- SparseGP settings (SING-recommended, modest budget) -----
    sing_rho_0 = jnp.logspace(-3, -2, 10)
    sing_rho = jnp.concatenate(
        [sing_rho_0[:n_em], sing_rho_0[-1] * jnp.ones(max(0, n_em - 10))])
    sing_lr = 1e-3 * jnp.ones(n_em)

    # Both start from the SAME (deliberately wrong) init so we see who
    # finds ℓ_true.
    ls_init = 0.3 if ls_true >= 0.6 else 1.5
    print(f"  init ℓ={ls_init} (truth={ls_true});  "
          f"σ²_init=1.0 (truth={var_true})")

    # ----- EFGP-SING (JAX) -----
    # M-step now refreshes μ each Adam step (envelope-theorem-correct
    # gradient).  Use lr=0.05 with n_mstep=5 — this matches efgpnd's
    # standalone GP-regression hyperparameter recovery (~50 Adam steps
    # total to converge ℓ from a 4× wrong init).
    print("  fitting EFGP-SING (JAX, learn_kernel=True)...")
    t0 = time.time()
    mp_efgp, _, op_efgp, _, hist_efgp = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=efgp_rho,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=5, mstep_lr=0.05,
        n_hutchinson_mstep=16, kernel_warmup_iters=3,
        # Pinned grid: keeps mtot_per_dim static across the EM (no JIT
        # recompiles).  ~half-to-three-quarters of truth ℓ is plenty —
        # the M-step now finds the correct ℓ regardless of pin choice
        # (within reason), since μ-refresh removes the stale-gradient
        # bias that previously turned pin into an implicit regularizer.
        pin_grid=True, pin_grid_lengthscale=ls_true * 0.75,
        verbose=False,
    )
    t_efgp = time.time() - t0

    # ----- SparseGP -----
    print("  fitting SING-SparseGP (full M-step)...")
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=4.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.full((D,), float(ls_init)),
                                 output_scale=jnp.asarray(1.0))
    t0 = time.time()
    mp_sp, _, _, sp_drift_params, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=sigma,
        rho_sched=sing_rho,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=10,
        perform_m_step=True, learn_output_params=True,
        learning_rate=sing_lr,
        print_interval=999,
    )
    t_sp = time.time() - t0

    # ----- diagnostics -----
    rmse_obs_efgp = predict_obs_rmse(mp_efgp['m'][0], op_efgp, ys)
    rmse_obs_sp = predict_obs_rmse(mp_sp['m'][0], op_sp, ys)
    m_efgp = np.asarray(mp_efgp['m'][0])
    m_sp = np.asarray(mp_sp['m'][0])
    xs_np = np.asarray(xs)
    A_efgp, b_efgp, m_efgp_align = procrustes_align(m_efgp, xs_np)
    A_sp, b_sp, m_sp_align = procrustes_align(m_sp, xs_np)
    lat_rmse_efgp = float(np.sqrt(np.mean((m_efgp_align - xs_np) ** 2)))
    lat_rmse_sp = float(np.sqrt(np.mean((m_sp_align - xs_np) ** 2)))

    # Drift on a grid in TRUTH basis
    grid_lo = xs_np.min(0) - 0.3; grid_hi = xs_np.max(0) + 0.3
    xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
    ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
    GX, GY = np.meshgrid(xs_g, ys_g, indexing="ij")
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true_grid = drift_np(grid_pts)

    # EFGP drift in truth basis.
    # CRITICAL: use the SAME pinned grid that the EM loop used, otherwise
    # mu_r will be re-solved against a different ws and give a different
    # (and worse) drift.
    A_inv_efgp = np.linalg.inv(A_efgp)
    grid_efgp_basis = (grid_pts - b_efgp) @ A_inv_efgp.T
    X_template_e = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    pin_ls_used = ls_true * 0.75
    grid_jax_e = jp.spectral_grid_se(pin_ls_used,
                                      hist_efgp.variance[-1],
                                      X_template_e, eps=1e-2)
    ms_eval = jnp.asarray(m_efgp)
    Ss_eval = jnp.asarray(np.asarray(mp_efgp['S'][0]))
    SSs_eval = jnp.asarray(np.asarray(mp_efgp['SS'][0]))
    del_t = t_grid[1:] - t_grid[:-1]
    # Re-derive ws under the FINAL learned hypers (ℓ, σ²) on the pinned
    # grid — this matches what the EM loop's last iteration used.
    ws_final = jpd._ws_real_se(jnp.log(hist_efgp.lengthscale[-1]),
                                jnp.log(hist_efgp.variance[-1]),
                                grid_jax_e.xis_flat,
                                grid_jax_e.h_per_dim[0], D).astype(grid_jax_e.ws.dtype)
    grid_jax_e = grid_jax_e._replace(ws=ws_final)
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms_eval, Ss_eval, SSs_eval, del_t, grid_jax_e, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=2, D_lat=D, D_out=D,
    )
    Ef_grid_e, _, _ = jpd.drift_moments_jax(
        mu_r, grid_jax_e,
        jnp.asarray(grid_efgp_basis, dtype=jnp.float32),
        D_lat=D, D_out=D)
    f_efgp_truth = np.asarray(Ef_grid_e) @ A_efgp.T

    # SparseGP drift in truth basis
    A_inv_sp = np.linalg.inv(A_sp)
    grid_sp_basis = (grid_pts - b_sp) @ A_inv_sp.T
    gp_post_sp = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp_sp, jnp.ones((1, T), dtype=bool),
        sp_drift_params,
        jnp.zeros((1, T, 1)),
        jnp.zeros((D, 1)), sigma)
    f_sp_inferred = np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post_sp, sp_drift_params, jnp.asarray(grid_sp_basis)))
    f_sp_truth = f_sp_inferred @ A_sp.T

    drift_rmse_efgp = float(np.sqrt(np.mean(
        (f_efgp_truth - f_true_grid) ** 2)))
    drift_rmse_sp = float(np.sqrt(np.mean(
        (f_sp_truth - f_true_grid) ** 2)))
    drift_rmse_zero = float(np.sqrt(np.mean(f_true_grid ** 2)))

    ls_efgp_final = hist_efgp.lengthscale[-1]
    var_efgp_final = hist_efgp.variance[-1]
    ls_sp_final = float(jnp.mean(sp_drift_params['length_scales']))
    var_sp_final = float(sp_drift_params['output_scale']) ** 2

    print(f"\n  T={T} summary  (truth: ℓ={ls_true}, σ²={var_true})")
    print(f"    EFGP-SING (JAX):   wall {t_efgp:.1f}s")
    print(f"      ℓ {ls_efgp_final:.3f}  ({ls_efgp_final/ls_true*100:.0f}% of truth)"
          f"   σ² {var_efgp_final:.3f}  ({var_efgp_final/var_true*100:.0f}% of truth)")
    print(f"      obs RMSE {rmse_obs_efgp:.4f}   "
          f"latent RMSE {lat_rmse_efgp:.4f}   "
          f"drift RMSE {drift_rmse_efgp:.4f}")
    print(f"    SING-SparseGP:     wall {t_sp:.1f}s")
    print(f"      ℓ {ls_sp_final:.3f}  ({ls_sp_final/ls_true*100:.0f}% of truth)"
          f"   σ² {var_sp_final:.3f}  ({var_sp_final/var_true*100:.0f}% of truth)")
    print(f"      obs RMSE {rmse_obs_sp:.4f}   "
          f"latent RMSE {lat_rmse_sp:.4f}   "
          f"drift RMSE {drift_rmse_sp:.4f}")
    print(f"    zero-drift baseline RMSE {drift_rmse_zero:.4f}")

    return dict(
        T=T, ls_true=ls_true, var_true=var_true,
        t_efgp=t_efgp, t_sp=t_sp,
        ls_efgp=ls_efgp_final, var_efgp=var_efgp_final,
        ls_sp=ls_sp_final, var_sp=var_sp_final,
        rmse_obs_efgp=rmse_obs_efgp, rmse_obs_sp=rmse_obs_sp,
        lat_rmse_efgp=lat_rmse_efgp, lat_rmse_sp=lat_rmse_sp,
        drift_rmse_efgp=drift_rmse_efgp, drift_rmse_sp=drift_rmse_sp,
        drift_rmse_zero=drift_rmse_zero,
    )


def main():
    rows = []
    for T in [500, 2000]:
        rows.append(run(T, ls_true=0.8, var_true=1.0, n_em=25))

    print("\n" + "=" * 100)
    print(f"{'T':>5}  {'method':>14}  {'wall':>6}  "
          f"{'ℓ ratio':>8}  {'σ² ratio':>9}  "
          f"{'obs RMSE':>10}  {'lat RMSE':>10}  {'drift RMSE':>11}")
    print("-" * 100)
    for r in rows:
        ratio_ls_e = r['ls_efgp'] / r['ls_true']
        ratio_var_e = r['var_efgp'] / r['var_true']
        ratio_ls_s = r['ls_sp'] / r['ls_true']
        ratio_var_s = r['var_sp'] / r['var_true']
        print(f"{r['T']:>5}  {'EFGP':>14}  {r['t_efgp']:>5.1f}s  "
              f"{ratio_ls_e:>7.2f}x  {ratio_var_e:>8.2f}x  "
              f"{r['rmse_obs_efgp']:>10.4f}  {r['lat_rmse_efgp']:>10.4f}  "
              f"{r['drift_rmse_efgp']:>11.4f}")
        print(f"{r['T']:>5}  {'SparseGP':>14}  {r['t_sp']:>5.1f}s  "
              f"{ratio_ls_s:>7.2f}x  {ratio_var_s:>8.2f}x  "
              f"{r['rmse_obs_sp']:>10.4f}  {r['lat_rmse_sp']:>10.4f}  "
              f"{r['drift_rmse_sp']:>11.4f}")


if __name__ == "__main__":
    main()
