"""
T1a: Nested K monotonicity at fixed q(x), fixed θ, **fixed h**.

The "more Fourier modes can only help" guarantee from EFGP: with the
same physical Fourier resolution h and the same kernel hypers, growing
the integer mode lattice {-K..K}^d should make E[f] converge
monotonically toward a limit. Earlier informal K sweeps in this repo
were non-monotone, but those rebuilt the grid (h(ℓ) varied with K via
``Lfreq/K``). Here we hold h fixed at the K_max grid's value and just
truncate the central (2K+1)^d modes for each smaller K.

If this sweep shows a monotone-decreasing distance to K_max for E[f]
and E[J_f], the EFGP q(f) block is correct. If it's non-monotone or
fails to settle, that's evidence of a real EFGP bug.

Run:
  ~/myenv/bin/python demos/diag_K_monotone_fixed_h.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# IMPORTANT: import jax_finufft via efgp_jax_primitives FIRST (load order)
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Gaussian
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
T = 300
SIGMA = 0.3
LS = 1.0
VAR = 1.0
N_EM = 12
N_ESTEP = 6
SEED = 7

K_LIST = [4, 6, 8, 12, 16, 24, 32]   # K_max = 32


def setup_problem():
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 3.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, out_true, t_max


def converge_smoother(lik, t_max):
    """Run SparseGP fixed-hyper EM to converge a shared q(x). Returns (ms, Ss, SSs)."""
    yc = lik.ys_obs[0] - lik.ys_obs[0].mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    output_params_init = dict(C=Vt[:D].T, d=lik.ys_obs[0].mean(0),
                              R=jnp.full((lik.ys_obs.shape[2],), 0.1))
    init_params = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params = dict(length_scales=jnp.full((D,), float(LS)),
                        output_scale=jnp.asarray(math.sqrt(VAR)))

    rho_sched = jnp.linspace(0.05, 0.4, N_EM)
    lr_sched = jnp.full((N_EM,), 0.05)
    t_grid = jnp.linspace(0., t_max, T)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params,
        init_params=init_params, output_params=output_params_init,
        sigma=SIGMA,
        rho_sched=rho_sched,
        n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=lr_sched,
        print_interval=999,
    )
    return mp, t_grid


def make_baseline_grid(K_max: int, t_max: float):
    """Build the baseline EFGP grid at K_max with the same X_template the test uses."""
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
        K_per_dim=K_max, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )
    return grid


def truncated_grid(baseline: jp.JaxGridState, K: int):
    """Build a JaxGridState with the central (2K+1)^d modes and the SAME h
    as `baseline`. ws is recomputed from (LS, VAR, h) so it's correct for
    the smaller lattice."""
    h_val = float(baseline.h_per_dim[0])
    h = jnp.asarray(h_val, dtype=baseline.h_per_dim.dtype)
    k_lat = jnp.arange(-K, K + 1, dtype=baseline.h_per_dim.dtype)
    xis_1d = h * k_lat
    grids = jnp.meshgrid(*[xis_1d for _ in range(D)], indexing="ij")
    xis_flat = jnp.stack([g.ravel() for g in grids], axis=-1)

    # ws = sqrt(S(ξ) * h^d) — closed-form SE
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_ls = jnp.log(jnp.asarray(LS, dtype=jnp.float32))
    log_var = jnp.log(jnp.asarray(VAR, dtype=jnp.float32))
    ls_sq = jnp.exp(2 * log_ls)
    log_S = (D / 2.0) * jnp.log(2 * math.pi * ls_sq) + log_var \
            - 2 * (math.pi ** 2) * ls_sq * xi_norm_sq
    ws_real = jnp.sqrt(jnp.exp(log_S) * (h_val ** D))
    ws = ws_real.astype(baseline.ws.dtype)

    mtot = tuple(2 * K + 1 for _ in range(D))
    M = (2 * K + 1) ** D
    h_per_dim = jnp.full((D,), h_val, dtype=baseline.h_per_dim.dtype)
    return jp.JaxGridState(xis_flat=xis_flat, ws=ws, h_per_dim=h_per_dim,
                            mtot_per_dim=mtot, xcen=baseline.xcen,
                            M=M, d=D)


def eval_drift_at_K(grid: jp.JaxGridState, mp, t_grid, X_eval):
    """Run compute_mu_r + drift_moments at the given grid; return (Ef, Edfdx)
    on the held-in evaluation grid X_eval."""
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D,
        cg_tol=1e-7, max_cg_iter=4 * grid.M,
    )
    Ef, _, Edfdx = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval), D_lat=D, D_out=D)
    return np.asarray(Ef), np.asarray(Edfdx)


def main():
    print(f"[T1a] JAX devices: {jax.devices()}", flush=True)
    print(f"[T1a] D={D}, T={T}, sigma={SIGMA}, ls={LS}, var={VAR}", flush=True)
    xs, ys, lik, _, t_max = setup_problem()

    print("\n[T1a] Converging shared smoother under SparseGP...", flush=True)
    mp, t_grid = converge_smoother(lik, t_max)
    ms_np = np.asarray(mp['m'][0])
    print(f"[T1a] smoother m std: {ms_np.std(axis=0)}", flush=True)

    K_max = max(K_LIST)
    print(f"\n[T1a] Building baseline grid at K_max={K_max}...", flush=True)
    baseline = make_baseline_grid(K_max, t_max)
    h_val = float(baseline.h_per_dim[0])
    print(f"[T1a] baseline h={h_val:.6f}, M={baseline.M}", flush=True)

    # Held-in eval grid (avoid the boundary)
    grid_lo = ms_np.min(0) + 0.2 * (ms_np.max(0) - ms_np.min(0))
    grid_hi = ms_np.max(0) - 0.2 * (ms_np.max(0) - ms_np.min(0))
    n_per = 10
    g0 = np.linspace(grid_lo[0], grid_hi[0], n_per)
    g1 = np.linspace(grid_lo[1], grid_hi[1], n_per)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)

    # Compute reference at K_max
    print(f"\n[T1a] Computing reference at K_max={K_max}...", flush=True)
    Ef_ref, Edfdx_ref = eval_drift_at_K(baseline, mp, t_grid, X_eval)
    scale_f = float(np.sqrt(np.mean(Ef_ref ** 2)))
    scale_J = float(np.sqrt(np.mean(Edfdx_ref ** 2)))
    print(f"[T1a] reference E[f] scale={scale_f:.4f}, E[J_f] scale={scale_J:.4f}",
          flush=True)

    # Sweep K
    print("\n[T1a] Sweeping K (fixed h=baseline):", flush=True)
    print(f"  {'K':>4}  {'M':>6}  {'||Ef-Ef_ref||':>14}  {'rel%':>6}  "
          f"{'||EJ-EJ_ref||':>14}  {'rel%':>6}", flush=True)
    rows = []
    for K in K_LIST:
        gK = truncated_grid(baseline, K)
        Ef_K, EJ_K = eval_drift_at_K(gK, mp, t_grid, X_eval)
        rms_f = float(np.sqrt(np.mean((Ef_K - Ef_ref) ** 2)))
        rms_J = float(np.sqrt(np.mean((EJ_K - Edfdx_ref) ** 2)))
        rel_f = 100 * rms_f / scale_f
        rel_J = 100 * rms_J / scale_J
        print(f"  {K:>4}  {gK.M:>6}  {rms_f:>14.6e}  {rel_f:>6.2f}  "
              f"{rms_J:>14.6e}  {rel_J:>6.2f}", flush=True)
        rows.append((K, gK.M, rms_f, rel_f, rms_J, rel_J))

    # Monotonicity check (excluding the K_max row which has zero distance)
    print("\n[T1a] Monotonicity check:", flush=True)
    rms_f_seq = [r[2] for r in rows[:-1]]   # drop K_max (== 0)
    rms_J_seq = [r[4] for r in rows[:-1]]
    f_diffs = np.diff(rms_f_seq)
    J_diffs = np.diff(rms_J_seq)
    f_violations = int(np.sum(f_diffs > 0.01 * scale_f))
    J_violations = int(np.sum(J_diffs > 0.01 * scale_J))
    print(f"  E[f]: monotone non-increasing in K? "
          f"{'YES' if f_violations == 0 else f'NO ({f_violations} 1%-violations)'}",
          flush=True)
    print(f"  E[J_f]: monotone non-increasing in K? "
          f"{'YES' if J_violations == 0 else f'NO ({J_violations} 1%-violations)'}",
          flush=True)
    if f_violations == 0 and J_violations == 0:
        print("[T1a] PASS: nested-K convergence is monotone at fixed h.", flush=True)
    else:
        print("[T1a] FAIL: nested-K convergence is non-monotone — "
              "investigate EFGP q(f) block.", flush=True)


if __name__ == "__main__":
    main()
