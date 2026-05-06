"""Duffing multi-trial benchmark: K trials of length T vs K=1 of length K*T.

Drift: f(x, y) = (x - x^3, -y)  (2-well in x, restoring in y, same as
``bench_2well_mc_vs_gmix.py``).

Goal: K diverse-x0 trials let q(f) see more of the input domain than
one long trajectory localized in one basin, so for the same total
sample budget K·T:

  1) latent RMSE per trial should be roughly comparable
  2) drift RMSE on a held-out grid should be NO WORSE under K trials,
     and frequently better when one long traj gets trapped in one well
  3) ℓ recovery should be robust across both regimes

Usage: python demos/bench_duffing_multitrial.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs

D = 2
SIGMA = 0.4
OUT_DIR = Path(__file__).resolve().parent / "_bench_duffing_multitrial_out"
OUT_DIR.mkdir(exist_ok=True)


def two_well_drift_jax(x, t):
    return jnp.array([x[0] - x[0] ** 3, -x[1]])


def two_well_drift_np(x_batch):
    x = np.asarray(x_batch)
    return np.column_stack([x[:, 0] - x[:, 0] ** 3, -x[:, 1]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _eval_drift(ms, Ss, SSs, t_grid, ls, var, trial_mask,
                method, gmix_fine_N=None, gmix_stencil_r=None,
                grid_pts=None):
    """Evaluate the q(f) drift posterior on grid_pts using the recovered
    (ℓ, σ²) hyperparameters.  Returns drift mean Ef on grid_pts (N_pts, D)."""
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(ls), jnp.log(var),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)

    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    if method == 'mc':
        mu_r, _, _ = jpd.compute_mu_r_jax(
            m_src, S_src, d_src, C_src, w_src, grid, jr.PRNGKey(99),
            sigma_drift_sq=SIGMA ** 2, S_marginal=2, D_lat=D, D_out=D)
    else:
        mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
            m_src, S_src, d_src, C_src, w_src, grid,
            sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
            fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r))

    # Evaluate on grid_pts (treat as a K=1 "trial" of length N_pts).
    Ef_grid, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts, dtype=jnp.float32)[None],
        D_lat=D, D_out=D)
    return np.asarray(Ef_grid[0])


def _fit_and_report(ys_K, t_grid, ip_K, x0_K, true_xs_K, label, n_em, n_estep,
                    learn_kernel=True):
    K, T, N = ys_K.shape
    print(f"\n  --- fit {label}  (K={K}, T={T}) ---")
    trial_mask = jnp.ones((K, T), dtype=bool)
    lik = GLik(ys_K, trial_mask)

    # Init output params via SVD of pooled observations.
    ys_flat = ys_K.reshape(-1, N)
    yc = ys_flat - ys_flat.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys_flat.mean(0), R=jnp.full((N,), 0.1))

    rho = jnp.linspace(0.05, 0.4, n_em)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip_K, latent_dim=D,
        lengthscale=0.3, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-2,
        estep_method='gmix', S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=learn_kernel, n_mstep_iters=5, mstep_lr=0.05,
        n_hutchinson_mstep=16, kernel_warmup_iters=3,
        pin_grid=True, pin_grid_lengthscale=0.5,
        verbose=False,
        true_xs=np.asarray(true_xs_K),
    )
    wall = time.time() - t0
    ls_final = hist.lengthscale[-1]
    var_final = hist.variance[-1]

    # Per-trial latent RMSE
    ms = np.asarray(mp['m'])                                 # (K, T, D)
    per_trial_lat = np.sqrt(np.mean((ms - np.asarray(true_xs_K)) ** 2,
                                     axis=(1, 2)))             # (K,)
    print(f"    wall {wall:.1f}s  ℓ={ls_final:.3f}  σ²={var_final:.3f}")
    print(f"    per-trial latent RMSE: mean={per_trial_lat.mean():.3f}  "
          f"std={per_trial_lat.std():.3f}  min={per_trial_lat.min():.3f}  "
          f"max={per_trial_lat.max():.3f}")

    # Drift RMSE on a fixed evaluation grid
    grid_pts = _build_grid_pts()
    f_true = two_well_drift_np(grid_pts)
    f_pred = _eval_drift(
        jnp.asarray(ms), jnp.asarray(np.asarray(mp['S'])),
        jnp.asarray(np.asarray(mp['SS'])), t_grid,
        ls_final, var_final, trial_mask,
        method='gmix', gmix_fine_N=64, gmix_stencil_r=8,
        grid_pts=grid_pts)
    drift_rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))
    drift_zero = float(np.sqrt(np.mean(f_true ** 2)))
    print(f"    drift RMSE on grid (Procrustes-free): {drift_rmse:.3f}  "
          f"(scale {drift_zero:.3f}, ratio {drift_rmse/drift_zero:.2f})")

    return dict(label=label, K=K, T=T, wall=wall,
                ls_final=ls_final, var_final=var_final,
                per_trial_lat=per_trial_lat, drift_rmse=drift_rmse,
                drift_zero=drift_zero,
                ls_history=list(hist.lengthscale),
                var_history=list(hist.variance),
                latent_rmse_history=list(hist.latent_rmse))


def _build_grid_pts():
    g = np.linspace(-2.0, 2.0, 16)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    return np.stack([GX.ravel(), GY.ravel()], axis=-1)


def main(K=8, T=512, t_max_per_trial=25.6, n_em=12, n_estep=10):
    """K diverse-x0 trials of length T vs single trajectory of length K*T."""
    print(f"\n=== Duffing multi-trial bench ===")
    print(f"  K={K} trials of length T={T}; baseline K=1 of length {K*T}")
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)

    # ---- K diverse-x0 trials ----
    rng = np.random.default_rng(123)
    x0_K = jnp.asarray(rng.normal(0.0, 1.5, size=(K, D)).astype(np.float32))
    print(f"  K-trial x0 spread: std={np.asarray(jnp.std(x0_K, axis=0))}")
    xs_K_list = []
    for k in range(K):
        xs_k = simulate_sde(jr.PRNGKey(7000 + k), x0=x0_K[k],
                             f=two_well_drift_jax, t_max=t_max_per_trial,
                             n_timesteps=T, sigma=sigma_fn)
        xs_K_list.append(jnp.clip(xs_k, -3.0, 3.0))
    xs_K = jnp.stack(xs_K_list, axis=0)                       # (K, T, D)

    N_obs = 8
    C_true = rng.standard_normal((N_obs, D)).astype(np.float32) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys_K_list = [simulate_gaussian_obs(jr.PRNGKey(8000 + k), xs_K[k], out_true)
                  for k in range(K)]
    ys_K = jnp.stack(ys_K_list, axis=0)                       # (K, T, N_obs)
    t_grid = jnp.linspace(0., t_max_per_trial, T)
    ip_K = dict(mu0=x0_K, V0=jnp.tile(0.3 * jnp.eye(D), (K, 1, 1)))

    # ---- K=1 of length K*T baseline ----
    print(f"\n  simulating baseline K=1, length {K*T}, x0 = trial-0's x0")
    T_long = K * T
    xs_1 = simulate_sde(jr.PRNGKey(7000), x0=x0_K[0],
                         f=two_well_drift_jax, t_max=K * t_max_per_trial,
                         n_timesteps=T_long, sigma=sigma_fn)
    xs_1 = jnp.clip(xs_1, -3.0, 3.0)[None]                    # (1, T_long, D)
    ys_1 = simulate_gaussian_obs(jr.PRNGKey(8000), xs_1[0], out_true)[None]
    t_grid_long = jnp.linspace(0., K * t_max_per_trial, T_long)
    ip_1 = dict(mu0=x0_K[:1], V0=jnp.asarray([0.3 * jnp.eye(D)]))

    # ---- Run both fits ----
    res_K = _fit_and_report(ys_K, t_grid, ip_K, x0_K, xs_K,
                             label=f"K={K}, T={T}",
                             n_em=n_em, n_estep=n_estep)
    res_1 = _fit_and_report(ys_1, t_grid_long, ip_1, x0_K[:1], xs_1,
                             label=f"K=1, T={T_long}",
                             n_em=n_em, n_estep=n_estep)

    # ---- Save and summary ----
    np.savez(OUT_DIR / "duffing_multitrial.npz",
             res_K=res_K, res_1=res_1)
    print(f"\n  ===== summary =====")
    print(f"  K={K}, T={T}            : "
          f"latent_rmse={res_K['per_trial_lat'].mean():.3f}  "
          f"drift_rmse={res_K['drift_rmse']:.3f}  "
          f"ℓ={res_K['ls_final']:.3f}  σ²={res_K['var_final']:.3f}  "
          f"wall {res_K['wall']:.1f}s")
    print(f"  K=1, T={T_long}: "
          f"latent_rmse={res_1['per_trial_lat'].mean():.3f}  "
          f"drift_rmse={res_1['drift_rmse']:.3f}  "
          f"ℓ={res_1['ls_final']:.3f}  σ²={res_1['var_final']:.3f}  "
          f"wall {res_1['wall']:.1f}s")
    print(f"  saved: {OUT_DIR/'duffing_multitrial.npz'}")
    return res_K, res_1


if __name__ == "__main__":
    main()
