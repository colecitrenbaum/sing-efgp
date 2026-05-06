"""GP-drift multi-trial benchmark: hyper recovery from K diverse trials.

Drift is a fresh sample from the SE-kernel GP prior at known
(ℓ_true, σ²_true).  Compare:

  * K=8 trajectories of length T from diverse x0 (uniform [-2, 2]^D)
  * K=1 trajectory of length K*T from a single x0

For the same total sample budget, K diverse trials are expected to
yield BETTER hyper / drift-RMSE recovery because a single long
trajectory often stays localized in one region and under-constrains
the GP at unexplored x.

Reports: ℓ recovery, σ² recovery, drift RMSE on a held-out grid,
per-trial latent RMSE, wall time.

Usage: python demos/bench_gpdrift_multitrial.py
"""
from __future__ import annotations

import math
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
LS_TRUE = 0.6
VAR_TRUE = 1.0
OUT_DIR = Path(__file__).resolve().parent / "_bench_gpdrift_multitrial_out"
OUT_DIR.mkdir(exist_ok=True)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _gp_drift_factory(ls_true, var_true, key, extent=4.0, eps_grid=1e-2):
    """Sample a fresh d-dim drift from the SE-kernel GP prior.

    Returns ``drift(x, t) -> R^D`` and a vectorised grid evaluator.
    """
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)

    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps                           # (D, M)

    def drift_x(x):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    def drift(x, t):
        return drift_x(x)

    def drift_grid(X_eval):
        outs = []
        for r in range(D):
            fr = jp.nufft2(X_eval, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            outs.append(fr)
        return jnp.stack(outs, axis=-1)

    return drift, drift_grid


def _eval_drift_recovery(ms, Ss, SSs, t_grid, ls, var, trial_mask,
                          gmix_fine_N, gmix_stencil_r, X_eval):
    """Eval q(f) drift posterior on X_eval."""
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(ls), jnp.log(var),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)

    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r))
    Ef_grid, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval, dtype=jnp.float32)[None],
        D_lat=D, D_out=D)
    return np.asarray(Ef_grid[0])


def _fit_and_report(ys_K, t_grid, ip_K, true_xs_K, label, n_em, n_estep,
                    drift_grid_fn):
    K, T, N = ys_K.shape
    print(f"\n  --- fit {label}  (K={K}, T={T}) ---")
    trial_mask = jnp.ones((K, T), dtype=bool)
    lik = GLik(ys_K, trial_mask)

    ys_flat = ys_K.reshape(-1, N)
    yc = ys_flat - ys_flat.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys_flat.mean(0), R=jnp.full((N,), 0.1))

    rho = jnp.linspace(0.05, 0.4, n_em)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip_K, latent_dim=D,
        # Deliberately wrong init; we want to see hyper recovery.
        lengthscale=1.5, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-2,
        estep_method='gmix', S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=5, mstep_lr=0.05,
        n_hutchinson_mstep=16, kernel_warmup_iters=3,
        pin_grid=True, pin_grid_lengthscale=0.5,
        verbose=False,
        true_xs=np.asarray(true_xs_K),
    )
    wall = time.time() - t0
    ls_final = hist.lengthscale[-1]
    var_final = hist.variance[-1]

    ms = np.asarray(mp['m'])
    per_trial_lat = np.sqrt(np.mean((ms - np.asarray(true_xs_K)) ** 2,
                                     axis=(1, 2)))
    print(f"    wall {wall:.1f}s  ℓ={ls_final:.3f} (true {LS_TRUE})  "
          f"σ²={var_final:.3f} (true {VAR_TRUE})")
    print(f"    per-trial latent RMSE: mean={per_trial_lat.mean():.3f}  "
          f"std={per_trial_lat.std():.3f}")

    # Drift RMSE on a held-out grid
    g = np.linspace(-3.0, 3.0, 20)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_pred = _eval_drift_recovery(
        jnp.asarray(ms), jnp.asarray(np.asarray(mp['S'])),
        jnp.asarray(np.asarray(mp['SS'])), t_grid,
        ls_final, var_final, trial_mask,
        gmix_fine_N=64, gmix_stencil_r=8, X_eval=X_eval)
    f_true = np.asarray(drift_grid_fn(jnp.asarray(X_eval, dtype=jnp.float32)))
    drift_rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))
    drift_scale = float(np.sqrt(np.mean(f_true ** 2)))
    print(f"    drift RMSE on [-3,3]² grid: {drift_rmse:.3f}  "
          f"(scale {drift_scale:.3f}, ratio {drift_rmse/drift_scale:.2f})")

    return dict(label=label, K=K, T=T, wall=wall,
                ls_final=ls_final, var_final=var_final,
                per_trial_lat=per_trial_lat,
                drift_rmse=drift_rmse, drift_scale=drift_scale,
                ls_history=list(hist.lengthscale),
                var_history=list(hist.variance),
                latent_rmse_history=list(hist.latent_rmse))


def main(K=8, T=128, t_max_per_trial=4.0, n_em=15, n_estep=10):
    print(f"\n=== GP-drift multi-trial bench ===")
    print(f"  GP truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}")
    print(f"  K={K} trials of length T={T} vs K=1 of length {K*T}")

    drift_fn, drift_grid_fn = _gp_drift_factory(
        LS_TRUE, VAR_TRUE, jr.PRNGKey(456))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)

    # ---- K diverse-x0 trials ----
    rng = np.random.default_rng(789)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, size=(K, D)).astype(np.float32))
    xs_K_list = []
    for k in range(K):
        xs_k = simulate_sde(jr.PRNGKey(9000 + k), x0=x0_K[k],
                             f=drift_fn, t_max=t_max_per_trial,
                             n_timesteps=T, sigma=sigma_fn)
        xs_K_list.append(jnp.clip(xs_k, -3.0, 3.0))
    xs_K = jnp.stack(xs_K_list, axis=0)

    N_obs = 6
    C_true = rng.standard_normal((N_obs, D)).astype(np.float32) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys_K_list = [simulate_gaussian_obs(jr.PRNGKey(10_000 + k),
                                         xs_K[k], out_true)
                  for k in range(K)]
    ys_K = jnp.stack(ys_K_list, axis=0)
    t_grid = jnp.linspace(0., t_max_per_trial, T)
    ip_K = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K, 1, 1)))

    # ---- K=1 baseline of length K*T from a single x0 ----
    T_long = K * T
    xs_1 = simulate_sde(jr.PRNGKey(9000), x0=x0_K[0],
                         f=drift_fn, t_max=K * t_max_per_trial,
                         n_timesteps=T_long, sigma=sigma_fn)
    xs_1 = jnp.clip(xs_1, -3.0, 3.0)[None]
    ys_1 = simulate_gaussian_obs(jr.PRNGKey(10_000), xs_1[0], out_true)[None]
    t_grid_long = jnp.linspace(0., K * t_max_per_trial, T_long)
    ip_1 = dict(mu0=x0_K[:1], V0=jnp.asarray([0.1 * jnp.eye(D)]))

    res_K = _fit_and_report(ys_K, t_grid, ip_K, xs_K,
                             label=f"K={K}, T={T}",
                             n_em=n_em, n_estep=n_estep,
                             drift_grid_fn=drift_grid_fn)
    res_1 = _fit_and_report(ys_1, t_grid_long, ip_1, xs_1,
                             label=f"K=1, T={T_long}",
                             n_em=n_em, n_estep=n_estep,
                             drift_grid_fn=drift_grid_fn)

    np.savez(OUT_DIR / "gpdrift_multitrial.npz",
             res_K=res_K, res_1=res_1)
    print(f"\n  ===== summary =====")
    print(f"  K={K}, T={T}            : "
          f"ℓ={res_K['ls_final']:.3f} (true {LS_TRUE})  "
          f"σ²={res_K['var_final']:.3f} (true {VAR_TRUE})  "
          f"drift_rmse={res_K['drift_rmse']:.3f}  "
          f"latent_rmse={res_K['per_trial_lat'].mean():.3f}  "
          f"wall {res_K['wall']:.1f}s")
    print(f"  K=1, T={T_long}: "
          f"ℓ={res_1['ls_final']:.3f}  "
          f"σ²={res_1['var_final']:.3f}  "
          f"drift_rmse={res_1['drift_rmse']:.3f}  "
          f"latent_rmse={res_1['per_trial_lat'].mean():.3f}  "
          f"wall {res_1['wall']:.1f}s")
    print(f"  saved: {OUT_DIR/'gpdrift_multitrial.npz'}")
    return res_K, res_1


if __name__ == "__main__":
    main()
