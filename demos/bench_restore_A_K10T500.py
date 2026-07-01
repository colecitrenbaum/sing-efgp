"""End-to-end EFGP-SING fit comparison: drop A vs restore A via the
batched Hutchinson + NUFFT-2 estimator (efgp_qx_v_hutch.py).

Same K=10, T=500 GP-drift-recovery setup as
``bench_gmix_live_vs_shim_K10T500.py``, two fits:

* baseline: legacy linearised + custom-VJP shim, V dropped (Approx.~A
  active).
* with-A hutch: ``restore_qf_variance='hutch'`` -- precompute
  $\\hat\\omega_r(\\delta)$ via Hutchinson + FFT cross-correlation
  once per outer iter, then a single batched Type-2 NUFFT (with AD-driven
  adjoint for $\\nabla V$) gives $(V_i, \\nabla V_i)$ at all sources.
  Custom-VJP shim injects them per-transition. See
  ``efgp_estep.tex`` §6.

Outputs
-------
1. Learning-curve plot: per-EM-iter ℓ, σ², and per-trial drift RMSE for
   both fits, with truth horizontal lines.
2. Wall-clock timings (per-iter and total).
3. Final drift-RMSE summary along true latent paths.
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
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
SIGMA = 0.4
LS_TRUE = 0.6
VAR_TRUE = 1.0
LS_INIT = 1.5
VAR_INIT = 1.0
K = 10
T = 500
T_MAX = 8.0
N_OBS = 6
OUT_DIR = Path(__file__).resolve().parent / "_bench_restore_A_out"
OUT_DIR.mkdir(exist_ok=True)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def gp_drift_factory(ls, var, key, extent=4.0, eps_grid=1e-2):
    X_template = jnp.linspace(-extent, extent, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
        / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps

    def drift(x, t):
        x_b = x[None, :]
        return jnp.stack([
            jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real[0]
            for r in range(D)
        ])

    def drift_grid(X_eval):
        return jnp.stack([
            jp.nufft2(X_eval, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real
            for r in range(D)
        ], axis=-1)

    return drift, drift_grid


def simulate_data(seed=42):
    drift_fn, drift_grid_fn = gp_drift_factory(
        LS_TRUE, VAR_TRUE, jr.PRNGKey(seed))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    rng = np.random.default_rng(seed + 1)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, size=(K, D)).astype(np.float64))
    xs_list = []
    for k in range(K):
        xs_k = simulate_sde(jr.PRNGKey(seed + 100 + k), x0=x0_K[k],
                             f=drift_fn, t_max=T_MAX, n_timesteps=T,
                             sigma=sigma_fn)
        xs_list.append(jnp.clip(xs_k, -3.5, 3.5))
    xs_K = jnp.stack(xs_list, axis=0).astype(jnp.float64)
    C_true = rng.standard_normal((N_OBS, D)).astype(np.float64) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS, dtype=jnp.float64),
                    R=jnp.full((N_OBS,), 0.05, dtype=jnp.float64))
    ys_list = [simulate_gaussian_obs(jr.PRNGKey(seed + 200 + k),
                                       xs_K[k], out_true) for k in range(K)]
    ys_K = jnp.stack(ys_list, axis=0).astype(jnp.float64)
    return drift_fn, drift_grid_fn, xs_K, ys_K, x0_K, out_true


def fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid, *,
              restore_qf_variance, n_em, n_estep):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D, dtype=jnp.float64),
                                      (K_, 1, 1)))
    rho = jnp.linspace(0.05, 0.7, n_em)
    label = (f"restore_qf_variance={restore_qf_variance!r}"
             if restore_qf_variance != 'none' else 'BASELINE')
    print(f"  [EFGP / {label}] init ℓ={LS_INIT}, σ²={VAR_INIT}; "
          f"true ℓ={LS_TRUE}, σ²={VAR_TRUE}")
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=out_true, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
        estep_method='gmix', S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        pin_grid=True, pin_grid_lengthscale=LS_TRUE * 0.75,
        verbose=True, true_xs=np.asarray(xs_K),
        qx_moments_method='linearised_shim',
        restore_qf_variance=restore_qf_variance,
        qx_v_cg_tol=1e-3, qx_v_max_cg_iter=50,  # cheap CG for benchmark
    )
    return mp, hist, time.time() - t0


def _eval_drift_at_truth(mp, hist, t_grid, xs_K, trial_mask):
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    ms = jnp.asarray(np.asarray(mp['m']))
    Ss = jnp.asarray(np.asarray(mp['S']))
    SSs = jnp.asarray(np.asarray(mp['SS']))
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D))
    pin_ls = LS_TRUE * 0.75
    grid = jp.spectral_grid_se(pin_ls, var, X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(ls), jnp.log(var),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)
    Ef_per_t, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(xs_K, dtype=jnp.float64),
        D_lat=D, D_out=D)
    return np.asarray(Ef_per_t)


def main():
    drift_fn, _, xs_K, ys_K, x0_K, out_true = simulate_data(seed=42)
    print(f"data: K={K}, T={T}, D={D}, T_MAX={T_MAX}, N_OBS={N_OBS}")
    t_grid = jnp.linspace(0, T_MAX, T, dtype=jnp.float64)
    trial_mask = jnp.ones((K, T), dtype=bool)
    n_em, n_estep = 30, 10

    print("\n=== BASELINE (Approximation A active: V dropped) ===")
    mp_b, hist_b, wall_b = fit_efgp(
        ys_K, xs_K, x0_K, out_true, t_grid,
        restore_qf_variance='none', n_em=n_em, n_estep=n_estep)

    print("\n=== restore_qf_variance='hutch' (V via local-quadratic Taylor) ===")
    mp_v, hist_v, wall_v = fit_efgp(
        ys_K, xs_K, x0_K, out_true, t_grid,
        restore_qf_variance='hutch', n_em=n_em, n_estep=n_estep)

    f_true_K = jnp.stack([
        jnp.stack([drift_fn(xs_K[k, t], t_grid[t]) for t in range(T)], axis=0)
        for k in range(K)
    ], axis=0)
    f_true = np.asarray(f_true_K)
    print("\nevaluating drift at true latent paths...")
    f_b = _eval_drift_at_truth(mp_b, hist_b, t_grid, xs_K, trial_mask)
    f_v = _eval_drift_at_truth(mp_v, hist_v, t_grid, xs_K, trial_mask)
    drift_rmse_b = float(np.sqrt(((f_b - f_true) ** 2).mean()))
    drift_rmse_v = float(np.sqrt(((f_v - f_true) ** 2).mean()))

    print("\n--- summary ---")
    print(f"  wall:  baseline = {wall_b:.1f}s   restore_V = {wall_v:.1f}s   "
          f"({wall_v / wall_b:.2f}x)")
    print(f"  final ℓ:    baseline = {hist_b.lengthscale[-1]:.4f}   "
          f"restore_V = {hist_v.lengthscale[-1]:.4f}   (true = {LS_TRUE})")
    print(f"  final σ²:   baseline = {hist_b.variance[-1]:.4f}   "
          f"restore_V = {hist_v.variance[-1]:.4f}   (true = {VAR_TRUE})")
    print(f"  drift RMSE: baseline = {drift_rmse_b:.4f}   "
          f"restore_V = {drift_rmse_v:.4f}")

    iters = np.arange(1, len(hist_b.lengthscale) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    ax = axes[0]
    ax.plot(iters, hist_b.lengthscale, 'o-', label='baseline (V dropped)',
            color='C0')
    ax.plot(np.arange(1, len(hist_v.lengthscale) + 1), hist_v.lengthscale,
            's--', label='restore V', color='C2')
    ax.axhline(LS_TRUE, color='k', linestyle=':', label=f'truth = {LS_TRUE}')
    ax.set_xlabel('EM iter'); ax.set_ylabel('lengthscale ℓ')
    ax.set_title('ℓ recovery'); ax.legend()
    ax = axes[1]
    ax.plot(iters, hist_b.variance, 'o-', label='baseline (V dropped)',
            color='C0')
    ax.plot(np.arange(1, len(hist_v.variance) + 1), hist_v.variance,
            's--', label='restore V', color='C2')
    ax.axhline(VAR_TRUE, color='k', linestyle=':', label=f'truth = {VAR_TRUE}')
    ax.set_xlabel('EM iter'); ax.set_ylabel('variance σ²')
    ax.set_title('σ² recovery'); ax.legend()
    ax = axes[2]
    if hist_b.latent_rmse:
        ax.plot(np.arange(1, len(hist_b.latent_rmse) + 1),
                 hist_b.latent_rmse, 'o-',
                 label='baseline (V dropped)', color='C0')
    if hist_v.latent_rmse:
        ax.plot(np.arange(1, len(hist_v.latent_rmse) + 1),
                 hist_v.latent_rmse, 's--',
                 label='restore V', color='C2')
    ax.set_xlabel('EM iter'); ax.set_ylabel('latent RMSE')
    ax.set_title(f'latent recovery (drift RMSE: '
                  f'{drift_rmse_b:.3f} vs {drift_rmse_v:.3f})')
    ax.legend()
    fig.suptitle(f'Approximation A test: V dropped vs V restored '
                 f'— K={K}, T={T}, ratio={wall_v / wall_b:.1f}x', fontsize=11)
    plt.tight_layout()
    out_png = OUT_DIR / 'learning_curves.png'
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved plot: {out_png}")
    np.savez(OUT_DIR / 'history.npz',
             ls_b=np.array(hist_b.lengthscale),
             ls_v=np.array(hist_v.lengthscale),
             var_b=np.array(hist_b.variance),
             var_v=np.array(hist_v.variance),
             latent_rmse_b=np.array(hist_b.latent_rmse),
             latent_rmse_v=np.array(hist_v.latent_rmse),
             drift_rmse_b=drift_rmse_b, drift_rmse_v=drift_rmse_v,
             wall_b=wall_b, wall_v=wall_v,
             ls_true=LS_TRUE, var_true=VAR_TRUE,
             K=K, T=T, n_em=n_em, n_estep=n_estep)
    print(f"Saved history: {OUT_DIR / 'history.npz'}")


if __name__ == '__main__':
    main()
