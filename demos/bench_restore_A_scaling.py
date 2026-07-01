"""Scaling benchmark for V restoration: dropped vs Hutchinson-NUFFT
(homogeneous-S) vs Hutchinson + gmix-gather (heterogeneous-S), across
three (K, T) sizes.

Same GP-drift-recovery setup as ``bench_restore_A_K10T500.py``;
sweeps over (K, T) ∈ {(2, 100), (5, 250), (10, 500)} to see how each
method scales with the source count N = K·(T-1).

Outputs a single timing/recovery table.
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
N_OBS = 6
OUT_DIR = Path(__file__).resolve().parent / "_bench_restore_A_scaling_out"
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
    return drift


def simulate_data(K, T, T_MAX, seed=42):
    drift_fn = gp_drift_factory(LS_TRUE, VAR_TRUE, jr.PRNGKey(seed))
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
    return drift_fn, xs_K, ys_K, x0_K, out_true


def fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid, *, mode, n_em, n_estep):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D, dtype=jnp.float64),
                                      (K_, 1, 1)))
    rho = jnp.linspace(0.05, 0.7, n_em)
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
        verbose=False, true_xs=np.asarray(xs_K),
        qx_moments_method='linearised_shim',
        restore_qf_variance=mode,
        qx_v_cg_tol=1e-3, qx_v_max_cg_iter=50, qx_v_n_probes=4,
    )
    return mp, hist, time.time() - t0


def main():
    sizes = [(2, 100, 1.6), (5, 250, 4.0), (10, 500, 8.0)]
    n_em_short = 3                # warm-up: amortise JIT compile
    n_em, n_estep = 15, 8         # measurement run: enough iters for steady state
    rows = []
    for K, T, T_MAX in sizes:
        N_eff = K * (T - 1)
        print(f"\n=== K={K} T={T} (N_eff={N_eff}) ===")
        drift_fn, xs_K, ys_K, x0_K, out_true = simulate_data(K, T, T_MAX, seed=42)
        t_grid = jnp.linspace(0, T_MAX, T, dtype=jnp.float64)

        for mode in ['none', 'hutch', 'hutch_hetS']:
            # Short warm-up run (JIT compile happens here; result discarded
            # except for the steady-state per-iter cost we'll subtract).
            _, _, wall_warm = fit_efgp(
                ys_K, xs_K, x0_K, out_true, t_grid,
                mode=mode, n_em=n_em_short, n_estep=n_estep)
            # Measurement run: long enough that JIT amortises.
            mp, hist, wall = fit_efgp(
                ys_K, xs_K, x0_K, out_true, t_grid,
                mode=mode, n_em=n_em, n_estep=n_estep)
            # Subtract per-iter * warm to get JIT-free per-iter rate
            # (assumes ~constant per-iter; warm has same JIT, fewer iters).
            per_iter = (wall - wall_warm) / max(n_em - n_em_short, 1)
            ls_final = float(hist.lengthscale[-1])
            var_final = float(hist.variance[-1])
            lat_final = float(hist.latent_rmse[-1])
            print(f"  {mode:7s}: wall={wall:6.1f}s  per-iter={per_iter:.3f}s  "
                  f"ℓ={ls_final:.4f}  σ²={var_final:.4f}  "
                  f"latent_rmse={lat_final:.4f}")
            rows.append((K, T, mode, wall, per_iter, ls_final,
                         var_final, lat_final))

    print("\n--- summary ---")
    print(f"{'K':>3} {'T':>4} {'mode':>7} {'wall(s)':>9} {'per-iter(s)':>12} "
          f"{'ℓ':>8} {'σ²':>8} {'lat_rmse':>10}")
    for K, T, mode, wall, pi, ls, var, lat in rows:
        print(f"{K:>3} {T:>4} {mode:>7} {wall:>9.1f} {pi:>12.3f} "
              f"{ls:>8.4f} {var:>8.4f} {lat:>10.4f}")

    # Per-iter ratios (JIT-amortised)
    print("\n--- per-iter ratios (relative to 'none', JIT amortised) ---")
    for K, T, _ in sizes:
        rows_kt = [r for r in rows if r[0] == K and r[1] == T]
        none_pi = [r[4] for r in rows_kt if r[2] == 'none'][0]
        hutch_pi = [r[4] for r in rows_kt if r[2] == 'hutch'][0]
        het_pi = [r[4] for r in rows_kt
                    if r[2] == 'hutch_hetS']
        het_str = (f"  hutch_hetS/none={het_pi[0]/max(none_pi, 1e-9):.2f}x"
                    if het_pi else "")
        print(f"  K={K} T={T} (N={K*(T-1)}): "
              f"hutch/none={hutch_pi/max(none_pi, 1e-9):.2f}x{het_str}")

    np.savez(OUT_DIR / 'scaling.npz',
             rows=np.array(rows, dtype=object),
             sizes=np.array(sizes))


if __name__ == '__main__':
    main()
