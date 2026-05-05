"""
T18: Test the "EFGP just needs ~2× LR" hypothesis.

User observation: EFGP@lr=0.003 walks the same direction as SparseGP@lr=3e-3
but maybe 1/2 as fast.  Test EFGP at lr ∈ {0.004, 0.006, 0.008} (between
0.003 and 0.01 — where pure 0.01 starts oscillating).

If trajectory at lr ≈ 0.006 tracks SparseGP@3e-3 closely, the
mstep_lr-scaling hypothesis is confirmed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0
LS_INIT = 2.0
VAR_INIT = 0.3
N_EM = 100
LR_LIST = [0.004, 0.006, 0.008]


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
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
    fk_draw = grid.ws[None, :] * eps

    def drift(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)
    return drift


def setup():
    drift = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_efgp(lik, op, ip, t_grid, mstep_lr):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def main():
    print(f"[T18] EFGP lr-match-SparseGP test", flush=True)
    print(f"[T18] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)
    xs, ys, lik, op, ip, t_grid = setup()
    results = {}
    for lr in LR_LIST:
        print(f"\n[T18] EFGP @ mstep_lr={lr}, n_em={N_EM}...", flush=True)
        ls_t, var_t = fit_efgp(lik, op, ip, t_grid, lr)
        results[lr] = (ls_t, var_t)
        print(f"  Final: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}", flush=True)
        for chk in [10, 25, 50, 75, 100]:
            print(f"    iter {chk}: ℓ={ls_t[chk]:.3f}, σ²={var_t[chk]:.3f}",
                  flush=True)
    np.savez('/tmp/T18_lr_match.npz',
              **{f'lr_{lr}_ls': v[0] for lr, v in results.items()},
              **{f'lr_{lr}_var': v[1] for lr, v in results.items()})

    # Plot vs SparseGP@3e-3 + EFGP@0.003 (default warmup) + EFGP@0.01
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d_landscape = np.load('/tmp/T10_gp_drift_landscape.npz', allow_pickle=True)
        d11 = np.load('/tmp/T11_gp_drift_long.npz', allow_pickle=True)
        d14 = np.load('/tmp/T14_efgp_lr_sweep.npz', allow_pickle=True)
        LOG_LS = d_landscape['log_ls_grid']; LOG_VAR = d_landscape['log_var_grid']
        L_grid = d_landscape['L_grid']
        L_min = float(d_landscape['L_min'])
        ll_min = float(d_landscape['ll_min']); lv_min = float(d_landscape['lv_min'])

        sp_3e3 = d11['sp_lr0.003_nem100']
        efgp_003 = (d14['mstep_lr_0.003_ls'], d14['mstep_lr_0.003_var'])
        efgp_01 = (d14['mstep_lr_0.01_ls'], d14['mstep_lr_0.01_var'])

        fig, ax = plt.subplots(figsize=(11, 7.5))
        L_norm = L_grid - L_min
        levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                          levels=[0] + levels, cmap='viridis_r', extend='max')
        fig.colorbar(cf, ax=ax, label='L − L_min')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                        levels=levels, colors='k', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
        for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ax.plot(LOG_LS, LOG_LS * 2 + log_alpha_show,
                    color='white', linestyle=':', linewidth=0.7, alpha=0.5)

        # Plot EFGP at all lr values + SparseGP
        ax.plot(np.log(efgp_003[0]), np.log(efgp_003[1]),
                '-o', color='C0', markersize=2, linewidth=1.0, alpha=0.7,
                label='EFGP @ lr=0.003')
        for lr in LR_LIST:
            ls_t, var_t = results[lr]
            ax.plot(np.log(ls_t), np.log(var_t), '-o',
                    markersize=3, linewidth=1.4, alpha=0.85,
                    label=f'EFGP @ lr={lr} (NEW)')
        ax.plot(np.log(efgp_01[0]), np.log(efgp_01[1]),
                '-o', color='C5', markersize=2, linewidth=1.0, alpha=0.5,
                label='EFGP @ lr=0.01 (oscillates)')
        ax.plot(np.log(sp_3e3[0]), np.log(sp_3e3[1]),
                '-s', color='C3', markersize=3, linewidth=1.6,
                label='SparseGP @ lr=3e-3 (target)')

        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6,
                    label='init')
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                    marker='X', s=240, color='red', edgecolor='k', zorder=6,
                    label='TRUTH')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=5,
                    label='loss min')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_title('Does EFGP at higher lr match SparseGP@3e-3 trajectory?')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig('/tmp/T18_lr_match.png', dpi=140)
        print(f"\n[T18] Saved /tmp/T18_lr_match.png", flush=True)
    except Exception as e:
        print(f"[T18] Plot failed: {e}", flush=True)


if __name__ == '__main__':
    main()
