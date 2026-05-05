"""
Render an interim landscape plot from the dense_M sweep using the data
already in /tmp/dense_M.log (the script saves the .npz only at end).
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sing.simulate_data import simulate_sde


D = 2
T = 400
LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64, 144, 256]
ZS_LIM = 1.8
LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21
VAR_INIT = 1.0


# Final values parsed from /tmp/dense_M.log.  Per-iter trajectories not
# logged, so we'll plot the start (ls_init, VAR_INIT) and end points.
# Re-run script will save full trajectories to .npz.
RESULTS = {
    0.7: {
        'efgp': dict(ls_final=0.672, var_final=0.331, wall=44.3),
        'sp':   {25:  dict(ls_final=1.349, var_final=1.024, wall=54.9),
                  64:  dict(ls_final=0.854, var_final=0.396, wall=130.7),
                  144: dict(ls_final=0.820, var_final=0.370, wall=570.1),
                  256: dict(ls_final=0.815, var_final=0.367, wall=3132.9)},
    },
    3.0: {
        'efgp': dict(ls_final=0.902, var_final=0.817, wall=39.0),
        'sp':   {25:  dict(ls_final=1.817, var_final=2.258, wall=50.0)},
    },
}


def gt_landscape(xs, sigma_drift, t_grid, LOG_LS, LOG_VAR):
    import scipy.linalg
    xs_np = np.asarray(xs)
    inputs = xs_np[:-1]
    dt = float(np.asarray(t_grid[1] - t_grid[0]))
    velocities = (xs_np[1:] - xs_np[:-1]) / dt
    n_obs = xs_np.shape[0] - 1
    D_out = velocities.shape[1]
    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs ** 2).sum(-1)
    noise_var = sigma_drift ** 2 / dt
    eye_n = np.eye(n_obs)
    L_grid = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        ell = math.exp(float(ll))
        K_unsc = np.exp(-0.5 * sq_dists / (ell ** 2))
        for j, lv in enumerate(LOG_VAR):
            var_ = math.exp(float(lv))
            A_ = var_ * K_unsc + noise_var * eye_n
            try:
                L_chol = np.linalg.cholesky(A_)
            except np.linalg.LinAlgError:
                L_grid[i, j] = np.nan
                continue
            logdet = 2.0 * np.log(np.diag(L_chol)).sum()
            quad = 0.0
            for d in range(D_out):
                z = scipy.linalg.solve_triangular(
                    L_chol, velocities[:, d], lower=True)
                quad += float(z @ z)
            L_grid[i, j] = 0.5 * quad + 0.5 * D_out * logdet
    return L_grid


def main():
    t_max = 15.0
    sigma = 0.2
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    drift = lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]])
    xs = simulate_sde(jr.PRNGKey(13), x0=jnp.array([1.2, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3., 3.)
    t_grid = jnp.linspace(0., t_max, T)
    LOG_LS = np.linspace(*LOG_LS_RANGE, N_GRID)
    LOG_VAR = np.linspace(*LOG_VAR_RANGE, N_GRID)
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])

    L_norm = L_gt - L_gt[gb]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    levels = list(np.logspace(np.log10(max(finite.min(), finite.max()/1000)),
                                np.log10(finite.max()), 10))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    cmap_M = {25: 'C3', 64: 'C1', 144: 'C2', 256: 'C4'}

    for col, ls_init in enumerate(LS_INIT_LIST):
        ax = axes[col]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0]+levels,
                     cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                         colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        cell = RESULTS[ls_init]
        e = cell['efgp']
        ax.plot([math.log(ls_init), math.log(e['ls_final'])],
                 [math.log(VAR_INIT), math.log(e['var_final'])],
                 '-o', color='C0', markersize=8, linewidth=2.0,
                 label=f"EFGP [{e['wall']:.0f}s]  ℓ→{e['ls_final']:.3f}",
                 zorder=8)
        for M in M_LIST:
            sm = cell['sp'].get(M)
            if sm is None:
                continue
            ax.plot([math.log(ls_init), math.log(sm['ls_final'])],
                     [math.log(VAR_INIT), math.log(sm['var_final'])],
                     '-s', color=cmap_M[M], markersize=7, linewidth=1.4,
                     alpha=0.85,
                     label=f"SP M={M} [{sm['wall']:.0f}s]  ℓ→{sm['ls_final']:.3f}")
        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=9,
                    label=f"start (ℓ={ls_init})")
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                    edgecolor='k', zorder=10,
                    label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                           f"σ²={math.exp(gb_lv):.2f}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}" +
                      ("  (in progress)" if ls_init == 3.0 else ""),
                      fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(f"Duffing T={T} zs_lim={ZS_LIM} — start→end of M-step "
                  f"trajectories, per inducing-grid M",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/tmp/dense_M_landscape_partial.png', dpi=120)
    print('saved /tmp/dense_M_landscape_partial.png')


if __name__ == '__main__':
    main()
