"""
render_duffx64.py

Re-render the duffx64 sweep plots from the saved .npz files (in case
the in-process render crashed).  Recomputes the GT log-marginal
landscape per T (cheap, ~20s).

Run:
  ~/myenv/bin/python demos/render_duffx64.py
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
from pathlib import Path
import glob

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.random as jr

from sing.simulate_data import simulate_sde


D = 2
T_BASE = 400
T_MAX_BASE = 15.0
SIGMA = 0.2
SEED = 13
X0 = jnp.array([1.2, 0.0])
DRIFT = lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]])

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21
LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64]
VAR_INIT = 1.0


def make_xs(T):
    t_max = T_MAX_BASE * (T / T_BASE)
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(SEED), x0=X0, f=DRIFT, t_max=t_max,
                      n_timesteps=T, sigma=sigma_fn)
    return np.clip(np.asarray(xs), -3., 3.), t_max


def gt_landscape(xs, t_grid, LOG_LS, LOG_VAR):
    import scipy.linalg
    T_ = xs.shape[0]
    inputs = xs[:-1]
    dt = float(np.asarray(t_grid[1] - t_grid[0]))
    velocities = (xs[1:] - xs[:-1]) / dt
    n_obs = T_ - 1
    D_out = velocities.shape[1]
    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs ** 2).sum(-1)
    noise_var = SIGMA ** 2 / dt
    eye_n = np.eye(n_obs)
    L = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        ell = math.exp(float(ll))
        K_unsc = np.exp(-0.5 * sq_dists / (ell ** 2))
        for j, lv in enumerate(LOG_VAR):
            var_ = math.exp(float(lv))
            A = var_ * K_unsc + noise_var * eye_n
            try:
                Lc = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                L[i, j] = np.nan
                continue
            ld = 2.0 * np.log(np.diag(Lc)).sum()
            q = 0.0
            for d in range(D_out):
                z = scipy.linalg.solve_triangular(Lc, velocities[:, d], lower=True)
                q += float(z @ z)
            L[i, j] = 0.5 * q + 0.5 * D_out * ld
    return L


def render_landscape_per_T(T, npz_files, out_png):
    """Landscape + trajectories for one T, one panel per ls_init."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs, t_max = make_xs(T)
    t_grid = np.linspace(0., t_max, T)
    LOG_LS = np.linspace(*LOG_LS_RANGE, N_GRID)
    LOG_VAR = np.linspace(*LOG_VAR_RANGE, N_GRID)
    print(f"  computing GT landscape T={T}...", flush=True)
    L_gt = gt_landscape(xs, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}",
          flush=True)
    L_norm = L_gt - L_gt[gb]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        levels = list(np.logspace(np.log10(max(lo, hi/1000)),
                                    np.log10(hi), 10))
    else:
        levels = [1, 10, 100]

    n = len(LS_INIT_LIST)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]
    by_ls = {}
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=False))
        by_ls[float(d['ls_init'])] = d

    for col, ls_init in enumerate(LS_INIT_LIST):
        ax = axes[col]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0]+levels,
                     cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                         colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        if ls_init in by_ls:
            d = by_ls[ls_init]
            e_ls_t = np.asarray(d['e_ls_traj'])
            e_var_t = np.asarray(d['e_var_traj'])
            ax.plot(np.log(e_ls_t), np.log(e_var_t),
                     '-o', color='C0', markersize=3, linewidth=1.6,
                     label=f"EFGP [{float(d['e_wall']):.0f}s, "
                            f"drift={float(d['e_rmse_pc']):.2f}, "
                            f"lat={float(d['e_lat_pc']):.3f}]")
            cmap = ['C3', 'C1']
            for i, M in enumerate(M_LIST):
                if f's{M}_ls_traj' in d:
                    s_ls_t = np.asarray(d[f's{M}_ls_traj'])
                    s_var_t = np.asarray(d[f's{M}_var_traj'])
                    if not np.all(np.isnan(s_ls_t)):
                        ax.plot(np.log(s_ls_t), np.log(s_var_t),
                                 '-s', color=cmap[i], markersize=3,
                                 linewidth=1.4,
                                 label=f"SP M={M} "
                                        f"[{float(d[f's{M}_wall']):.0f}s, "
                                        f"drift={float(d[f's{M}_rmse_pc']):.2f}, "
                                        f"lat={float(d[f's{M}_lat_pc']):.3f}]")
        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6)
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                    edgecolor='k', zorder=7,
                    label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                           f"σ²={math.exp(gb_lv):.2f}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}", fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(f"Duffing T={T} (float64) — GT log-marginal landscape "
                  f"+ EM trajectories per ls_init", fontsize=13)
    import matplotlib.pyplot as plt2
    plt2.tight_layout(rect=[0, 0, 1, 0.95])
    plt2.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def render_scatter_per_T(T, npz_files, out_png):
    """Wall vs (drift_pc, latent_pc) scatter for one T."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    init_colors = {0.7: 'tab:blue', 3.0: 'tab:orange'}
    sp_marker = {25: 'o', 64: 's'}

    by_ls = {}
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=False))
        by_ls[float(d['ls_init'])] = d

    for ax_i, (ax, metric) in enumerate(zip(axes,
            ['drift', 'latent'])):
        for ls_init in LS_INIT_LIST:
            if ls_init not in by_ls:
                continue
            d = by_ls[ls_init]
            color = init_colors[ls_init]
            # EFGP
            e_w = float(d['e_wall'])
            e_v = float(d[f'e_{"rmse" if metric=="drift" else "lat"}_pc'])
            if math.isfinite(e_w) and math.isfinite(e_v):
                ax.scatter(e_w, e_v, marker='*', s=220,
                            color=color, edgecolor='k',
                            linewidth=0.8, zorder=5)
            for M in M_LIST:
                w = float(d[f's{M}_wall']) if f's{M}_wall' in d else np.nan
                v = (float(d[f's{M}_rmse_pc'])
                     if metric == 'drift'
                     else float(d[f's{M}_lat_pc']))
                if math.isfinite(w) and math.isfinite(v):
                    face = color if M == 64 else 'none'
                    ax.scatter(w, v, marker=sp_marker[M], s=130,
                                edgecolor=color, facecolor=face,
                                linewidths=1.5, zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('wall time (s)', fontsize=11)
        ax.set_ylabel('drift rmse / Var(f)' if metric == 'drift'
                       else 'Procrustes-aligned latent RMSE', fontsize=11)
        ax.set_title(metric.title(), fontsize=12)
        ax.grid(alpha=0.3, which='both')

    legend_elements = [
        Line2D([], [], marker='*', color='w', markerfacecolor='gray',
                markersize=15, markeredgecolor='k', label='EFGP'),
        Line2D([], [], marker='o', color='gray', markerfacecolor='none',
                markersize=10, linestyle='', markeredgecolor='gray',
                label='SP M=25'),
        Line2D([], [], marker='s', color='gray', markerfacecolor='gray',
                markersize=10, linestyle='', markeredgecolor='gray',
                label='SP M=64'),
    ]
    for ls_init in LS_INIT_LIST:
        legend_elements.append(
            Line2D([], [], marker='o', color=init_colors[ls_init],
                    markersize=10, linestyle='',
                    label=f'ls_init={ls_init}'))
    fig.legend(handles=legend_elements, loc='center right',
                bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle(f"Duffing T={T} (float64) — wall time vs accuracy",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def render_table(npz_files, T):
    print(f"\n{'='*100}")
    print(f"T = {T}  (Duffing, float64)")
    print(f"{'='*100}")
    print(f"  {'ls_init':>7s}  {'method':>9s}  {'ℓ_final':>8s}  "
          f"{'σ²_final':>9s}  {'drift_pc':>9s}  {'drift_raw':>10s}  "
          f"{'lat_pc':>7s}  {'lat_raw':>8s}  {'wall':>7s}")
    by_ls = {}
    for f in npz_files:
        d = dict(np.load(f, allow_pickle=False))
        by_ls[float(d['ls_init'])] = d
    for ls_init in LS_INIT_LIST:
        if ls_init not in by_ls:
            continue
        d = by_ls[ls_init]
        e_ls = float(d['e_ls_traj'][-1]) if d['e_ls_traj'].size else np.nan
        e_var = float(d['e_var_traj'][-1]) if d['e_var_traj'].size else np.nan
        print(f"  {ls_init:>7.1f}  {'EFGP':>9s}  {e_ls:>8.3f}  "
              f"{e_var:>9.3f}  {float(d['e_rmse_pc']):>9.4f}  "
              f"{float(d['e_rmse_raw']):>10.4f}  "
              f"{float(d['e_lat_pc']):>7.4f}  {float(d['e_lat_raw']):>8.4f}  "
              f"{float(d['e_wall']):>6.1f}s")
        for M in M_LIST:
            if f's{M}_wall' not in d:
                continue
            s_ls = (float(d[f's{M}_ls_traj'][-1])
                    if d[f's{M}_ls_traj'].size else np.nan)
            s_var = (float(d[f's{M}_var_traj'][-1])
                     if d[f's{M}_var_traj'].size else np.nan)
            print(f"  {ls_init:>7.1f}  {f'SP M={M}':>9s}  "
                  f"{s_ls:>8.3f}  {s_var:>9.3f}  "
                  f"{float(d[f's{M}_rmse_pc']):>9.4f}  "
                  f"{float(d[f's{M}_rmse_raw']):>10.4f}  "
                  f"{float(d[f's{M}_lat_pc']):>7.4f}  "
                  f"{float(d[f's{M}_lat_raw']):>8.4f}  "
                  f"{float(d[f's{M}_wall']):>6.1f}s")


def main():
    for T in [2000, 5000]:
        files = sorted(glob.glob(f'/tmp/duffx64_T{T}_ls*.npz'))
        if not files:
            print(f"No data for T={T} yet, skipping.", flush=True)
            continue
        print(f"\n=== Rendering T={T} from {len(files)} npz files ===",
              flush=True)
        render_table(files, T)
        render_landscape_per_T(T, files,
                                f'/tmp/duffx64_landscape_T{T}.png')
        render_scatter_per_T(T, files,
                              f'/tmp/duffx64_scatter_T{T}.png')


if __name__ == '__main__':
    main()
