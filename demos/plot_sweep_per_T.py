"""
plot_sweep_per_T.py

Re-render the wall-time vs accuracy scatter from `bench_overnight_sweep.py`
.npz files, but with one panel per T (instead of one panel per benchmark).

Run:
  ~/myenv/bin/python demos/plot_sweep_per_T.py
"""
from __future__ import annotations
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

T_LIST = [400, 2000, 5000, 10000]
M_LIST = [25, 64, 144]
N_EM_LIST = [50, 100]


BENCH_COLORS = {
    'linear':     'tab:blue',
    'duffing':    'tab:red',
    'anharmonic': 'tab:green',
}
BENCH_FULL = {
    'linear': 'Damped rotation (linear)',
    'duffing': 'Duffing double-well',
    'anharmonic': 'Anharmonic oscillator',
}
SP_MARKER = {25: 'o', 64: 's', 144: '^'}
EFGP_MARKER = '*'


def load_all():
    """Return {n_em: {T: {short: dict}}}."""
    out = {n: {T: {} for T in T_LIST} for n in N_EM_LIST}
    for f in sorted(glob.glob('/tmp/sweep_*_T*_nem*.npz')):
        name = Path(f).stem  # sweep_<short>_T<T>_nem<nem>
        parts = name.split('_')
        short = parts[1]
        T = int(parts[2][1:])
        nem = int(parts[3][3:])
        if T not in T_LIST or nem not in N_EM_LIST:
            continue
        d = np.load(f, allow_pickle=True)
        rec = dict(short=short, T=T, nem=nem,
                    e_wall=float(d['e_wall']),
                    e_rmse=float(d['e_rel_mse']))
        for M in M_LIST:
            try:
                rec[f's{M}_wall'] = float(d[f's{M}_wall'])
                rec[f's{M}_rmse'] = float(d[f's{M}_rel_mse'])
            except KeyError:
                pass
        out[nem][T][short] = rec
    return out


def render_per_T(records, n_em, out_png):
    """1×len(T_LIST) panels, one per T.  Each panel: scatter of all
    benchmarks × methods at this (T, n_em).  NaN points are drawn at the
    top of the y-range with an 'x' marker so failures are visible."""
    fig, axes = plt.subplots(1, len(T_LIST), figsize=(6 * len(T_LIST), 6),
                              sharey=True)
    if len(T_LIST) == 1:
        axes = [axes]

    # Y-axis: pad above to leave room for NaN markers
    finite_rmse = []
    finite_walls = []
    for T in T_LIST:
        for r in records[T].values():
            for k in ['e_rmse'] + [f's{M}_rmse' for M in M_LIST]:
                v = r.get(k, np.nan)
                if np.isfinite(v):
                    finite_rmse.append(v)
            for k in ['e_wall'] + [f's{M}_wall' for M in M_LIST]:
                v = r.get(k, np.nan)
                if np.isfinite(v):
                    finite_walls.append(v)
    if finite_rmse:
        ymin = min(finite_rmse) * 0.7
        ymax_finite = max(finite_rmse) * 1.3
        nan_y = ymax_finite * 1.7
        ymax_disp = ymax_finite * 2.5
    else:
        ymin, ymax_finite, nan_y, ymax_disp = 1e-3, 1.0, 2.0, 3.0
    if finite_walls:
        xmin_disp = min(finite_walls) * 0.5
        xmax_disp = max(finite_walls) * 2.0
    else:
        xmin_disp, xmax_disp = 1.0, 1e4

    for col, T in enumerate(T_LIST):
        ax = axes[col]
        per_T = records[T]
        any_data = False
        nan_count = 0
        for short, r in per_T.items():
            color = BENCH_COLORS[short]
            # EFGP point
            e_w = r.get('e_wall', np.nan); e_r = r.get('e_rmse', np.nan)
            if np.isfinite(e_w):
                if np.isfinite(e_r):
                    ax.scatter(e_w, e_r, marker=EFGP_MARKER, s=240,
                                color=color, edgecolor='k',
                                linewidth=1.0, zorder=5)
                else:
                    ax.scatter(e_w, nan_y, marker='x', s=160,
                                color=color, linewidth=2.5, zorder=5)
                    nan_count += 1
                any_data = True
            for M in M_LIST:
                w = r.get(f's{M}_wall', np.nan)
                rmse = r.get(f's{M}_rmse', np.nan)
                if not np.isfinite(w):
                    continue
                if np.isfinite(rmse):
                    face = color if M == 64 else 'none'
                    ax.scatter(w, rmse, marker=SP_MARKER[M], s=130,
                                edgecolor=color, facecolor=face,
                                linewidths=1.6, zorder=4)
                else:
                    ax.scatter(w, nan_y, marker='x', s=160,
                                color=color, linewidth=2.5, zorder=4)
                    nan_count += 1
                any_data = True

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin_disp, xmax_disp)
        ax.set_ylim(ymin, ymax_disp)
        ax.axhline(nan_y, color='gray', linestyle=':', linewidth=0.7,
                    alpha=0.5)
        ax.text(xmin_disp * 1.1, nan_y * 1.05,
                f'NaN ({nan_count})', fontsize=9, color='gray',
                verticalalignment='bottom')
        if not any_data:
            ax.text(np.sqrt(xmin_disp * xmax_disp), np.sqrt(ymin * ymax_disp),
                    '(no data)', fontsize=11, color='gray',
                    ha='center', va='center')
        ax.set_xlabel('wall time (s)', fontsize=11)
        if col == 0:
            ax.set_ylabel('relative drift error  MSE(f) / Var(f)', fontsize=11)
        ax.set_title(f'T = {T}', fontsize=12)
        ax.grid(alpha=0.3, which='both')

    # Single legend on the right
    legend_elements = [
        Line2D([], [], marker=EFGP_MARKER, color='w', markerfacecolor='gray',
                markersize=15, markeredgecolor='k', label='EFGP'),
    ]
    for M in M_LIST:
        face = 'gray' if M == 64 else 'none'
        legend_elements.append(
            Line2D([], [], marker=SP_MARKER[M], color='gray',
                    markerfacecolor=face, markersize=10,
                    markeredgecolor='gray', linestyle='',
                    label=f'SparseGP M={M}'))
    legend_elements.append(
        Line2D([], [], marker='x', color='gray', markersize=10,
                linestyle='', label='NaN (run failed)'))
    for short in ['linear', 'duffing', 'anharmonic']:
        legend_elements.append(
            Line2D([], [], marker='o', color=BENCH_COLORS[short],
                    markersize=10, linestyle='', label=BENCH_FULL[short]))
    fig.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True)

    fig.suptitle(f'Wall-time vs relative drift error  (n_em={n_em})  '
                  '[oracle C frozen, lr=0.01, n_iters_m=4]', fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f'[plot] Saved {out_png}', flush=True)


def main():
    all_recs = load_all()
    for n_em in N_EM_LIST:
        n_data = sum(len(rs) for rs in all_recs[n_em].values())
        if n_data == 0:
            print(f'[plot] no data for n_em={n_em}, skipping')
            continue
        render_per_T(all_recs[n_em], n_em,
                      f'/tmp/sweep_time_vs_acc_byT_nem{n_em}.png')


if __name__ == '__main__':
    main()
