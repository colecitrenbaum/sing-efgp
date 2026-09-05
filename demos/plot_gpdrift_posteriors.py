"""
plot_gpdrift_posteriors.py

Diagnostic (not-in-paper) sanity-check figure: a grid of inferred latent
posteriors, rows = T, cols = method. Each panel overlays, per latent dim,
the true path xs_true[:,d] vs the inferred mean m_inf[:,d] with a +/-2 sigma
band from S_diag[:,d], over the (subsampled) time index.

Pure numpy + matplotlib (login-node friendly), tolerant of partial data.

  python demos/plot_gpdrift_posteriors.py --seed 0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


METHOD_ORDER = [('efgp', 0, 'EFGP'),
                ('sp', 49, 'SparseGP M=49'),
                ('sp', 100, 'SparseGP M=100')]
DIM_COLORS = ['C0', 'C3']
MAX_PTS = 2000  # subsample long trajectories for legibility


def load(out_dir, seed, x_key='T'):
    """Return (cells{(method,M,x): arrays}, sorted x values, ordered columns)."""
    cells = {}
    xs = set()
    cols = set()
    for path in sorted(glob.glob(str(Path(out_dir) / 'cell_*.npz'))):
        z = np.load(path, allow_pickle=True)
        if 'seed' in z.files and int(z['seed']) != seed:
            continue
        if 'status' in z.files and str(z['status']) != 'ok':
            continue
        if 'm_inf' not in z.files:
            continue
        method = str(z['method']); M = int(z['M']); x = int(z[x_key])
        cells[(method, M, x)] = dict(m_inf=z['m_inf'], S_diag=z['S_diag'],
                                     xs_true=z['xs_true'])
        xs.add(x); cols.add((method, M))
    # column order: efgp first, then sparsegp by ascending M
    order = sorted(cols, key=lambda mM: (mM[0] != 'efgp', mM[1]))
    label = {('efgp', 0): 'EFGP'}
    columns = [(m, M, label.get((m, M), f'SparseGP M={M}')) for (m, M) in order]
    return cells, sorted(xs), columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='demos/_bench_duffing_scaling_out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--x-key', choices=['T', 'K'], default='T')
    args = ap.parse_args()

    cells, Ts, columns = load(args.out_dir, args.seed, x_key=args.x_key)
    if not Ts:
        print("no completed cells with posteriors yet")
        return

    nrow, ncol = len(Ts), max(1, len(columns))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow),
                             squeeze=False)

    for r, T in enumerate(Ts):
        for c, (method, M, mlabel) in enumerate(columns):
            ax = axes[r][c]
            cell = cells.get((method, M, T))
            if cell is None:
                ax.text(0.5, 0.5, 'pending', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(mlabel, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{args.x_key}={T}", fontsize=10)
                continue

            m = np.asarray(cell['m_inf'])            # (T, D)
            sd = np.sqrt(np.maximum(np.asarray(cell['S_diag']), 0.0))
            xt = np.asarray(cell['xs_true'])          # (T, D)
            Tlen, D = m.shape
            step = max(1, Tlen // MAX_PTS)
            idx = np.arange(0, Tlen, step)

            for d in range(D):
                col = DIM_COLORS[d % len(DIM_COLORS)]
                ax.plot(idx, xt[idx, d], color=col, lw=0.8, alpha=0.5,
                        label=f'true x{d}' if (r == 0 and c == 0) else None)
                ax.plot(idx, m[idx, d], color=col, lw=1.0, ls='--',
                        label=f'inf x{d}' if (r == 0 and c == 0) else None)
                ax.fill_between(idx, m[idx, d] - 2 * sd[idx, d],
                                m[idx, d] + 2 * sd[idx, d],
                                color=col, alpha=0.15)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(mlabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{args.x_key}={T}\nlatent", fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel('time index', fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=8, ncol=2)
    fig.suptitle('GP-drift latent posteriors (mean +/- 2 sd vs truth)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = Path(args.out_dir) / 'gpdrift_posteriors.png'
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")


if __name__ == '__main__':
    main()
