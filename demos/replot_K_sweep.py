"""Re-render the K-sweep figures with pooled drift accuracy (instead of
per-trial bars).  Reads from saved K{K_}.npz outputs and writes
``bench_*_K{K}_pooled.png`` alongside.

Usage:  python demos/replot_K_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_pooled(out_dir: Path, K_list, *, ls_true=None, var_true=None,
                    title_prefix: str, ls_init: float, var_init: float,
                    n_em: int = 50):
    for K_ in K_list:
        npz = out_dir / f"K{K_}.npz"
        if not npz.exists():
            print(f"  skipping {npz} — not found")
            continue
        d = np.load(npz, allow_pickle=True)
        wall_e = float(d['wall_efgp']); wall_25 = float(d['wall_sp25'])
        wall_49 = float(d['wall_sp49'])
        rel_e = float(d['rel_drift_rmse_efgp'])
        rel_25 = float(d['rel_drift_rmse_sp25'])
        rel_49 = float(d['rel_drift_rmse_sp49'])
        ls_e = list(d['ls_hist_efgp']); var_e = list(d['var_hist_efgp'])
        ls_25 = list(d['ls_hist_sp25']); var_25 = list(d['var_hist_sp25'])
        ls_49 = list(d['ls_hist_sp49']); var_49 = list(d['var_hist_sp49'])
        drift_scale = float(d['drift_scale'])

        has_49 = not (np.isnan(rel_49) or np.isnan(np.asarray(ls_49)).all())

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        if has_49:
            methods = ['EFGP-gmix', 'SparseGP M=25', 'SparseGP M=49']
            walls = [wall_e, wall_25, wall_49]
            rels = [rel_e, rel_25, rel_49]
            cols = ['C0', 'C1', 'C2']
        else:
            methods = ['EFGP-gmix', 'SparseGP M=25']
            walls = [wall_e, wall_25]
            rels = [rel_e, rel_25]
            cols = ['C0', 'C1']

        # wall
        ax = axes[0, 0]
        bars = ax.bar(methods, walls, color=cols, edgecolor='k')
        for b, w in zip(bars, walls):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f'{w:.0f}s', ha='center', va='bottom')
        ax.set_ylabel('wall clock (s)')
        ax.set_title(f'wall clock  (K={K_}, T={int(d["T"])}, n_em={n_em})')
        ax.grid(alpha=0.3, axis='y')

        # pooled drift RMSE (single bar per method)
        ax = axes[0, 1]
        bars = ax.bar(methods, rels, color=cols, edgecolor='k')
        for b, r in zip(bars, rels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f'{r:.3f}', ha='center', va='bottom')
        ax.axhline(1.0, color='k', linestyle=':', alpha=0.5,
                    label='zero-pred RMSE / drift_scale')
        ax.set_ylabel('rel drift RMSE  (pooled across trials, /drift_scale)')
        ax.set_title(f'drift accuracy along true latent paths  '
                      f'(drift_scale={drift_scale:.2f})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3, axis='y')

        # ℓ trajectory
        ax = axes[1, 0]
        iters = np.arange(0, len(ls_e) + 1)
        ax.plot(iters, [ls_init] + ls_e, '-o', label='EFGP-gmix',
                 color='C0', markersize=4)
        ax.plot(iters, [ls_init] + ls_25, '-s', label='SparseGP M=25',
                 color='C1', markersize=4)
        if has_49:
            ax.plot(iters, [ls_init] + ls_49, '-^', label='SparseGP M=49',
                     color='C2', markersize=4)
        if ls_true is not None:
            ax.axhline(ls_true, color='k', linestyle='--', alpha=0.5,
                        label=f'truth ℓ={ls_true}')
        ax.set_xlabel('EM iter'); ax.set_ylabel('ℓ')
        ax.set_title('lengthscale trajectory')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        # σ² trajectory
        ax = axes[1, 1]
        ax.plot(iters, [var_init] + var_e, '-o', label='EFGP-gmix',
                 color='C0', markersize=4)
        ax.plot(iters, [var_init] + var_25, '-s', label='SparseGP M=25',
                 color='C1', markersize=4)
        if has_49:
            ax.plot(iters, [var_init] + var_49, '-^', label='SparseGP M=49',
                     color='C2', markersize=4)
        if var_true is not None:
            ax.axhline(var_true, color='k', linestyle='--', alpha=0.5,
                        label=f'truth σ²={var_true}')
        ax.set_xlabel('EM iter'); ax.set_ylabel('σ²')
        ax.set_yscale('log')
        ax.set_title('output-scale trajectory  (log y)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3, which='both')

        suptitle = f'{title_prefix}  K={K_}  T={int(d["T"])}'
        if ls_true is not None:
            suptitle += f'  (ℓ_true={ls_true}, σ²_true={var_true})'
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        out = out_dir / f"bench_pooled_K{K_}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  saved: {out}")


def main():
    base = Path('/Users/colecitrenbaum/Documents/GPs/sing/demos')

    print("--- 2-well K-sweep ---")
    render_pooled(
        base / "_bench_2well_K_sweep_out",
        K_list=[1, 10, 30],
        ls_true=None, var_true=None,
        title_prefix="2-well drift  (out-of-model)",
        ls_init=0.3, var_init=1.0,
    )

    print("\n--- GP-drift K-sweep ---")
    render_pooled(
        base / "_bench_gpdrift_K_sweep_out",
        K_list=[1, 10, 30],
        ls_true=0.6, var_true=1.0,
        title_prefix="GP-drift  (in-model)",
        ls_init=0.3, var_init=1.0,
    )


if __name__ == "__main__":
    main()
