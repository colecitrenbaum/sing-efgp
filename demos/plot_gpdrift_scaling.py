"""
plot_gpdrift_scaling.py

Paper subfigure for the GP-drift scaling law: two vertically stacked panels
sharing a log-T x-axis. Top = wall-clock time; bottom = drift error
(Procrustes-aligned rel MSE). Three curves: EFGP, SparseGP M=49, SparseGP M=100.

Pure numpy + matplotlib (no jax/sing import) so it is light enough to run on
the login node. Tolerant of partial data -- regenerate every poll as cells land.
Failed cells are skipped (and noted).

  python demos/plot_gpdrift_scaling.py
  python demos/plot_gpdrift_scaling.py --seed 0 \
      --out-dir demos/_bench_gpdrift_scaling_out
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


# (method, M) -> (label, color, marker)
SERIES = {
    ('efgp', 0):   ('EFGP-SING',        'C0', '*'),
    ('sp', 49):    ('SparseGP M=49',    'C1', 's'),
    ('sp', 100):   ('SparseGP M=100',   'C3', 'o'),
    ('sp', 400):   ('SparseGP M=400',   'C2', 'D'),
}


def _scalar(z, key, default=np.nan):
    return float(z[key]) if key in z.files else default


def _drift(z):
    """Primary drift metric = NRMSE at visited states; fall back to
    sqrt(rel_mse) for older npz that only stored drift_rel_mse."""
    if 'drift_nrmse' in z.files:
        return _scalar(z, 'drift_nrmse')
    return float(np.sqrt(_scalar(z, 'drift_rel_mse')))


def load_cells(out_dir, seed, x_key='T'):
    """Return {(method, M): {'ok': [(x, wall, drift)...], 'failed': [x...]}},
    where x is the swept variable (T for the seq-len sweep, K for the trial
    sweep)."""
    series = {k: {'ok': [], 'failed': []} for k in SERIES}
    for path in sorted(glob.glob(str(Path(out_dir) / 'cell_*.npz'))):
        z = np.load(path, allow_pickle=True)
        if 'seed' in z.files and int(z['seed']) != seed:
            continue
        method = str(z['method'])
        M = int(z['M']) if 'M' in z.files else 0
        key = (method, M)
        if key not in series:
            continue
        status = str(z['status']) if 'status' in z.files else 'ok'
        x = int(z[x_key])
        if status == 'ok':
            series[key]['ok'].append((x, _scalar(z, 'wall'), _drift(z)))
        else:
            series[key]['failed'].append(x)
    for key in series:
        series[key]['ok'].sort()
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='demos/_bench_duffing_scaling_out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--x-key', choices=['T', 'K'], default='T',
                    help='swept variable: T (seq length) or K (num trials)')
    args = ap.parse_args()

    series = load_cells(args.out_dir, args.seed, x_key=args.x_key)
    x_label = ('T (time points)' if args.x_key == 'T'
               else 'K (num trials, T=1000 each)')

    fig, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(4.0, 5.0), sharex=True)

    any_pts = False
    for key, (label, color, marker) in SERIES.items():
        ok = series[key]['ok']
        if ok:
            Ts = [r[0] for r in ok]
            walls = [r[1] for r in ok]
            drifts = [r[2] for r in ok]
            ax_w.plot(Ts, walls, marker=marker, color=color, label=label,
                      markersize=6, linewidth=1.5)
            ax_d.plot(Ts, drifts, marker=marker, color=color, label=label,
                      markersize=6, linewidth=1.5)
            any_pts = True
        # annotate failed cells on the wall panel
        for T in series[key]['failed']:
            ax_w.scatter([T], [np.nan])  # keeps axis autoscale sane
            ax_w.annotate('OOM/fail', xy=(T, 1), xycoords=('data', 'axes fraction'),
                          ha='center', va='top', fontsize=6, color=color,
                          rotation=90)

    ax_w.set_xscale('log')
    ax_w.set_yscale('log')
    ax_d.set_yscale('log')
    ax_w.set_ylabel('wall time (s)', fontsize=9)
    ax_d.set_ylabel('drift error\n(NRMSE @ states)', fontsize=9)
    ax_d.set_xlabel(x_label, fontsize=9)
    for ax in (ax_w, ax_d):
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(labelsize=8)
    ax_w.legend(fontsize=7, loc='best')

    if not any_pts:
        ax_w.text(0.5, 0.5, 'no completed cells yet',
                  ha='center', va='center', transform=ax_w.transAxes)

    fig.tight_layout()
    out_png = Path(args.out_dir) / 'gpdrift_scaling.png'
    out_pdf = Path(args.out_dir) / 'gpdrift_scaling.pdf'
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")

    # brief text summary
    for key, (label, *_ ) in SERIES.items():
        ok = series[key]['ok']
        failed = series[key]['failed']
        cells = ", ".join(f"{args.x_key}={r[0]}: {r[1]:.0f}s/drift={r[2]:.3f}"
                          for r in ok)
        fail = f"  failed@{args.x_key}={failed}" if failed else ""
        print(f"  {label:16s} {cells}{fail}")


if __name__ == '__main__':
    main()
