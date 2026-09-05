"""
plot_gpdrift_scaling_pub.py

Publication-quality (PNAS) version of the GP-drift scaling-law subfigure.
Two vertically stacked panels sharing a log-T x-axis:

    (top)    wall-clock time  vs  sequence length T
    (bottom) drift error (Procrustes-aligned NRMSE)  vs  T

Three methods: EFGP-SING (hero), SparseGP M=49, SparseGP M=100. The
SparseGP M=100 cell OOMs at T=1e5; that failure is marked explicitly.

Design targets PNAS single-column width (8.7 cm) so it drops cleanly into a
broader composite. Vector PDF + 600-dpi PNG are both written.

Pure numpy + matplotlib -- light enough for the login node.

  python demos/plot_gpdrift_scaling_pub.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


# ---- style ---------------------------------------------------------------
# Okabe-Ito colour-blind-safe palette. EFGP is the hero (deep blue).
C_EFGP = '#0072B2'   # blue
C_SP49 = '#E69F00'   # orange
C_SP100 = '#D55E00'  # vermillion
C_FAIL = '#B00020'   # muted red for the OOM marker

# (method, M) -> (label, colour, marker, is_hero)
SERIES = {
    ('efgp', 0):  ('EFGP-SING',       C_EFGP,  '*', True),
    ('sp', 49):   ('SparseGP (M=49)',  C_SP49,  's', False),
    ('sp', 100):  ('SparseGP (M=100)', C_SP100, 'o', False),
}


def _rcparams():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.solid_capstyle': 'round',
        'savefig.dpi': 600,
        'pdf.fonttype': 42,   # embed TrueType (editable text in Illustrator)
        'ps.fonttype': 42,
    })


def _scalar(z, key, default=np.nan):
    return float(z[key]) if key in z.files else default


def _drift(z):
    if 'drift_nrmse' in z.files:
        return _scalar(z, 'drift_nrmse')
    return float(np.sqrt(_scalar(z, 'drift_rel_mse')))


def load_cells(out_dir, seed):
    """{(method, M): {'ok': [(T, wall, drift)...], 'failed': [T...]}}."""
    series = {k: {'ok': [], 'failed': []} for k in SERIES}
    for path in sorted(glob.glob(str(Path(out_dir) / 'cell_*.npz'))):
        z = np.load(path, allow_pickle=True)
        if 'seed' in z.files and int(z['seed']) != seed:
            continue
        key = (str(z['method']), int(z['M']) if 'M' in z.files else 0)
        if key not in series:
            continue
        status = str(z['status']) if 'status' in z.files else 'ok'
        T = int(z['T'])
        if status == 'ok':
            series[key]['ok'].append((T, _scalar(z, 'wall'), _drift(z)))
        else:
            series[key]['failed'].append(T)
    for key in series:
        series[key]['ok'].sort()
    return series


def _log_tick_fmt(vals):
    """Formatter that prints a chosen set of values as plain numbers on a
    log axis and hides everything else."""
    valset = set(vals)

    def _f(x, _pos):
        for v in vals:
            if abs(x - v) < 1e-9 * max(1.0, abs(v)):
                if v >= 1:
                    return f'{v:g}'
                return f'{v:g}'
        return ''
    return FuncFormatter(_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='demos/_bench_duffing_scaling_out')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    _rcparams()
    series = load_cells(args.out_dir, args.seed)

    # PNAS single column ~ 8.7 cm wide.
    fig, (ax_w, ax_d) = plt.subplots(
        2, 1, figsize=(3.42, 4.05), sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12})

    def _plot(ax, idx):
        for key, (label, color, marker, hero) in SERIES.items():
            ok = series[key]['ok']
            if not ok:
                continue
            Ts = [r[0] for r in ok]
            ys = [r[idx] for r in ok]
            lw = 2.0 if hero else 1.3
            ms = 9 if marker == '*' else (5.5 if not hero else 6)
            z = 5 if hero else 3
            ax.plot(Ts, ys, marker=marker, color=color, label=label,
                    markersize=ms, linewidth=lw, zorder=z,
                    markeredgecolor='white', markeredgewidth=0.6,
                    clip_on=False)

    _plot(ax_w, 1)   # wall time
    _plot(ax_d, 2)   # drift NRMSE

    # ---- OOM annotation for SparseGP M=100 at T=1e5 -----------------------
    # Deferred until after y-limits are fixed so the dashed continuation and
    # the cross line up exactly.
    _oom = None
    fails = series[('sp', 100)]['failed']
    ok100 = series[('sp', 100)]['ok']
    if fails and ok100:
        _oom = (max(fails), ok100[-1][0], ok100[-1][1])  # T_fail, T_last, w_last

    # ---- axes scaling & ticks -------------------------------------------
    ax_w.set_xscale('log')
    ax_w.set_yscale('log')
    ax_d.set_yscale('log')

    ax_w.set_xlim(8e2, 1.3e5)
    ax_w.set_ylim(50, 560)
    ax_d.set_ylim(0.08, 0.36)

    # x ticks: the three sampled sequence lengths.
    for ax in (ax_w, ax_d):
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.xaxis.set_major_formatter(_log_tick_fmt([1e3, 1e4, 1e5]))
        ax.xaxis.set_minor_formatter(NullFormatter())

    wall_ticks = [50, 100, 200, 300, 500]
    ax_w.set_yticks(wall_ticks)
    ax_w.yaxis.set_major_formatter(_log_tick_fmt(wall_ticks))
    ax_w.yaxis.set_minor_formatter(NullFormatter())

    drift_ticks = [0.1, 0.2, 0.3]
    ax_d.set_yticks(drift_ticks)
    ax_d.yaxis.set_major_formatter(_log_tick_fmt(drift_ticks))
    ax_d.yaxis.set_minor_formatter(NullFormatter())

    # Place OOM cross + dashed continuation now that y-limits are fixed.
    if _oom is not None:
        T_fail, T_last, w_last = _oom
        y_cross = 505.0   # just above the SparseGP(M=49) endpoint (467 s)
        ax_w.plot([T_last, T_fail], [w_last, y_cross],
                  ls=(0, (2, 1.6)), color=C_SP100, lw=1.1, zorder=2)
        ax_w.plot([T_fail], [y_cross], marker='X', color=C_FAIL,
                  markersize=8.5, markeredgewidth=0.6, markeredgecolor='white',
                  zorder=6, clip_on=False)
        ax_w.annotate('out of\nmemory', xy=(T_fail, y_cross),
                      xytext=(-6, 1), textcoords='offset points',
                      ha='right', va='center', fontsize=6.3, color=C_FAIL,
                      linespacing=0.95)

    # ---- labels & spines -------------------------------------------------
    ax_w.set_ylabel('Wall-clock time (s)')
    ax_d.set_ylabel('Drift error (NRMSE)')
    ax_d.set_xlabel('Sequence length  $T$')

    for ax in (ax_w, ax_d):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='major', axis='both', color='0.85',
                linewidth=0.5, zorder=0)
        ax.tick_params(which='both', length=3)
        ax.tick_params(which='minor', length=0)

    # ---- shared legend below the panels ---------------------------------
    handles, labels = ax_w.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=1,
                     frameon=False, handlelength=1.6, columnspacing=1.0,
                     bbox_to_anchor=(0.56, -0.005), borderaxespad=0.0)
    for line in leg.get_lines():
        line.set_markeredgewidth(0.6)

    fig.subplots_adjust(left=0.185, right=0.965, top=0.98, bottom=0.20)

    out_png = Path(args.out_dir) / 'gpdrift_scaling_pub.png'
    out_pdf = Path(args.out_dir) / 'gpdrift_scaling_pub.pdf'
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f'saved {out_png}')
    print(f'saved {out_pdf}')

    # text summary
    for key, (label, *_) in SERIES.items():
        ok = series[key]['ok']
        failed = series[key]['failed']
        cells = ', '.join(f'T={r[0]}: {r[1]:.0f}s / NRMSE={r[2]:.3f}'
                          for r in ok)
        fail = f'  FAILED@T={failed}' if failed else ''
        print(f'  {label:18s} {cells}{fail}')


if __name__ == '__main__':
    main()
