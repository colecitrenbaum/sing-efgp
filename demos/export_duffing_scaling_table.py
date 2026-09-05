"""Export the Duffing scaling cells as a tidy scalar table for paper plots.

The cell_*.npz files each carry the full arrays needed for offline re-scoring
(xs_true, eval_pts, f_pred_states_*, m_inf, S_diag), which makes them ~5MB at
T=1e5. For plotting you only need the scalars, so this writes a flat CSV +
JSON alongside them. plot_gpdrift_scaling_pub.py still reads the npz directly;
this is for hand-rolled/paper plotting that shouldn't have to load 30MB.

  python demos/export_duffing_scaling_table.py \
      --out-dir demos/_bench_duffing_keepall_out [more dirs ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

# Scalars worth carrying into a plot or a table. Anything missing from a given
# cell (older cells predate the keep-all/gather fields) comes back as None.
FIELDS = [
    'dynamics', 'T', 'method', 'M', 'ls_init', 'seed', 'dt', 'eps_grid',
    'status',
    # which E-step / q(x)-moment path produced this cell
    'estep_method', 'analytic_order', 'qx_moments_method',
    'gather_N', 'gather_stencil_r', 'backend', 'peak_gb',
    # primary metrics
    'wall', 'drift_nrmse', 'drift_nrmse_raw', 'drift_rel_mse', 'var_f',
    'lat_pc', 'lat_raw',
    # recovered hyperparameters
    'ls_final', 'var_final',
]


def _get(z, key):
    if key not in z.files:
        return None
    v = z[key]
    if v.ndim > 0:                      # trajectories etc. -- not scalars
        return None
    v = v.item()
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def collect(out_dirs):
    rows = []
    for d in out_dirs:
        for path in sorted(glob.glob(str(Path(d) / 'cell_*.npz'))):
            z = np.load(path, allow_pickle=True)
            row = {k: _get(z, k) for k in FIELDS}
            row['source_dir'] = str(d)
            row['cell'] = Path(path).name
            # Older cells have no explicit status; they were written on success.
            if row['status'] is None:
                row['status'] = 'ok'
            rows.append(row)
    rows.sort(key=lambda r: (str(r['method']), r['T'] or 0, r['M'] or 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', nargs='+', required=True,
                    help='one or more cell dirs to pool')
    ap.add_argument('--dest', default=None,
                    help='output basename (default: <first out-dir>/table)')
    args = ap.parse_args()

    rows = collect(args.out_dir)
    dest = Path(args.dest) if args.dest else Path(args.out_dir[0]) / 'table'
    dest.parent.mkdir(parents=True, exist_ok=True)

    cols = FIELDS + ['source_dir', 'cell']
    with open(dest.with_suffix('.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(dest.with_suffix('.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)

    print(f"wrote {dest.with_suffix('.csv')}  ({len(rows)} cells)")
    print(f"wrote {dest.with_suffix('.json')}")
    hdr = f"\n{'method':22s} {'T':>7} {'M':>4} {'wall':>8} {'nrmse':>8} " \
          f"{'lat_pc':>8} {'ls':>7} {'var':>7}"
    print(hdr)
    for r in rows:
        if r['status'] != 'ok':
            print(f"{str(r['method']):22s} {r['T']:>7} {r['M'] or 0:>4} "
                  f"{'FAILED':>8}")
            continue
        f = lambda k, w=8, p=4: (f"{r[k]:>{w}.{p}f}" if r[k] is not None
                                 else f"{'-':>{w}}")
        print(f"{str(r['method']):22s} {r['T']:>7} {r['M'] or 0:>4} "
              f"{f('wall', 8, 1)} {f('drift_nrmse')} {f('lat_pc')} "
              f"{f('ls_final', 7, 3)} {f('var_final', 7, 3)}")


if __name__ == '__main__':
    main()
