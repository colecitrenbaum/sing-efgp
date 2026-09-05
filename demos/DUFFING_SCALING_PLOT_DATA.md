# Duffing scaling-law plot data (EFGP vs SparseGP vs keep-all)

## What to download

Everything lives in **`demos/_bench_duffing_keepall_out/`**. Grab the whole
directory (~17 MB):

```bash
rsync -avP \
  ccitren@dtn.sherlock.stanford.edu:/scratch/users/ccitren/sing-efgp/demos/_bench_duffing_keepall_out/ \
  ./duffing_scaling/
```

(Use the DTN, not the login node.)

If you only want the numbers and not the arrays, the two small files are
enough — everything the wall/NRMSE panels need is in them:

```
duffing_scaling_table.csv     # tidy, one row per cell
duffing_scaling_table.json    # same content
```

## Contents

| file | what it is |
|---|---|
| `cell_T{1000,10000,100000}_efgp_seed0.npz` | EFGP-SING, production shim |
| `cell_T{...}_efgp_keepall_seed0.npz` | EFGP-SING, keep-all exact E-step |
| `cell_T{...}_efgp_keepall_gather_gN512r64_seed0.npz` | keep-all, gmix-gather moments (diagnostic, not a paper series) |
| `cell_T{...}_sp49_seed0.npz` | SparseGP-SING, M=49, isotropic RBF |
| `duffing_scaling_table.csv` / `.json` | scalar export of all of the above |

All cells are one seed (`seed=0`), fp64, H100 SXM5. Every method within a
given T ran back-to-back on the **same allocated GPU** in a fresh python
process, so the walls are directly comparable.

## Regenerating the figure

```bash
python demos/plot_gpdrift_scaling_pub.py --out-dir demos/_bench_duffing_keepall_out
```

`plot_gpdrift_scaling_pub.py` globs `cell_*.npz` in ONE directory and keys on
`(method, M)`, so point it at a single dir — pooling two dirs (e.g. adding
`_bench_duffing_scaling_newcanon`) double-plots each series.

`SERIES` in that script registers `efgp`, `efgp_keepall`, `sp49`, `sp100`.
`efgp_keepall_gather` is deliberately NOT registered: it is a diagnostic
(same NRMSE as the exact path, slower), so it stays in the CSV and out of
the figure.

## Regenerating the table

```bash
python demos/export_duffing_scaling_table.py \
    --out-dir demos/_bench_duffing_keepall_out \
    --dest demos/_bench_duffing_keepall_out/duffing_scaling_table
```

## Per-cell fields (beyond the plotted scalars)

Each `.npz` carries arrays for offline re-scoring, which is why the T=1e5
cells are ~5 MB:

- `xs_true` (T, D) true latent path; `m_inf` (T, D) posterior mean; `S_diag`
- `eval_pts`, `f_true_states`, `f_pred_states_pc`, `f_pred_states_raw`
- `procrustes_A`, `procrustes_b` (the alignment actually applied)
- `ls_traj`, `var_traj` per-EM-iter hyperparameter trajectories

So you can re-derive the drift metric, or plot hyperparameter trajectories /
posterior overlays, without re-running any fits.

## Provenance / caveats

- `_bench_duffing_scaling_newcanon/` holds the older EFGP+SparseGP-only cells
  (July). They agree with the fresh re-runs: SparseGP NRMSE reproduces
  exactly (0.3131 / 0.1624 / 0.1221); the shimmed EFGP NRMSE varies
  run-to-run at T=1e5 (0.0856 / 0.0914 / 0.0953 across three runs), so quote
  it with that spread in mind. Keep-all is stable there (0.0481 / 0.0483).
- Walls are compile-dominated at small T (~55 s of the T=1e3 wall is one-time
  XLA compile), so read the wall panel as "fixed compile + a T-dependent
  execution term", not as pure E-step cost. `_bench_duffing_keepall_prof/`
  has the `EFGP_PROFILE_ITERS=1` runs that split the two.
