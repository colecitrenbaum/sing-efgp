"""Quick smoke: isotropic SparseGP fit + drift-field eval on neural data (GPU).
Confirms IsotropicRBF works end-to-end (incl. the E_dKzxdx re-dispatch path and
get_posterior_f_mean drift field) before the full swl1 sweep. Run via srun on dev.
"""
from __future__ import annotations
import sys
from pathlib import Path
_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import demos.bench_neural_efgp_vs_sparsegp as bench
import demos.bench_neural_inducing_sweep_iso_x64 as iso

print("jax", jax.__version__, "devices:", jax.devices(), flush=True)
bench.SUBSAMPLE_T = 8      # T ~ 525
bench.N_EM = 3
bench.N_ESTEP = 3
bench.N_MSTEP_INNER = 2
D = bench.D

norm = bench.load_neural_data()[::bench.SUBSAMPLE_T]
n_t, n_n = norm.shape
t_grid = jnp.arange(n_t) * (bench.DT * bench.SUBSAMPLE_T)
ys = jnp.asarray(norm[None])
inputs, o1, o2 = bench.build_inputs(n_t)
op, x0 = bench.initialize_params_pca(D, ys)
xs = np.asarray((ys[0] - op['d']) @ op['C'])
lo = xs.min(0) - 1.0; hi = xs.max(0) + 1.0
ls_init = float(np.max(hi - lo)) / 8.0
print(f"T={n_t} N={n_n} ls_init={ls_init:.3f}", flush=True)

print("=== isotropic SparseGP fit (3 EM, 5x5) ===", flush=True)
mp, fn, dp, gp, B, elbos, ls_h, var_h, wall = iso.fit_sparsegp_iso(
    ys, inputs, op, x0, t_grid, lo, hi, ls_init, num_per_dim=5)
m = np.asarray(mp['m'][0])
print(f"wall={wall:.1f}s  ls={ls_h[-1]:.4f} (scalar) var={var_h[-1]:.4f}  "
      f"m_finite={bool(np.all(np.isfinite(m)))}  dp_keys={list(dp.keys())}", flush=True)

print("=== drift field eval (get_posterior_f_mean) ===", flush=True)
gx = np.linspace(lo[0], hi[0], 8); gy = np.linspace(lo[1], hi[1], 8)
GX, GY = np.meshgrid(gx, gy, indexing='ij')
pts = np.stack([GX.ravel(), GY.ravel()], -1)
F = bench.sparsegp_drift_field(fn, dp, gp, pts)
print(f"drift field shape={F.shape} finite={bool(np.all(np.isfinite(F)))}", flush=True)

ok = np.all(np.isfinite(m)) and np.all(np.isfinite(F)) and np.isfinite(ls_h[-1])
print("\n=== ISO SMOKE", "PASS" if ok else "FAIL", "===", flush=True)
sys.exit(0 if ok else 1)
