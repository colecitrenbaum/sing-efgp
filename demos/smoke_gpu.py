"""GPU smoke test (Phase B): confirm jax_finufft + EFGP + SparseGP run on GPU.

Trimmed schedule on the neural data — just enough to prove both
``fit_efgp_sing_jax`` and ``fit_variational_em`` compile and step on GPU
without NaN. Go/no-go gate before the swl1 sweep.

Run under Slurm (demos/smoke_gpu.sbatch), NOT the login node.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax.* use
import numpy as np
import jax.numpy as jnp

print("=== device / finufft check ===", flush=True)
print("jax", jax.__version__, "devices:", jax.devices(), flush=True)
import jax_finufft
# 2-D type-2 NUFFT (cuFINUFFT GPU supports 2-D/3-D only, not 1-D). EFGP's
# neural problem is D=2, so this matches the real use case.
F = jnp.ones((8, 8), dtype=jnp.complex128)
x = jnp.linspace(-1.0, 1.0, 5)
y = jnp.linspace(-1.0, 1.0, 5)
out = jax_finufft.nufft2(F, x, y)
print("nufft2 (2D) device:", getattr(out, "device", "n/a"), "dtype:", out.dtype,
      "finite:", bool(np.all(np.isfinite(np.asarray(out)))), flush=True)

# Reuse the neural-demo helpers; patch its module globals to a tiny schedule.
import demos.bench_neural_efgp_vs_sparsegp as bench
bench.SUBSAMPLE_T = 8      # T ~ 525
bench.N_EM = 3
bench.N_ESTEP = 3
bench.N_MSTEP_INNER = 2

D = bench.D

print("\n=== load + init ===", flush=True)
norm = bench.load_neural_data()[::bench.SUBSAMPLE_T]
n_t, n_n = norm.shape
t_grid = jnp.arange(n_t) * (bench.DT * bench.SUBSAMPLE_T)
ys = jnp.asarray(norm[None])
inputs, o1, o2 = bench.build_inputs(n_t)
output_params, x0 = bench.initialize_params_pca(D, ys)
xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
lo = xs_pca.min(0) - 1.0
hi = xs_pca.max(0) + 1.0
ls_init = float(np.max(hi - lo)) / 8.0
X_template = (jnp.linspace(lo.min(), hi.max(), max(n_t, 64))[:, None]
              * jnp.ones((1, D)))
print(f"T={n_t} N={n_n} ls_init={ls_init:.3f}", flush=True)

print("\n=== EFGP fit (3 EM iters) ===", flush=True)
t0 = time.time()
mp_e, hist_e, wall_e = bench.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                      X_template, ls_init)
ls_e, var_e = float(hist_e.lengthscale[-1]), float(hist_e.variance[-1])
m_e = np.asarray(mp_e['m'][0])
print(f"EFGP wall={time.time()-t0:.1f}s ls={ls_e:.3f} var={var_e:.3f} "
      f"m_finite={bool(np.all(np.isfinite(m_e)))}", flush=True)

print("\n=== SparseGP fit (3 EM iters, 5x5 inducing) ===", flush=True)
t0 = time.time()
(mp_s, fn_s, dp_s, gp_s, B_s, elbos_s,
 ls_h, var_h, wall_s) = bench.fit_sparsegp(ys, inputs, output_params, x0,
                                           t_grid, lo, hi, ls_init, num_per_dim=5)
m_s = np.asarray(mp_s['m'][0])
print(f"SparseGP wall={time.time()-t0:.1f}s ls={ls_h[-1]:.3f} var={var_h[-1]:.3f} "
      f"m_finite={bool(np.all(np.isfinite(m_s)))}", flush=True)

ok = (np.all(np.isfinite(m_e)) and np.all(np.isfinite(m_s))
      and np.isfinite(ls_e) and np.isfinite(ls_h[-1]))
print("\n=== SMOKE", "PASS" if ok else "FAIL", "===", flush=True)
sys.exit(0 if ok else 1)
