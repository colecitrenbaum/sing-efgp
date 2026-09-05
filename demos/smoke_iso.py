"""Quick wiring smoke for IsotropicRBF SparseGP fit — tiny T, M, N_EM.

Verifies the isotropic kernel plugs into SING's E-step / M-step and that a
single scalar lengthscale is learned without shape/NaN errors.  NOT a
scientific run — just a green-light before submitting the full sweep.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import sys, math, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import numpy as np
import jax.numpy as jnp

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_iso_x64 as iso

print("devices:", jax.devices(), flush=True)

# Shrink the workload so this finishes in ~1 min.
bench.T = 200
iso.T = 200
iso.N_EM = 6
bench.N_EM = 6

xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
xs_np = np.asarray(xs)
print("data range x0:", xs_np[:,0].min(), xs_np[:,0].max(), flush=True)

# Sanity: kernel evaluates on a couple points and matches an explicit isotropic RBF.
k = iso.IsotropicRBF(latent_dim=iso.D)
kp = dict(length_scale=jnp.asarray(0.8), output_scale=jnp.asarray(math.sqrt(1.5)))
x1 = jnp.array([0.1, -0.2]); x2 = jnp.array([0.3, 0.4])
got = float(k.K(x1, x2, kp))
ref = 1.5 * math.exp(-0.5 * float(((x1 - x2) / 0.8 ** 1) .__pow__(2).sum()))
print(f"K check: got={got:.6f} ref={ref:.6f} diff={abs(got-ref):.2e}", flush=True)
assert abs(got - ref) < 1e-9, "IsotropicRBF.K mismatch"

t0 = time.perf_counter()
s = iso.fit_sparsegp_iso(lik, op, ip, t_grid, sigma, num_per_dim=5,
                         ls_init=0.7, xs_np=xs_np)
print(f"iso fit OK: ls={s['ls']:.4f} var={s['var']:.4f} "
      f"ls_traj_last3={s['ls_traj'][-3:]} wall={time.perf_counter()-t0:.1f}s",
      flush=True)
assert np.isfinite(s['ls']) and np.isfinite(s['var']), "NaN in iso fit"
# drift eval path
d = bench.eval_sp_drift(s, jnp.array([[0.0, 0.0], [0.5, -0.5]]))
print("drift eval OK, shape:", np.asarray(d).shape, flush=True)
print("SMOKE PASS", flush=True)
