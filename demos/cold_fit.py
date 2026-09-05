"""True single COLD fit wall (fresh process, no warm-up) + startup attribution.

Answers: what does one real fit_efgp_sing_jax(n_em=50, T=10000) cost from a
cold process, and where does the startup go?  Per-iter timing via
EFGP_PROFILE_ITERS shows iter-0 (all first-time compile: CUDA/finufft/gmix/CG
+ E-step scan graph) vs iters 1+ (execution) vs iter-8 (M-step compile).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ['EFGP_PROFILE_ITERS'] = '1'

_t = time.perf_counter()
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
t_import = time.perf_counter() - _t
print(f"[startup] import jax+deps: {t_import:.1f}s", flush=True)

_t = time.perf_counter()
_ = jnp.zeros(1).block_until_ready()          # first device op → CUDA/XLA init
t_devinit = time.perf_counter() - _t
print(f"[startup] first device op (CUDA/XLA init): {t_devinit:.1f}s "
      f"backend={jax.default_backend()}", flush=True)

import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base

T = 10000
xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(T, 0)
xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
rho = jnp.linspace(0.05, 0.7, 50)

print("\n=== SINGLE COLD FIT (n_em=50, n_estep=10, T=10000) — no warm-up ===", flush=True)
t0 = time.perf_counter()
mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
    likelihood=lik, t_grid=t_grid, output_params=op, init_params=ip,
    latent_dim=2, lengthscale=0.7, variance=base.VAR_INIT, sigma=sigma,
    sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2, n_em_iters=50,
    n_estep_iters=10, rho_sched=rho, learn_emissions=False, update_R=False,
    learn_kernel=True, n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
    n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=xt,
    estep_method='gmix', verbose=False)
jax.block_until_ready(mp['m'])
wall = time.perf_counter() - t0
ls = float(hist.lengthscale[-1]) if len(hist.lengthscale) else float('nan')
print(f"\nCOLD FIT wall = {wall:.1f}s   (+import {t_import:.1f}s +devinit "
      f"{t_devinit:.1f}s)   ls_recovered≈{ls:.3f}", flush=True)
