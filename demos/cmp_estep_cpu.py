"""
cmp_estep_cpu.py — time the EFGP q(f) estep_method 'gmix' vs 'analytic' on the
SAME data/settings, to test whether the analytic Taylor-envelope NUFFT path wins
on CPU (where gmix's spread+FFT is expensive) even if it doesn't on GPU.

Runs the full canonical EFGP fit (single trial, Duffing) for each estep_method,
timing a COLD run (incl JIT compile, the benchmark metric) and a WARM run
(compute only, jit cache reused). Reports wall + drift NRMSE for each.

Usage (CPU node): python demos/cmp_estep_cpu.py --T 10000
Force CPU with JAX_PLATFORMS=cpu in the environment.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import jax, jax.numpy as jnp
import demos.bench_gpdrift_scaling as run
import demos.bench_duffing_scaling as ds
import demos.bench_gpdrift_x64 as base


def fit_once(lik, op, ip, t_grid, sigma, ls_init, x_template, estep):
    N_EM = base.N_EM
    rho = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid, output_params=op, init_params=ip,
        latent_dim=base.D, lengthscale=ls_init, variance=base.VAR_INIT,
        sigma=sigma, sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho,
        learn_emissions=False, update_R=False, learn_kernel=True,
        n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=x_template,
        estep_method=estep, verbose=False)
    jax.block_until_ready(mp['m'])
    wall = time.perf_counter() - t0
    ls = float(hist.lengthscale[-1]) if len(hist.lengthscale) else ls_init
    return mp, wall, ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    print(f"device={jax.devices()}  T={args.T}", flush=True)
    xs, lik, op, ip, t_grid, sigma = ds.make_data(args.T, args.seed)
    xs_np = np.asarray(xs)
    xt = jnp.asarray(run.data_aware_template(xs_np))
    for estep in ['gmix', 'analytic']:
        for label in ['cold', 'warm']:
            mp, wall, ls = fit_once(lik, op, ip, t_grid, sigma, 0.7, xt, estep)
            print(f"  estep={estep:8s} {label}: wall={wall:7.1f}s  l={ls:.3f}",
                  flush=True)


if __name__ == '__main__':
    main()
