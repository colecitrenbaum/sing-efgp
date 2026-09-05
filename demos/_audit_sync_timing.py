"""THROWAWAY AUDIT SCRIPT (do not cite in the paper).

Re-times one benchmark cell reporting THREE walls:
  wall_bench  -- exactly what the bench reports (perf_counter around the fit
                 call, NO device sync)
  wall_sync   -- same t0, but stopped only after jax.block_until_ready() on the
                 full returned pytree (marginal params + everything the metric
                 later touches)
  wall_metric -- t0 .. after the drift metric has been evaluated (i.e. the
                 whole cell)

Also reports the pre-fit (data-gen / inducing-grid) time that each method's
timer excludes.

Usage: python -u demos/_audit_sync_timing.py --T 1000 --method efgp [--reps 1]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import demos.bench_duffing_scaling as duffs
import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base
import demos.bench_gpdrift_inducing_sweep_iso_x64 as iso

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from sing.expectation import GaussHermiteQuadrature
from sing.sde import SparseGP
from sing.sing import fit_variational_em


def _sync(tree):
    jax.block_until_ready(jax.tree_util.tree_leaves(tree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--method', required=True,
                    choices=['efgp', 'efgp_keepall', 'efgp_keepall_gather', 'sp'])
    ap.add_argument('--M', type=int, default=49)
    ap.add_argument('--gather-N', type=int, default=None)
    ap.add_argument('--gather-stencil-r', type=int, default=None)
    args = ap.parse_args()

    print(f"[audit] T={args.T} method={args.method} x64="
          f"{jax.config.read('jax_enable_x64')} dev={jax.devices()}", flush=True)

    t_pre0 = time.perf_counter()
    xs, lik, op, ip, t_grid, sigma = duffs.make_data(args.T, 0)
    xs_np = np.asarray(xs)
    _sync((xs, lik.ys_obs, t_grid))
    t_pre = time.perf_counter() - t_pre0
    print(f"[audit] data-gen (EXCLUDED from both timers) = {t_pre:.2f}s", flush=True)

    _QX = {'efgp': 'gmix_batched',
           'efgp_keepall': 'gmix_full_batched',
           'efgp_keepall_gather': 'gmix_full_batched_gather'}

    if args.method in _QX:
        N_EM = base.N_EM
        rho_sched = jnp.linspace(0.05, 0.7, N_EM)
        t0 = time.perf_counter()
        mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
            likelihood=lik, t_grid=t_grid,
            output_params=op, init_params=ip, latent_dim=base.D,
            lengthscale=0.7, variance=base.VAR_INIT, sigma=sigma,
            sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
            qf_nufft_eps=1e-4, qf_cg_tol=1e-4,
            n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
            learn_emissions=False, update_R=False,
            learn_kernel=True, n_mstep_iters=base.N_M_INNER,
            mstep_lr=base.MSTEP_LR,
            n_hutchinson_mstep=4, kernel_warmup_iters=8,
            X_template=None, K_min_lengthscale=None,
            restore_qf_variance='none',
            estep_method='auto', analytic_order=1,
            qx_moments_method=_QX[args.method],
            qx_v_gather_N=args.gather_N,
            qx_v_gather_stencil_r=args.gather_stencil_r,
            verbose=False)
        wall_bench = time.perf_counter() - t0
        _sync((mp, hist.mu_r if hasattr(hist, 'mu_r') else None))
        _sync(mp)
        wall_sync = time.perf_counter() - t0
        st = dict(mp=mp, hist=hist)
        f_eval = lambda g: base.efgp_em.posterior_drift_mean(hist, g)
        ls_fin = float(hist.lengthscale[-1]); var_fin = float(hist.variance[-1])
    else:
        n_per = int(round(math.sqrt(args.M)))
        quad = GaussHermiteQuadrature(D=base.D, n_quad=5)
        # NOTE: the bench builds `zs` BEFORE t0 -> excluded from SparseGP wall.
        t_zs0 = time.perf_counter()
        zs = base._data_aware_zs(n_per, xs_np)
        sparse = SparseGP(zs=zs, kernel=iso.IsotropicRBF(latent_dim=base.D),
                          expectation=quad)
        _sync(zs)
        print(f"[audit] sp inducing-grid build (EXCLUDED) = "
              f"{time.perf_counter()-t_zs0:.3f}s", flush=True)
        drift_params0 = dict(length_scale=jnp.asarray(0.7),
                             output_scale=jnp.asarray(math.sqrt(base.VAR_INIT)))
        rho_sched = jnp.linspace(0.05, 0.7, base.N_EM)
        history = []
        t0 = time.perf_counter()
        mp, npar, gp_post, dp, *_ = fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=base.N_EM, n_iters_e=10,
            n_iters_m=base.N_M_INNER, perform_m_step=True,
            learn_output_params=False,
            learning_rate=jnp.full((base.N_EM,), base.MSTEP_LR),
            print_interval=999, drift_params_history=history)
        wall_bench = time.perf_counter() - t0
        _sync((mp, gp_post, dp))
        wall_sync = time.perf_counter() - t0
        st = dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp)
        f_eval = lambda g: base.eval_sp_drift(st, g)
        ls_fin = float(dp['length_scale']); var_fin = float(dp['output_scale']) ** 2

    d = run._drift_metrics_at_states(st['mp'], xs_np, duffs.duffing_f_true, f_eval)
    wall_metric = time.perf_counter() - t0

    print(f"[audit] RESULT method={args.method} T={args.T}\n"
          f"   wall_bench  = {wall_bench:8.2f}s   (what the paper reports)\n"
          f"   wall_sync   = {wall_sync:8.2f}s   (delta = "
          f"{wall_sync - wall_bench:+.3f}s)\n"
          f"   wall_metric = {wall_metric:8.2f}s  (delta = "
          f"{wall_metric - wall_bench:+.3f}s)\n"
          f"   pre_fit_excluded = {t_pre:.2f}s\n"
          f"   nrmse={d['nrmse']:.4f} l={ls_fin:.4f} var={var_fin:.4f}",
          flush=True)


if __name__ == '__main__':
    main()
