"""
bench_gpdrift_inducing_sweep_iso_x64.py

ISOTROPIC-kernel re-run of demos/bench_gpdrift_inducing_sweep_x64.py.

Motivation
----------
The ARD sweep (bench_gpdrift_inducing_sweep_x64.py) found that SparseGP does
NOT converge to EFGP / the oracle MLE as #inducing grows: EFGP lands at
ℓ≈0.85 (oracle MLE ℓ≈0.78) while ARD SparseGP settles at ℓ≈1.5 (init 0.7) or
sticks at ℓ≈3.0 (init 3.0).  The suspected cause is the KERNEL PARAMETRISATION,
not the inducing approximation: bench.fit_sparsegp learns a `length_scales`
VECTOR of shape (D,) — one lengthscale per input dimension (ARD) — whereas EFGP
uses a single ISOTROPIC lengthscale.  On this well-specified problem the drift
GP is genuinely isotropic (f_true ~ GP(0, RBF(ℓ_true, σ²_true)) with a scalar
ℓ), so the extra ARD degrees of freedom let SparseGP wander to a different
optimum in the projected (mean-ℓ, σ²) plane.

This script re-runs the exact same sweep with SparseGP constrained to a single
shared (isotropic) lengthscale via `IsotropicRBF`, so the M-step optimises ONE
scalar ℓ instead of D independent ones — an apples-to-apples match to EFGP.
Everything else (data, matched training hypers, inducing layout, metrics,
oracle landscape) is imported verbatim from the ARD sweep / head-to-head demos.

Deliverables (in demos/_bench_gpdrift_inducing_sweep_iso_out/):
  landscape_paths.png / convergence.png / sweep.npz — same schema as the ARD run.

Run under Slurm (demos/bench_gpdrift_inducing_sweep_iso.sbatch), NOT login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax.* use (CLAUDE.md)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.random as jr

# Reuse ALL data / EFGP-fit / metric / landscape machinery from the head-to-head
# demo, and the render_* + sweep constants from the ARD inducing sweep.
import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep

from sing.kernels import RBF
from sing.sde import SparseGP
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em

D = bench.D
T = bench.T
LS_TRUE = bench.LS_TRUE
VAR_TRUE = bench.VAR_TRUE
VAR_INIT = bench.VAR_INIT
N_EM = bench.N_EM
N_M_INNER = bench.N_M_INNER
MSTEP_LR = bench.MSTEP_LR

# Reuse the sweep's grid / M / ls_init settings so the two runs are comparable.
LS_INIT_LIST = sweep.LS_INIT_LIST
M_LIST = sweep.M_LIST

# The render_* helpers in the ARD sweep module read module globals LS_INIT_LIST /
# M_LIST off `sweep`; they already equal ours, so the imported renderers work as-is.

OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)


class IsotropicRBF(RBF):
    """RBF with a single SHARED (isotropic) lengthscale.

    `kernel_params` carries a scalar `length_scale`; internally it is broadcast
    to a (D,) `length_scales` vector so every closed-form RBF expectation
    (E_Kxz / E_KzxKxz / E_dKzxdx, which each expect a (D,) `length_scales`)
    works unchanged.  Because the optimised pytree leaf is one scalar, the
    SING M-step ties all D dimensions together — the isotropic constraint EFGP
    also imposes — instead of learning D independent ARD lengthscales.
    """

    def _expand(self, kernel_params):
        # Idempotent: base RBF.E_dKzxdx internally re-dispatches to self.E_Kxz
        # with already-expanded params, so pass those straight through.
        if "length_scale" not in kernel_params:
            return kernel_params
        ls = kernel_params["length_scale"] * jnp.ones(self.latent_dim)
        return {"length_scales": ls, "output_scale": kernel_params["output_scale"]}

    def K(self, x1, x2, kernel_params):
        return super().K(x1, x2, self._expand(kernel_params))

    def E_Kxx(self, expectation, key, m, S, kernel_params):
        return super().E_Kxx(expectation, key, m, S, self._expand(kernel_params))

    def E_Kxz(self, expectation, key, z, m, S, kernel_params, **kwargs):
        return super().E_Kxz(expectation, key, z, m, S,
                             self._expand(kernel_params), **kwargs)

    def E_KzxKxz(self, expectation, key, z1, z2, m, S, kernel_params, **kwargs):
        return super().E_KzxKxz(expectation, key, z1, z2, m, S,
                                self._expand(kernel_params), **kwargs)

    def E_dKzxdx(self, expectation, key, z, m, S, kernel_params, **kwargs):
        return super().E_dKzxdx(expectation, key, z, m, S,
                                self._expand(kernel_params), **kwargs)


def fit_sparsegp_iso(lik, op, ip, t_grid, sigma, num_per_dim, ls_init, xs_np):
    """Identical to bench.fit_sparsegp but with the ISOTROPIC kernel: a single
    scalar `length_scale` in drift_params (learned as one shared ℓ) rather than
    a (D,) ARD `length_scales` vector."""
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = bench._data_aware_zs(num_per_dim, xs_np)
    sparse = SparseGP(zs=zs, kernel=IsotropicRBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scale=jnp.asarray(float(ls_init)),
                         output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((N_EM,), MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] +
                       [float(h['length_scale']) for h in history])
    var_traj = np.array([VAR_INIT] +
                        [float(h['output_scale']) ** 2 for h in history])
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls_traj=ls_traj, var_traj=var_traj,
                ls=float(dp['length_scale']),
                var=float(dp['output_scale']) ** 2,
                wall=wall)


def main():
    print(f"bench_gpdrift_inducing_sweep_ISO: T={T} ls_init={LS_INIT_LIST} "
          f"M={M_LIST} θ_true=(ℓ={LS_TRUE}, σ²={VAR_TRUE})  "
          f"x64={jax.config.read('jax_enable_x64')}  devices={jax.devices()}",
          flush=True)

    print("Sampling drift f_true ~ GP(0, RBF(ℓ_true, σ²_true))...", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)
    print(f"  trajectory range: x[0]∈[{xs_np[:,0].min():.2f}, "
          f"{xs_np[:,0].max():.2f}], x[1]∈[{xs_np[:,1].min():.2f}, "
          f"{xs_np[:,1].max():.2f}]", flush=True)

    LOG_LS = np.linspace(bench.LOG_LS_RANGE[0], bench.LOG_LS_RANGE[1], bench.N_GRID)
    LOG_VAR = np.linspace(bench.LOG_VAR_RANGE[0], bench.LOG_VAR_RANGE[1], bench.N_GRID)
    print(f"Computing GT oracle landscape ({bench.N_GRID}×{bench.N_GRID})...",
          flush=True)
    t0 = time.perf_counter()
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"(true: ℓ={LS_TRUE}, σ²={VAR_TRUE})  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    results = {}
    for ls_init in LS_INIT_LIST:
        print(f"\n--- ls_init = {ls_init} ---", flush=True)
        cell = dict(ls_init=ls_init, sparse={})

        print("  EFGP fit...", flush=True)
        e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
        try:
            ed = bench.compute_drift_metrics(
                e['mp'], xs_np,
                f_eval_fn=lambda g: sweep.eval_efgp_drift(e['hist'], g),
                drift_fn=drift_fn)
        except Exception as ex:
            print(f"    (EFGP drift metric skipped: {type(ex).__name__}: {ex})",
                  flush=True)
            ed = dict(rel_mse=np.nan, rel_mse_raw=np.nan)
        elr = bench.latent_recovery_rmse(e['mp'], xs_np)
        print(f"    ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
              f"drift_pc={ed['rel_mse']:.4f}, lat_pc={elr['pc']:.4f}, "
              f"wall={e['wall']:.1f}s", flush=True)
        cell['efgp'] = dict(state=e, drift=ed, latent_rmse=elr)

        for M in M_LIST:
            n_per = int(round(math.sqrt(M)))
            print(f"  SP(iso) M={M} ({n_per}×{n_per}) fit...", flush=True)
            try:
                s = fit_sparsegp_iso(lik, op, ip, t_grid, sigma, n_per,
                                     ls_init, xs_np)
            except Exception as ex:
                print(f"    EXC (fit): {type(ex).__name__}: {ex}", flush=True)
                nan_state = dict(ls_traj=np.array([np.nan]),
                                 var_traj=np.array([np.nan]),
                                 ls=np.nan, var=np.nan, wall=np.nan)
                cell['sparse'][M] = dict(
                    state=nan_state,
                    drift=dict(rel_mse=np.nan, rel_mse_raw=np.nan),
                    latent_rmse=dict(pc=np.nan, raw=np.nan))
                continue
            try:
                sd = bench.compute_drift_metrics(
                    s['mp'], xs_np,
                    f_eval_fn=lambda g, s_=s: bench.eval_sp_drift(s_, g),
                    drift_fn=drift_fn)
            except Exception as ex:
                print(f"    (SP drift metric skipped: {type(ex).__name__}: {ex})",
                      flush=True)
                sd = dict(rel_mse=np.nan, rel_mse_raw=np.nan)
            slr = bench.latent_recovery_rmse(s['mp'], xs_np)
            print(f"    ℓ={s['ls']:.3f}, σ²={s['var']:.3f}, "
                  f"drift_pc={sd['rel_mse']:.4f}, lat_pc={slr['pc']:.4f}, "
                  f"wall={s['wall']:.1f}s", flush=True)
            cell['sparse'][M] = dict(state=s, drift=sd, latent_rmse=slr)
        results[ls_init] = cell

    T_data = dict(T=T, xs_np=xs_np, t_grid=t_grid, sigma=sigma,
                  LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                  gb_ll=gb_ll, gb_lv=gb_lv, results=results)

    # Reuse the ARD sweep's renderers (same result schema).
    sweep.render_landscape(T_data, OUT_DIR / 'landscape_paths.png')
    sweep.render_convergence(T_data, OUT_DIR / 'convergence.png')

    # ---- save raw ----
    save_kwargs = dict(T=T, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                       gb_ll=gb_ll, gb_lv=gb_lv, xs_np=xs_np,
                       ls_true=LS_TRUE, var_true=VAR_TRUE,
                       M_list=np.asarray(M_LIST),
                       ls_init_list=np.asarray(LS_INIT_LIST))
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e = cell['efgp']['state']
        save_kwargs[f'lsinit{ls_init}_efgp_ls_traj'] = e['ls_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_var_traj'] = e['var_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_ls'] = e['ls']
        save_kwargs[f'lsinit{ls_init}_efgp_var'] = e['var']
        save_kwargs[f'lsinit{ls_init}_efgp_wall'] = e['wall']
        save_kwargs[f'lsinit{ls_init}_efgp_drift_pc'] = \
            cell['efgp']['drift']['rel_mse']
        for M in M_LIST:
            st = cell['sparse'][M]['state']
            save_kwargs[f'lsinit{ls_init}_sp{M}_ls_traj'] = st['ls_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_var_traj'] = st['var_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_ls'] = st['ls']
            save_kwargs[f'lsinit{ls_init}_sp{M}_var'] = st['var']
            save_kwargs[f'lsinit{ls_init}_sp{M}_wall'] = st['wall']
            save_kwargs[f'lsinit{ls_init}_sp{M}_drift_pc'] = \
                cell['sparse'][M]['drift']['rel_mse']
    np.savez(OUT_DIR / 'sweep.npz', **save_kwargs)
    print(f"  saved {OUT_DIR / 'sweep.npz'}", flush=True)

    # ---- summary table ----
    print(f"\n{'='*90}")
    print(f"SUMMARY (ISOTROPIC SparseGP) — well-specified GP drift T={T}  "
          f"(θ_true: ℓ={LS_TRUE}, σ²={VAR_TRUE}; "
          f"oracle MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f})")
    print(f"{'='*90}")
    print(f"  {'ls_init':>7s}  {'method':>11s}  {'ℓ_final':>8s}  "
          f"{'σ²_final':>9s}  {'drift_pc':>9s}  {'wall':>7s}")
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e = cell['efgp']['state']
        print(f"  {ls_init:>7.1f}  {'EFGP':>11s}  {e['ls']:>8.3f}  "
              f"{e['var']:>9.3f}  {cell['efgp']['drift']['rel_mse']:>9.4f}  "
              f"{e['wall']:>6.1f}s")
        for M in M_LIST:
            st = cell['sparse'][M]['state']
            dp = cell['sparse'][M]['drift']['rel_mse']
            ls_s = f"{st['ls']:>8.3f}" if math.isfinite(st['ls']) else f"{'NaN':>8s}"
            var_s = f"{st['var']:>9.3f}" if math.isfinite(st['var']) else f"{'NaN':>9s}"
            dp_s = f"{dp:>9.4f}" if math.isfinite(dp) else f"{'NaN':>9s}"
            w_s = f"{st['wall']:>6.1f}s" if math.isfinite(st['wall']) else f"{'NaN':>7s}"
            print(f"  {ls_init:>7.1f}  {f'SP(iso) M={M}':>11s}  {ls_s}  {var_s}  "
                  f"{dp_s}  {w_s}")


if __name__ == '__main__':
    main()
