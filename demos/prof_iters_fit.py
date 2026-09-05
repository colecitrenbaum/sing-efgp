"""Per-outer-iter wall attribution for one canonical T=10000 fit (GPU).

Runs a small warm-up fit (populate shared-primitive + module-level M-step jit
caches), then ONE n_em=50, n_estep=10 fit with EFGP_PROFILE_ITERS=1 so each
outer iter prints estep / mstep+rest wall.  iter-0 exposes the one-time E-step
scan compile; iters 1..49 are steady-state execution; the M-step (post
warmup iter 8) should now be cheap (module-level jit, no per-call recompile).
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault('EFGP_PROFILE_ITERS', '1')
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base


def fit(b, n_em, n_estep):
    rho = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.perf_counter()
    mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
        likelihood=b['lik'], t_grid=b['t_grid'], output_params=b['op'],
        init_params=b['ip'], latent_dim=2, lengthscale=0.7,
        variance=base.VAR_INIT, sigma=b['sigma'], sigma_drift_sq=b['sigma'] ** 2,
        eps_grid=1e-3, S_marginal=2, n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=rho, learn_emissions=False, update_R=False, learn_kernel=True,
        n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=b['xt'],
        estep_method='gmix', verbose=False)
    jax.block_until_ready(mp['m'])
    ls = float(hist.lengthscale[-1]) if len(hist.lengthscale) else float('nan')
    return time.perf_counter() - t0, ls


def main():
    print(f"device={jax.devices()} backend={jax.default_backend()}", flush=True)
    T = 10000
    xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(T, 0)
    xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
    b = dict(lik=lik, op=op, ip=ip, t_grid=t_grid, sigma=sigma, xt=xt)

    print("\n===== WARM-UP fit (n_em=3, n_estep=10) — populate shared caches =====", flush=True)
    os.environ['EFGP_PROFILE_ITERS'] = ''      # quiet during warm-up
    wu, _ = fit(b, 3, 10)
    print(f"warm-up wall={wu:.1f}s", flush=True)

    print("\n===== MEASURED fit (n_em=50, n_estep=10) per-iter =====", flush=True)
    os.environ['EFGP_PROFILE_ITERS'] = '1'
    tot, ls = fit(b, 50, 10)
    print(f"\nTOTAL n_em=50 wall = {tot:.1f}s   ls_recovered≈{ls:.3f}", flush=True)


if __name__ == '__main__':
    main()
