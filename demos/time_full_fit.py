"""Time a full canonical EFGP fit (post M-step-recompile fix) on GPU.

Answers: how long should a T=10000, n_em=50, n_estep=10 fit take now?
Reports cold (incl. compile) and warm (jit cache reused) total wall, plus the
per-outer-iter M-step cost in isolation (should now be ms, not ~0.7s).

Run on GPU: python demos/time_full_fit.py
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
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
    for T in [10000]:
        xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(T, 0)
        xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
        b = dict(lik=lik, op=op, ip=ip, t_grid=t_grid, sigma=sigma, xt=xt)
        print(f"\n### T={T}, canonical n_em=50 n_estep=10 ###", flush=True)
        # small warm-up fit to populate shared-primitive caches (nufft etc.)
        fit(b, 3, 4)
        c50, ls_c = fit(b, 50, 10)
        w50, ls_w = fit(b, 50, 10)
        print(f"  n_em=50 n_estep=10:  cold={c50:6.1f}s  warm={w50:6.1f}s  "
              f"(ls_recovered≈{ls_w:.3f})", flush=True)
        # scaling refs
        c20, _ = fit(b, 20, 10); w20, _ = fit(b, 20, 10)
        print(f"  n_em=20 n_estep=10:  cold={c20:6.1f}s  warm={w20:6.1f}s", flush=True)
        per_outer = (w50 - w20) / 30 * 1000
        print(f"  => per-outer-iter (warm) ≈ {per_outer:.1f} ms  "
              f"(× ~42 M-step iters dominated M-step recompile before fix)", flush=True)


if __name__ == '__main__':
    main()
