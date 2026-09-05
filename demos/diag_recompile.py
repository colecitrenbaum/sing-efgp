"""Diagnostic: count XLA (re)compilations during a WARM SING fit.

If the warm fit recompiles inside the Python EM loop, JAX_LOG_COMPILES=1 will
print a "Compiling <fn>" line per retrace.  We run one cold fit (populate
caches), then a second (warm) fit and count/report compile events emitted
during it.  Compile behaviour (count + which fn) is platform-independent, so
this is valid on CPU.

Run: JAX_LOG_COMPILES=1 JAX_PLATFORMS=cpu python demos/diag_recompile.py 2> err.txt
"""
from __future__ import annotations
import sys, os, io, time, logging
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base


def fit(b, n_em, n_estep):
    rho = jnp.linspace(0.05, 0.7, n_em)
    mp, _, _, _, _ = base.efgp_em.fit_efgp_sing_jax(
        likelihood=b['lik'], t_grid=b['t_grid'], output_params=b['op'],
        init_params=b['ip'], latent_dim=2, lengthscale=0.7,
        variance=base.VAR_INIT, sigma=b['sigma'], sigma_drift_sq=b['sigma'] ** 2,
        eps_grid=1e-3, S_marginal=2, n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=rho, learn_emissions=False, update_R=False, learn_kernel=True,
        n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=b['xt'],
        estep_method='gmix', verbose=False)
    jax.block_until_ready(mp['m'])


class CountHandler(logging.Handler):
    def __init__(self):
        super().__init__(); self.records = []
    def emit(self, r):
        m = r.getMessage()
        if 'Compiling' in m or 'compiling' in m:
            self.records.append(m.split('\n')[0][:110])


def main():
    T = 2000
    xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(T, 0)
    xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
    b = dict(lik=lik, op=op, ip=ip, t_grid=t_grid, sigma=sigma, xt=xt)
    n_em, n_estep = 14, 6   # >warmup(8) so several M-steps run

    print(f"device={jax.devices()} T={T} n_em={n_em} n_estep={n_estep}")
    print("=== COLD fit (populate caches) ===", flush=True)
    fit(b, n_em, n_estep)

    # Attach a counting handler to the jax logger for the WARM fit only.
    h = CountHandler()
    jlog = logging.getLogger("jax")
    jlog.addHandler(h); jlog.setLevel(logging.DEBUG)
    print("=== WARM fit (counting compiles) ===", flush=True)
    t0 = time.perf_counter()
    fit(b, n_em, n_estep)
    wall = time.perf_counter() - t0
    jlog.removeHandler(h)

    print(f"\nWARM wall = {wall:.1f}s")
    print(f"compile events during WARM fit: {len(h.records)}")
    from collections import Counter
    c = Counter(r.split('Compiling')[-1].split(' with')[0].strip()
                if 'Compiling' in r else r for r in h.records)
    for name, cnt in c.most_common(20):
        print(f"  {cnt:3d}x  {name[:80]}")


if __name__ == '__main__':
    main()
