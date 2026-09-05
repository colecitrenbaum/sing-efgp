"""THROWAWAY: print the EFGP spectral-grid size M (number of Fourier features)
that the Duffing scaling cells actually use, for comparison with SparseGP's
M=49 inducing points.  Runs n_em_iters=1 with verbose=True.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import demos.bench_duffing_scaling as duffs
import demos.bench_gpdrift_x64 as base
import numpy as np, jax, jax.numpy as jnp

for T in [int(a) for a in sys.argv[1:]] or [1000]:
    xs, lik, op, ip, t_grid, sigma = duffs.make_data(T, 0)
    print(f"\n######## T={T}  x range={np.asarray(xs).min(0)}..{np.asarray(xs).max(0)}",
          flush=True)
    t0 = time.perf_counter()
    mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid, output_params=op, init_params=ip,
        latent_dim=base.D, lengthscale=0.7, variance=base.VAR_INIT,
        sigma=sigma, sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        qf_nufft_eps=1e-4, qf_cg_tol=1e-4,
        n_em_iters=1, n_estep_iters=10,
        rho_sched=jnp.linspace(0.05, 0.7, 1),
        learn_emissions=False, update_R=False, learn_kernel=True,
        n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        restore_qf_variance='none', estep_method='auto',
        qx_moments_method='gmix_batched', verbose=True)
    print(f"  1-iter fit wall={time.perf_counter()-t0:.1f}s "
          f"final_mu_r.shape={hist.final_mu_r.shape}  "
          f"M_modes={hist.final_grid.M}", flush=True)
