"""Quick correctness check for the jitted M-step RHS + hoisted M-step jit.
Runs a short T=2000 fit (past kernel_warmup) and prints per-iter timing +
recovered ls/var. Should complete without error and recover sane ls (~1).
"""
import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ['EFGP_PROFILE_ITERS'] = '1'
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base

xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(2000, 0)
xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
rho = jnp.linspace(0.05, 0.7, 14)
mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
    likelihood=lik, t_grid=t_grid, output_params=op, init_params=ip,
    latent_dim=2, lengthscale=0.7, variance=base.VAR_INIT, sigma=sigma,
    sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2, n_em_iters=14,
    n_estep_iters=10, rho_sched=rho, learn_emissions=False, update_R=False,
    learn_kernel=True, n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
    n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=xt,
    estep_method='gmix', verbose=False)
jax.block_until_ready(mp['m'])
print(f"\nOK: ls={float(hist.lengthscale[-1]):.3f}  var={float(hist.variance[-1]):.3f} "
      f"(LS_TRUE={base.LS_TRUE}, VAR_TRUE={base.VAR_TRUE})")
