"""
Test the kernel_warmup_iters fix.

Shows that increasing kernel_warmup_iters from 2 to 8 (or 10) prevents ℓ collapse
and produces learned hyperparameters consistent with SparseGP's behavior.

Run with: ~/myenv/bin/python demos/diag_warmup_fix.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import time

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
import sing.efgp_em as em

print("JAX devices:", jax.devices())

D = 2; T = 300; t_max = 6.0; sigma_diffusion = 0.3; N_obs = 8
A_true = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([2.0, 0.0]),
                  f=lambda x, t: A_true @ x, t_max=t_max, n_timesteps=T,
                  sigma=lambda x, t: sigma_diffusion * jnp.eye(D))
t_grid = jnp.linspace(0.0, t_max, T)
xs_np = np.asarray(xs)

rng = np.random.default_rng(0)
C_true = rng.standard_normal((N_obs, D)) * 0.5
ys = simulate_gaussian_obs(jr.PRNGKey(1), xs,
    dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.05)))

class _GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean)**2 + var) / R)

lik = _GLik(ys[None], jnp.ones((1, T), dtype=bool))
yc = ys - ys.mean(0)
U, s, Vt = np.linalg.svd(yc.T, full_matrices=False)
C0 = jnp.asarray(U[:, :D] * s[:D] / np.sqrt(T))
ip_init = {'mu0': jnp.zeros(D)[None], 'V0': (jnp.eye(D) * 0.5)[None]}
op_init = dict(C=C0, d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.1))

init_ls = 1.5; init_var = 1.0
n_em = 25; n_estep = 8; n_mstep = 10; mstep_lr = 0.01
rho_sched = jnp.logspace(-2, -1, n_em)
X_template = jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))

def procrustes_rmse(inferred, true):
    A = np.linalg.lstsq(inferred, true, rcond=None)[0]
    return float(np.sqrt(np.mean((inferred @ A - true)**2)))

for warmup in [2, 8]:
    print(f"\n{'='*55}")
    print(f"kernel_warmup_iters={warmup} ({n_em} total EM, n_mstep={n_mstep})")
    t0 = time.time()
    mp, nat_p, dp, ip, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma_diffusion,
        rho_sched=rho_sched,
        n_em_iters=n_em, n_estep_iters=n_estep,
        learn_kernel=True,
        n_mstep_iters=n_mstep, mstep_lr=mstep_lr,
        kernel_warmup_iters=warmup,
        X_template=X_template, seed=42,
    )
    elapsed = time.time() - t0
    ms = np.asarray(mp['m'][0])
    rmse = procrustes_rmse(ms, xs_np)
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Final ℓ={hist.lengthscale[-1]:.4f}, σ²={hist.variance[-1]:.4f}")
    print(f"  Latent RMSE: {rmse:.4f}")
    print(f"  ℓ trajectory (every 5 iters): {[f'{v:.3f}' for v in hist.lengthscale[4::5]]}")

print(f"\n{'='*55}")
print("Reference: SparseGP ℓ ≈ 1.43-1.83 on this problem (from benchmarks).")
print("With warmup=2: expect ℓ < 1.0 (collapse).")
print("With warmup=8: expect ℓ > 1.0 (correct direction).")
