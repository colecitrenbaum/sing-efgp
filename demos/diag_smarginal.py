"""
Diagnostic: does S_marginal (MC sample count for E_q[f(x)]) explain the RMSE gap?

EFGP uses S_marginal samples from q(x_t)=N(m_t,S_t) per time step to form
the BTTB kernel and RHS h_r. Default is S_marginal=2. SparseGP uses
GaussHermite quadrature with n_quad^D = 5^2 = 25 deterministic points.

If S_marginal=2 is causing high variance in drift moments, increasing it
should close the RMSE gap toward SparseGP.

Run with: ~/myenv/bin/python demos/diag_smarginal.py
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
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs
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
U, s, _ = np.linalg.svd(yc.T, full_matrices=False)
C0 = jnp.asarray(U[:, :D] * s[:D] / np.sqrt(T))
ip_init = {'mu0': jnp.zeros(D)[None], 'V0': (jnp.eye(D) * 0.5)[None]}
op_init = dict(C=C0, d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.1))

X_template = jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))
n_em = 20; n_estep = 8
rho_sched = jnp.logspace(-2, -1, n_em)

def procrustes_rmse(m, truth):
    A = np.linalg.lstsq(m, truth, rcond=None)[0]
    return float(np.sqrt(np.mean((m @ A - truth)**2)))

# SparseGP reference (run once)
print("Running SparseGP reference (n_quad=5, 64 inducing pts)...")
zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
quad = GaussHermiteQuadrature(D=D, n_quad=5)
sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
dp0 = dict(length_scales=jnp.full((D,), 1.5), output_scale=jnp.asarray(1.0))
t0 = time.time()
mp_sp, *_ = fit_variational_em(
    key=jr.PRNGKey(33), fn=sparse_drift, likelihood=lik, t_grid=t_grid,
    drift_params=dp0, init_params=ip_init, output_params=op_init,
    sigma=sigma_diffusion, rho_sched=rho_sched,
    n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
    perform_m_step=False, learn_output_params=False,
    learning_rate=0.01 * jnp.ones(n_em), print_interval=9999,
)
rmse_sp = procrustes_rmse(np.asarray(mp_sp['m'][0]), xs_np)
print(f"  SparseGP RMSE={rmse_sp:.4f}  ({time.time()-t0:.0f}s)\n")

# EFGP sweep over S_marginal
print("EFGP sweep over S_marginal (# MC samples per time step):")
print(f"  {'S_marginal':>12}  {'M_pseudo':>10}  {'RMSE':>8}  {'wall(s)':>8}  {'ratio vs SP':>12}")

for S in [1, 2, 5, 10, 20, 50]:
    t0 = time.time()
    mp_ef, *_ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=1.5, variance=1.0,
        sigma=sigma_diffusion, rho_sched=rho_sched,
        n_em_iters=n_em, n_estep_iters=n_estep,
        learn_kernel=False, S_marginal=S,
        X_template=X_template, seed=42, verbose=False,
    )
    rmse = procrustes_rmse(np.asarray(mp_ef['m'][0]), xs_np)
    t_el = time.time() - t0
    n_pseudo = (T - 1) * S
    print(f"  {S:>12}  {n_pseudo:>10}  {rmse:>8.4f}  {t_el:>8.1f}  {rmse/rmse_sp:>12.2f}x")

print(f"\n  SparseGP (GH, n_quad=5): RMSE={rmse_sp:.4f}  (reference)")
print(f"\nIf RMSE drops with larger S → MC variance is the bottleneck.")
print(f"If RMSE plateaus regardless → structural difference (grid / moment formula).")
