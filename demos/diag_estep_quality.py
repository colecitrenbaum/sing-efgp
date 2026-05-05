"""
Diagnostic: why does EFGP E-step plateau at RMSE~0.18 while SparseGP reaches 0.076?

Tests:
  A. Convergence speed: run both for 20 / 40 / 80 fixed-ℓ EM iters.
     If gap closes → EFGP is just slower.
  B. Grid density: run EFGP with K_per_dim=10/20 (M=441/1681) at 20 iters.
     If dense grid fixes it → spectral approximation is too coarse.
  C. rho schedule: run EFGP with 10x faster rho (logspace(-1,0)) at 20 iters.
     If fast rho fixes it → SING step size is the bottleneck.

Prints per-iter RMSE trajectory for each config so we can see the shape
of convergence (steady improvement vs. plateau vs. oscillation).

Run with: ~/myenv/bin/python demos/diag_estep_quality.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial
import time

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import (
    natural_to_marginal_params, natural_to_mean_params,
    dynamics_to_natural_params, InitParams,
)
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

print("JAX devices:", jax.devices())

# ---------------------------------------------------------------------------
# Problem setup (same as all diag scripts)
# ---------------------------------------------------------------------------
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
trial_mask = jnp.ones((1, T), dtype=bool)
yc = ys - ys.mean(0)
U, s, Vt = np.linalg.svd(yc.T, full_matrices=False)
C0 = jnp.asarray(U[:, :D] * s[:D] / np.sqrt(T))
ip_init = {'mu0': jnp.zeros(D)[None], 'V0': (jnp.eye(D) * 0.5)[None]}
op_init = dict(C=C0, d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.1))

init_ls = 1.5; init_var = 1.0; sigma = sigma_diffusion
X_template = jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))
n_estep = 8

def procrustes_rmse(inferred, true):
    A = np.linalg.lstsq(inferred, true, rcond=None)[0]
    return float(np.sqrt(np.mean((inferred @ A - true)**2)))

# ---------------------------------------------------------------------------
# A. Convergence speed: both methods for 20 / 40 / 80 EM iters
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("TEST A — Convergence speed (fixed ℓ=1.5, vary n_em)")
print("="*65)

for n_em in [20, 40, 80]:
    rho_sched = jnp.logspace(-2, -1, n_em)    # 0.01→0.1

    # --- EFGP ---
    t0 = time.time()
    mp_ef, _, _, _, hist_ef = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma, rho_sched=rho_sched,
        n_em_iters=n_em, n_estep_iters=n_estep,
        learn_kernel=False,
        X_template=X_template, seed=42, verbose=False,
    )
    rmse_ef = procrustes_rmse(np.asarray(mp_ef['m'][0]), xs_np)
    t_ef = time.time() - t0

    # --- SparseGP ---
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    dp0 = dict(length_scales=jnp.full((D,), init_ls), output_scale=jnp.asarray(init_var))
    t0 = time.time()
    mp_sp, _, _, _, _, op_sp, _, elbos_sp = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=dp0, init_params=ip_init, output_params=op_init,
        sigma=sigma, rho_sched=rho_sched,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=0.01 * jnp.ones(n_em), print_interval=9999,
    )
    rmse_sp = procrustes_rmse(np.asarray(mp_sp['m'][0]), xs_np)
    t_sp = time.time() - t0

    print(f"\nn_em={n_em}  (rho {float(rho_sched[0]):.3f}→{float(rho_sched[-1]):.3f})")
    print(f"  EFGP    RMSE={rmse_ef:.4f}  ({t_ef:.0f}s)")
    print(f"  SparseGP RMSE={rmse_sp:.4f}  ({t_sp:.0f}s)")
    print(f"  ratio: EFGP/SP = {rmse_ef/rmse_sp:.2f}x")

# ---------------------------------------------------------------------------
# B. Grid density: K_per_dim=10 vs 20 vs 30 at 20 EM iters
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("TEST B — Grid density (fixed ℓ=1.5, 20 EM iters)")
print("="*65)

n_em = 20
rho_sched = jnp.logspace(-2, -1, n_em)

for K in [10, 20, 30]:
    t0 = time.time()
    mp_ef, _, _, _, hist_ef = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma, rho_sched=rho_sched,
        n_em_iters=n_em, n_estep_iters=n_estep,
        learn_kernel=False, K_per_dim=K,
        X_template=X_template, seed=42, verbose=False,
    )
    rmse = procrustes_rmse(np.asarray(mp_ef['m'][0]), xs_np)
    t_el = time.time() - t0
    # Compute M
    grid_tmp = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(init_ls, jnp.float32)),
        log_var=jnp.log(jnp.asarray(init_var, jnp.float32)),
        K_per_dim=K,
        X_extent=float((np.asarray(X_template).max(0) - np.asarray(X_template).min(0)).max()),
        xcen=jnp.zeros(D, jnp.float32), d=D, eps=1e-2,
    )
    print(f"  K_per_dim={K:2d}  M={grid_tmp.M:5d}  RMSE={rmse:.4f}  ({t_el:.0f}s)")

# SparseGP reference at n_em=20
zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
quad = GaussHermiteQuadrature(D=D, n_quad=5)
sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
dp0 = dict(length_scales=jnp.full((D,), init_ls), output_scale=jnp.asarray(init_var))
t0 = time.time()
mp_sp20, *_ = fit_variational_em(
    key=jr.PRNGKey(33), fn=sparse_drift, likelihood=lik, t_grid=t_grid,
    drift_params=dp0, init_params=ip_init, output_params=op_init,
    sigma=sigma, rho_sched=rho_sched,
    n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
    perform_m_step=False, learn_output_params=False,
    learning_rate=0.01 * jnp.ones(n_em), print_interval=9999,
)
rmse_sp20 = procrustes_rmse(np.asarray(mp_sp20['m'][0]), xs_np)
print(f"  SparseGP (64 inducing)      RMSE={rmse_sp20:.4f}  ({time.time()-t0:.0f}s)  [reference]")

# ---------------------------------------------------------------------------
# C. rho schedule: fast vs slow at 20 EM iters
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("TEST C — rho schedule speed (fixed ℓ=1.5, 20 EM iters, K=10)")
print("="*65)

n_em = 20
schedules = {
    'slow  logspace(-2,-1)': jnp.logspace(-2, -1, n_em),   # 0.01→0.1
    'mid   logspace(-1, 0)': jnp.logspace(-1,  0, n_em),   # 0.1→1.0
    'fast  ones*0.3':        jnp.full(n_em, 0.3),           # constant 0.3
}

for label, rho_s in schedules.items():
    # EFGP
    t0 = time.time()
    mp_ef, _, _, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma, rho_sched=rho_s,
        n_em_iters=n_em, n_estep_iters=n_estep,
        learn_kernel=False,
        X_template=X_template, seed=42, verbose=False,
    )
    rmse_ef = procrustes_rmse(np.asarray(mp_ef['m'][0]), xs_np)
    t_ef = time.time() - t0

    # SparseGP
    zs2 = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    quad2 = GaussHermiteQuadrature(D=D, n_quad=5)
    sd2 = SparseGP(zs=zs2, kernel=RBF(latent_dim=D), expectation=quad2)
    dp02 = dict(length_scales=jnp.full((D,), init_ls), output_scale=jnp.asarray(init_var))
    t0 = time.time()
    mp_sp2, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sd2, likelihood=lik, t_grid=t_grid,
        drift_params=dp02, init_params=ip_init, output_params=op_init,
        sigma=sigma, rho_sched=rho_s,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=0.01 * jnp.ones(n_em), print_interval=9999,
    )
    rmse_sp = procrustes_rmse(np.asarray(mp_sp2['m'][0]), xs_np)
    t_sp = time.time() - t0

    print(f"  {label}:  EFGP={rmse_ef:.4f} ({t_ef:.0f}s)  SP={rmse_sp:.4f} ({t_sp:.0f}s)  ratio={rmse_ef/rmse_sp:.2f}x")

# ---------------------------------------------------------------------------
# D. Per-iteration RMSE trajectory: EFGP vs SparseGP
#    (20 EM iters, record RMSE after each outer EM step)
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("TEST D — Per-iteration RMSE: does EFGP gap open early or late?")
print("  (20 EM iters, fixed ℓ=1.5, fast rho=0.3)")
print("="*65)

# We'll do this by running fit_efgp_sing_jax with n_em=1 repeatedly,
# warm-starting from the previous state.
# SparseGP: similarly run 1 iter at a time.
# This is slow but gives us the trajectory.

# Actually, simpler: run EFGP/SP for k iters at a time and record RMSE.
rho_fast = jnp.full(20, 0.3)
checkpoints = [1, 2, 3, 5, 8, 12, 20]

print("  iter  EFGP_RMSE  SP_RMSE  ratio")
for n in checkpoints:
    rho_sub = rho_fast[:n]
    mp_ef_k, _, _, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma, rho_sched=rho_sub,
        n_em_iters=n, n_estep_iters=n_estep,
        learn_kernel=False,
        X_template=X_template, seed=42, verbose=False,
    )
    r_ef = procrustes_rmse(np.asarray(mp_ef_k['m'][0]), xs_np)

    zs3 = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    quad3 = GaussHermiteQuadrature(D=D, n_quad=5)
    sd3 = SparseGP(zs=zs3, kernel=RBF(latent_dim=D), expectation=quad3)
    dp03 = dict(length_scales=jnp.full((D,), init_ls), output_scale=jnp.asarray(init_var))
    mp_sp_k, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sd3, likelihood=lik, t_grid=t_grid,
        drift_params=dp03, init_params=ip_init, output_params=op_init,
        sigma=sigma, rho_sched=rho_sub,
        n_iters=n, n_iters_e=n_estep, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=0.01 * jnp.ones(n), print_interval=9999,
    )
    r_sp = procrustes_rmse(np.asarray(mp_sp_k['m'][0]), xs_np)
    print(f"  {n:4d}  {r_ef:.4f}     {r_sp:.4f}   {r_ef/r_sp:.2f}x")

print("\nConclusion guidance:")
print("  If A shows gap closes at 40-80 iters → convergence speed only.")
print("  If B shows gap closes at K=20/30    → grid is too coarse.")
print("  If C shows gap closes with fast rho → rho schedule is bottleneck.")
print("  If D shows gap opens from iter 1    → structural EFGP E-step difference.")
