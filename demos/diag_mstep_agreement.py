"""
Diagnostic: shared-smoother M-step agreement (Exp 1 + 4 from plan).

Exp 4 — fixed-hyper E-step comparison:
  Run both methods for N EM iters at fixed ℓ=1.5, compare smoother marginals.
  If marginals agree → divergence is purely in the M-step.
  If marginals differ → E-step is the root cause.

Exp 1 — shared-smoother M-step agreement:
  Feed identical smoother marginals to both M-steps, compare ℓ trajectories.
  If they agree → E-step mismatch caused the benchmark divergence.
  If EFGP ℓ↓ and SparseGP ℓ↑ from same marginals → EFGP M-step is wrong.

Run with: ~/myenv/bin/python demos/diag_mstep_agreement.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import optax
from functools import partial
import time

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em, compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

print("JAX devices:", jax.devices())

# ---------------------------------------------------------------------------
# 1. Problem setup — damped 2D rotation
# ---------------------------------------------------------------------------
D = 2
T = 300              # shorter than full notebook to run faster
t_max = 6.0
sigma_diffusion = 0.3
N_obs = 8

A_true = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

def drift_jax(x, t):   return A_true @ x
def drift_np(x_batch): return np.asarray(x_batch) @ np.asarray(A_true).T

sigma_fn = lambda x, t: sigma_diffusion * jnp.eye(D)
xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([2.0, 0.0]),
                  f=drift_jax, t_max=t_max, n_timesteps=T,
                  sigma=sigma_fn)
t_grid = jnp.linspace(0.0, t_max, T)

rng = np.random.default_rng(0)
C_true = rng.standard_normal((N_obs, D)) * 0.5
out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                R=jnp.full((N_obs,), 0.05))
ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)

xs_np = np.asarray(xs)
print(f"xs: {xs.shape}, range {float(xs.min()):.2f}..{float(xs.max()):.2f}")

# Shared init — lightweight Gaussian likelihood wrapper
class _GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean)**2 + var) / R)

lik = _GLik(ys[None], jnp.ones((1, T), dtype=bool))
t_mask = lik.t_mask
trial_mask = jnp.ones((1, T), dtype=bool)

# PCA init for emissions
yc = ys - ys.mean(0)
U, s, Vt = np.linalg.svd(yc.T, full_matrices=False)
C0 = jnp.asarray(U[:, :D] * s[:D] / np.sqrt(T))
mu0 = jnp.zeros(D)
V0 = jnp.eye(D) * 0.5
ip_init = {'mu0': mu0[None], 'V0': V0[None]}
op_init = dict(C=C0, d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.1))

# Fixed hypers
init_ls = 1.5
init_var = 1.0
sigma = sigma_diffusion
n_em = 20
n_estep = 8
rho_sched = jnp.logspace(-2, -1, n_em)

# ---------------------------------------------------------------------------
# 2. SparseGP — fixed ℓ=1.5 run (no M-step)
# ---------------------------------------------------------------------------
print("\n=== Exp 4: Fixed-ℓ E-step comparison ===")
print("Running SparseGP with perform_m_step=False ...")
zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
quad = GaussHermiteQuadrature(D=D, n_quad=5)
sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
dp_sp_fixed = dict(
    length_scales=jnp.full((D,), init_ls),
    output_scale=jnp.asarray(init_var),
)

t0 = time.time()
mp_sp, nat_p_sp, gp_post_sp, dp_sp_out, ip_sp, op_sp, ie_sp, elbos_sp = fit_variational_em(
    key=jr.PRNGKey(33),
    fn=sparse_drift, likelihood=lik, t_grid=t_grid,
    drift_params=dp_sp_fixed,
    init_params=ip_init, output_params=op_init, sigma=sigma,
    rho_sched=rho_sched,
    n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
    perform_m_step=False, learn_output_params=False,
    learning_rate=0.01 * jnp.ones(n_em),
    print_interval=5,
)
t_sp = time.time() - t0
print(f"SparseGP fixed-ℓ: {t_sp:.1f}s, ELBO={float(elbos_sp[-1]):.2f}")

ms_sp = np.asarray(mp_sp['m'][0])   # (T, D)
Ss_sp = np.asarray(mp_sp['S'][0])   # (T, D, D)
SSs_sp = np.asarray(mp_sp['SS'][0]) # (T-1, D, D)

# Procrustes alignment for RMSE
def procrustes_rmse(inferred, true):
    """Align inferred to true via linear transform, return RMSE."""
    A_proc = np.linalg.lstsq(inferred, true, rcond=None)[0]
    aligned = inferred @ A_proc
    return float(np.sqrt(np.mean((aligned - true)**2))), A_proc

rmse_sp, A_sp = procrustes_rmse(ms_sp, xs_np)
print(f"SparseGP latent RMSE (Procrustes): {rmse_sp:.4f}")

# ---------------------------------------------------------------------------
# 3. EFGP — fixed ℓ=1.5 run (no kernel learning)
# ---------------------------------------------------------------------------
print("\nRunning EFGP with learn_kernel=False ...")
X_template = jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))

t0 = time.time()
mp_ef, nat_p_ef, dp_ef, ip_ef, hist_ef = em.fit_efgp_sing_jax(
    likelihood=lik, t_grid=t_grid,
    output_params=op_init, init_params=ip_init,
    latent_dim=D, lengthscale=init_ls, variance=init_var,
    sigma=sigma,
    rho_sched=rho_sched,
    n_em_iters=n_em, n_estep_iters=n_estep,
    learn_kernel=False,
    X_template=X_template, seed=42,
)
t_ef = time.time() - t0
print(f"EFGP fixed-ℓ: {t_ef:.1f}s")

ms_ef = np.asarray(mp_ef['m'][0])   # (T, D)
Ss_ef = np.asarray(mp_ef['S'][0])   # (T, D, D)
SSs_ef = np.asarray(mp_ef['SS'][0]) # (T-1, D, D)

rmse_ef, A_ef = procrustes_rmse(ms_ef, xs_np)
print(f"EFGP latent RMSE (Procrustes): {rmse_ef:.4f}")

# Mean difference between smoother marginals (in truth space via Procrustes)
ms_sp_aligned = ms_sp @ A_sp
ms_ef_aligned = ms_ef @ A_ef
mean_diff = float(np.sqrt(np.mean((ms_sp_aligned - ms_ef_aligned)**2)))
print(f"\nE-step mean trajectory difference (both aligned to truth): {mean_diff:.4f}")
print("(If << latent RMSE, marginals are similar; if ~same magnitude, E-steps diverge)")

# ---------------------------------------------------------------------------
# 4. Exp 1: Isolated M-step tests from SparseGP marginals
# ---------------------------------------------------------------------------
print("\n=== Exp 1: Isolated M-step from SparseGP marginals ===")

# --- Build EFGP grid and mu_r from SparseGP marginals ---
X_extent = float((np.asarray(X_template).max(0) - np.asarray(X_template).min(0)).max())
xcen_arr = jnp.zeros(D, dtype=jnp.float32)
K_per_dim = 10

grid_sp = jp.spectral_grid_se_fixed_K(
    log_ls=jnp.log(jnp.asarray(init_ls, dtype=jnp.float32)),
    log_var=jnp.log(jnp.asarray(init_var, dtype=jnp.float32)),
    K_per_dim=K_per_dim, X_extent=X_extent, xcen=xcen_arr,
    d=D, eps=1e-2,
)
print(f"Grid: M={grid_sp.M}, mtot_per_dim={grid_sp.mtot_per_dim}")

del_t = t_grid[1:] - t_grid[:-1]

print("Computing mu_r from SparseGP marginals ...")
ms_sp_j = jnp.asarray(ms_sp)
Ss_sp_j = jnp.asarray(Ss_sp)
SSs_sp_j = jnp.asarray(SSs_sp)

mu_r_sp, X_ps_sp, top_sp = jpd.compute_mu_r_jax(
    ms_sp_j, Ss_sp_j, SSs_sp_j, del_t, grid_sp, jr.PRNGKey(0),
    sigma_drift_sq=sigma**2, S_marginal=2, D_lat=D, D_out=D,
)

# Build z_r from SparseGP marginals
ws0_sp = grid_sp.ws.real.astype(grid_sp.ws.dtype)
ws_safe_sp = jnp.where(jnp.abs(ws0_sp) < 1e-30, jnp.array(1e-30, dtype=ws0_sp.dtype), ws0_sp)
A0_sp = jp.make_A_apply(grid_sp.ws, top_sp, sigmasq=1.0)
h_r0_sp = jax.vmap(A0_sp)(mu_r_sp)
z_r_sp = h_r0_sp / ws_safe_sp

print("Running EFGP M-step from SparseGP marginals (ℓ₀=1.5, 50 steps) ...")
t0 = time.time()
log_ls_efgp_from_sp, log_var_efgp_from_sp, lh_sp = jpd.m_step_kernel_jax(
    np.log(init_ls), np.log(init_var),
    mu_r_fixed=mu_r_sp, z_r=z_r_sp, top=top_sp,
    xis_flat=grid_sp.xis_flat, h_per_dim=grid_sp.h_per_dim,
    D_lat=D, D_out=D,
    n_inner=50, lr=0.01,
    n_hutchinson=16, include_trace=True,
    key=jr.PRNGKey(42),
)
ls_efgp_from_sp = float(jnp.exp(log_ls_efgp_from_sp))
var_efgp_from_sp = float(jnp.exp(log_var_efgp_from_sp))
print(f"  → EFGP M-step from SparseGP marginals: ℓ={ls_efgp_from_sp:.4f}, σ²={var_efgp_from_sp:.4f}  ({time.time()-t0:.1f}s)")
print(f"     loss path: {[f'{v:.2f}' for v in lh_sp[::10]]}")

# --- Also run from EFGP marginals for comparison ---
print("\nComputing mu_r from EFGP marginals ...")
ms_ef_j = jnp.asarray(ms_ef)
Ss_ef_j = jnp.asarray(Ss_ef)
SSs_ef_j = jnp.asarray(SSs_ef)

mu_r_ef, X_ps_ef, top_ef = jpd.compute_mu_r_jax(
    ms_ef_j, Ss_ef_j, SSs_ef_j, del_t, grid_sp, jr.PRNGKey(1),
    sigma_drift_sq=sigma**2, S_marginal=2, D_lat=D, D_out=D,
)
ws_safe_ef = ws_safe_sp  # same grid
A0_ef = jp.make_A_apply(grid_sp.ws, top_ef, sigmasq=1.0)
h_r0_ef = jax.vmap(A0_ef)(mu_r_ef)
z_r_ef = h_r0_ef / ws_safe_ef

print("Running EFGP M-step from EFGP marginals (ℓ₀=1.5, 50 steps) ...")
t0 = time.time()
log_ls_efgp_from_ef, log_var_efgp_from_ef, lh_ef = jpd.m_step_kernel_jax(
    np.log(init_ls), np.log(init_var),
    mu_r_fixed=mu_r_ef, z_r=z_r_ef, top=top_ef,
    xis_flat=grid_sp.xis_flat, h_per_dim=grid_sp.h_per_dim,
    D_lat=D, D_out=D,
    n_inner=50, lr=0.01,
    n_hutchinson=16, include_trace=True,
    key=jr.PRNGKey(42),
)
ls_efgp_from_ef = float(jnp.exp(log_ls_efgp_from_ef))
var_efgp_from_ef = float(jnp.exp(log_var_efgp_from_ef))
print(f"  → EFGP M-step from EFGP marginals: ℓ={ls_efgp_from_ef:.4f}, σ²={var_efgp_from_ef:.4f}  ({time.time()-t0:.1f}s)")
print(f"     loss path: {[f'{v:.2f}' for v in lh_ef[::10]]}")

# --- SparseGP M-step from its own marginals ---
print("\nRunning SparseGP M-step standalone (ℓ₀=1.5, 50 steps via explicit Adam) ...")

# Need Ap from natural_to_marginal_params
_, Ap_sp = vmap(natural_to_marginal_params)(nat_p_sp, trial_mask)

inputs_dummy = jnp.zeros((1, T, 1))

def loss_fn_sp(key, dp):
    return -compute_elbo_over_batch(
        key, lik.ys_obs, t_mask, trial_mask,
        sparse_drift, lik, t_grid, dp, ip_sp,
        op_sp, nat_p_sp, mp_sp, Ap_sp,
        inputs_dummy, ie_sp, sigma,
    )[0]

loss_fn_sp_jit = jax.jit(jax.value_and_grad(loss_fn_sp, argnums=1))

dp_test = dict(
    length_scales=jnp.full((D,), init_ls),
    output_scale=jnp.asarray(init_var),
    inducing_points=zs,
)
opt_sp = optax.adam(0.01)
opt_state_sp = opt_sp.init(dp_test)
ls_sp_path = []

t0 = time.time()
# Warm up JIT
_ = loss_fn_sp_jit(jr.PRNGKey(0), dp_test)
for step in range(50):
    loss, grads = loss_fn_sp_jit(jr.PRNGKey(step), dp_test)
    updates, opt_state_sp = opt_sp.update(grads, opt_state_sp, dp_test)
    dp_test = optax.apply_updates(dp_test, updates)
    ls_sp_path.append(float(jnp.mean(dp_test['length_scales'])))

ls_sp_final_isolated = ls_sp_path[-1]
var_sp_final_isolated = float(dp_test['output_scale'])**2
print(f"  → SparseGP M-step from SparseGP marginals: ℓ={ls_sp_final_isolated:.4f}, σ²={var_sp_final_isolated:.4f}  ({time.time()-t0:.1f}s)")
print(f"     ls path[::10]: {[f'{v:.4f}' for v in ls_sp_path[::10]]}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nExp 4 — E-step comparison (fixed ℓ=1.5, {n_em} EM iters):")
print(f"  EFGP    latent RMSE: {rmse_ef:.4f}")
print(f"  SparseGP latent RMSE: {rmse_sp:.4f}")
print(f"  Smoother mean diff (Procrustes): {mean_diff:.4f}")
if mean_diff < 0.5 * (rmse_ef + rmse_sp):
    print("  → Marginals are SIMILAR. Divergence is in the M-step.")
else:
    print("  → Marginals DIFFER significantly. E-step is contributing to divergence.")

print(f"\nExp 1 — M-step from SparseGP marginals (ℓ₀=1.5, 50 Adam steps, lr=0.01):")
print(f"  EFGP M-step: ℓ={ls_efgp_from_sp:.4f}, σ²={var_efgp_from_sp:.4f}")
print(f"  SparseGP M-step: ℓ={ls_sp_final_isolated:.4f}, σ²={var_sp_final_isolated:.4f}")

efgp_direction = "↑" if ls_efgp_from_sp > init_ls else "↓"
sp_direction = "↑" if ls_sp_final_isolated > init_ls else "↓"
print(f"  EFGP direction: {efgp_direction}  ({init_ls:.2f} → {ls_efgp_from_sp:.4f})")
print(f"  SparseGP direction: {sp_direction}  ({init_ls:.2f} → {ls_sp_final_isolated:.4f})")

if (efgp_direction == sp_direction):
    print("  → M-steps AGREE on direction. Root cause is elsewhere (E-step or schedule).")
else:
    print("  → M-steps DISAGREE on direction. EFGP M-step or SparseGP M-step has a bug.")

print(f"\nCross-check: EFGP M-step from EFGP vs SparseGP marginals:")
print(f"  From SparseGP marginals: ℓ={ls_efgp_from_sp:.4f}")
print(f"  From EFGP marginals:     ℓ={ls_efgp_from_ef:.4f}")
if abs(ls_efgp_from_sp - ls_efgp_from_ef) < 0.2:
    print("  → EFGP M-step gives similar ℓ from both marginals → E-step diff is small.")
else:
    print("  → EFGP M-step gives different ℓ depending on marginals → E-step diff matters.")
