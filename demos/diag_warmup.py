"""
Diagnostic: does EFGP M-step direction flip at poor marginals?

Tests EFGP M-step direction from marginals obtained after 1, 2, 5, 10, 20
fixed-ℓ EM iters. Checks if the early (poorly-converged) marginals push ℓ DOWN
while later (well-converged) marginals push ℓ UP.

If early iters push ℓ↓: warmup is too short — kernel_warmup_iters needs increasing.

Run with: ~/myenv/bin/python demos/diag_warmup.py
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
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

print("JAX devices:", jax.devices())

# ---------------------------------------------------------------------------
# 1. Problem setup (same as diag_mstep_agreement.py)
# ---------------------------------------------------------------------------
D = 2; T = 300; t_max = 6.0; sigma_diffusion = 0.3; N_obs = 8

A_true = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
drift_jax = lambda x, t: A_true @ x
sigma_fn = lambda x, t: sigma_diffusion * jnp.eye(D)

xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([2.0, 0.0]),
                  f=drift_jax, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
t_grid = jnp.linspace(0.0, t_max, T)
xs_np = np.asarray(xs)

rng = np.random.default_rng(0)
C_true = rng.standard_normal((N_obs, D)) * 0.5
out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs), R=jnp.full((N_obs,), 0.05))
ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)

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

init_ls = 1.5; init_var = 1.0; sigma = sigma_diffusion
X_template = jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))
t_grid_arr = jnp.linspace(0.0, t_max, T)
del_t = t_grid_arr[1:] - t_grid_arr[:-1]

# Build grid (same as EM loop uses)
X_extent = float((np.asarray(X_template).max(0) - np.asarray(X_template).min(0)).max())
xcen_arr = jnp.zeros(D, dtype=jnp.float32)
grid = jp.spectral_grid_se_fixed_K(
    log_ls=jnp.log(jnp.asarray(init_ls, dtype=jnp.float32)),
    log_var=jnp.log(jnp.asarray(init_var, dtype=jnp.float32)),
    K_per_dim=10, X_extent=X_extent, xcen=xcen_arr, d=D, eps=1e-2,
)
print(f"Grid: M={grid.M}")

def procrustes_rmse(inferred, true):
    A_proc = np.linalg.lstsq(inferred, true, rcond=None)[0]
    return float(np.sqrt(np.mean((inferred @ A_proc - true)**2)))

def run_mstep_from_marginals(mp, label, n_adam=10):
    """Compute EFGP M-step direction from given marginal params."""
    ms = jnp.asarray(mp['m'][0])
    Ss = jnp.asarray(mp['S'][0])
    SSs = jnp.asarray(mp['SS'][0])

    mu_r, X_ps, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(0),
        sigma_drift_sq=sigma**2, S_marginal=2, D_lat=D, D_out=D,
    )
    ws0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws0) < 1e-30, jnp.array(1e-30, dtype=ws0.dtype), ws0)
    A0 = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h_r0 = jax.vmap(A0)(mu_r)
    z_r = h_r0 / ws_safe

    log_ls_new, log_var_new, lh = jpd.m_step_kernel_jax(
        np.log(init_ls), np.log(init_var),
        mu_r_fixed=mu_r, z_r=z_r, top=top,
        xis_flat=grid.xis_flat, h_per_dim=grid.h_per_dim,
        D_lat=D, D_out=D,
        n_inner=n_adam, lr=0.01,
        n_hutchinson=16, include_trace=True,
        key=jr.PRNGKey(42),
    )
    ls_new = float(jnp.exp(log_ls_new))
    direction = "↑" if ls_new > init_ls else "↓"
    print(f"  {label}: latent_rmse={procrustes_rmse(np.asarray(mp['m'][0]), xs_np):.4f}  "
          f"ℓ: {init_ls:.3f} → {ls_new:.4f} {direction}  "
          f"loss[0,last]: [{lh[0]:.2f}, {lh[-1]:.2f}]")
    return ls_new

# ---------------------------------------------------------------------------
# 2. Run EFGP with fixed ℓ for increasing numbers of EM iters
#    then check M-step direction from those marginals
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"EFGP M-step direction vs E-step convergence (n_adam=10, lr=0.01)")
print(f"{'='*65}")
print(f"  {'Warmup iters':<15}  {'lat RMSE':<10}  {'ℓ after 10 Adam steps':<25}  direction")

# Checkpoints to test
checkpoints = [1, 2, 5, 10, 15, 20]
rho_sched_full = jnp.logspace(-2, -1, max(checkpoints))

prev_n = 0
mp_curr = None

for n_iters in checkpoints:
    n_more = n_iters - prev_n
    rho_sub = rho_sched_full[prev_n:n_iters]

    # Run EFGP from current state (warm-start)
    # NOTE: fit_efgp_sing_jax doesn't support warm-start, so we re-run from scratch each time
    mp_ef, nat_p_ef, dp_ef, ip_ef, hist_ef = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op_init, init_params=ip_init,
        latent_dim=D, lengthscale=init_ls, variance=init_var,
        sigma=sigma,
        rho_sched=rho_sched_full[:n_iters],
        n_em_iters=n_iters, n_estep_iters=8,
        learn_kernel=False,
        X_template=X_template, seed=42, verbose=False,
    )
    print(f"\nAfter {n_iters} fixed-ℓ EM iters:")
    ls_result = run_mstep_from_marginals(mp_ef, f"{n_iters}-iter EFGP marginals", n_adam=10)

print(f"\n{'='*65}")
print("Hypothesis check: if early iters give ℓ↓ and late iters give ℓ↑,")
print("  → kernel_warmup_iters needs increasing (currently default=2).")
print("If all push ℓ↑ regardless of convergence,")
print("  → warmup is NOT the issue; E-step inaccuracy still biases M-step but differently.")

# ---------------------------------------------------------------------------
# 3. Quick check: what does the EFGP M-step do in the benchmark setting?
#    (n_adam=10 steps, from the very first iteration)
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print("Benchmark-replication check (2 warmup iters, then 10 M-step Adam steps):")

# Just 2 EM iters (= kernel_warmup_iters default), then M-step
mp_2it, _, _, _, _ = em.fit_efgp_sing_jax(
    likelihood=lik, t_grid=t_grid,
    output_params=op_init, init_params=ip_init,
    latent_dim=D, lengthscale=init_ls, variance=init_var,
    sigma=sigma,
    rho_sched=rho_sched_full[:2],
    n_em_iters=2, n_estep_iters=8,
    learn_kernel=False,
    X_template=X_template, seed=42, verbose=False,
)
print(f"After 2 warmup iters (rho={float(rho_sched_full[0]):.4f},{float(rho_sched_full[1]):.4f}):")
run_mstep_from_marginals(mp_2it, "2-iter EFGP marginals", n_adam=10)

# Also with n_adam=50 to see longer-term direction
print(f"\nSame marginals but 50 Adam steps:")
run_mstep_from_marginals(mp_2it, "2-iter EFGP marginals (50 Adam)", n_adam=50)
