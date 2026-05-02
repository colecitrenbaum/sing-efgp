"""
EFGP-SING demo on a small d=2 synthetic latent SDE.

Run as a script first to keep iteration fast; converted to a notebook below.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure both gp-quadrature and the cloned SING repo are importable.
_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
if str(_GP_QUAD) not in sys.path:
    sys.path.insert(0, str(_GP_QUAD))
_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import math
import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import SquaredExponential, GPParams
from sing.efgp_drift import EFGPDrift
from sing.efgp_em import fit_efgp_sing
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs

from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs

OUT_DIR = Path(__file__).resolve().parent / "_efgp_demo_out"
OUT_DIR.mkdir(exist_ok=True)
print(f"writing diagnostics to {OUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Synthetic data: a stable linear-drift latent SDE in d=2
# -----------------------------------------------------------------------------
D = 2
T = 60
sigma = 0.4
A_true = jnp.array([[-1.5, 0.0], [0.0, -1.5]])

def drift_true_jax(x, t):
    return A_true @ x


def drift_true_np(x):
    """For evaluating on a grid (NumPy)."""
    return x @ np.asarray(A_true).T   # x: (N, D) → (N, D), since (Ax)_i = A x_i


def sigma_fn(x, t):
    return sigma * jnp.eye(D)


print("Simulating SDE...")
xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.2, -0.8]),
                  f=drift_true_jax, t_max=2.0, n_timesteps=T, sigma=sigma_fn)
print(f"  xs shape={xs.shape}, range=({float(xs.min()):.2f}, {float(xs.max()):.2f})")

# Gaussian observations
N = 8
rng_np = np.random.default_rng(0)
C_true_np = rng_np.standard_normal((N, D)) * 0.5
out_true = dict(C=jnp.asarray(C_true_np), d=jnp.zeros(N), R=jnp.full((N,), 0.05))
ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
print(f"  ys shape={ys.shape}")


# -----------------------------------------------------------------------------
# 2. Likelihood + initialisation (PCA for C, d)
# -----------------------------------------------------------------------------
class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

ys_centered = ys - ys.mean(axis=0)
_, _, Vt = jnp.linalg.svd(ys_centered, full_matrices=False)
C_init = Vt[:D].T
d_init = ys.mean(axis=0)
output_params = dict(C=C_init, d=d_init, R=jnp.full((N,), 0.1))

init_params = jax.tree_util.tree_map(
    lambda x: x[None],
    dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1)
)
t_grid = jnp.linspace(0., 2.0, T)


# -----------------------------------------------------------------------------
# 3. Fit EFGP-SING
# -----------------------------------------------------------------------------
print("\n=== Fitting EFGP-SING ===")
ker_efgp = SquaredExponential(dimension=D, init_lengthscale=0.7,
                              init_variance=1.0)
gp_efgp = GPParams(kernel=ker_efgp, init_sig2=1.0)
drift_efgp = EFGPDrift(kernel=ker_efgp, gp_params=gp_efgp, latent_dim=D,
                       sigma_drift_sq=sigma ** 2,
                       eps_grid=1e-2,
                       S_marginal=2, J_hutchinson=2,
                       include_variance_in_ff=False)

# v0 caveat: the collapsed kernel M-step is gradient-descent on a
# log-marginal-likelihood that prefers ℓ → 0 when q(f) is still trivial
# (cold-start collapse).  For a clean demo we keep ℓ fixed at the init
# value; the kernel-learning path is exercised by the unit test
# ``test_collapsed_mstep_gradient_matches_finite_difference``.  See the
# README for a longer discussion.
t0 = time.time()
mp_efgp, _, op_efgp, _, hist_efgp = fit_efgp_sing(
    drift=drift_efgp, likelihood=lik, t_grid=t_grid,
    output_params=output_params, init_params=init_params,
    sigma=sigma, n_em_iters=8, n_estep_iters=10, n_mstep_iters=0,
    rho_sched=jnp.linspace(0.2, 0.9, 8),
    learn_emissions=True, learn_kernel=False, update_R=False,
    mstep_lr=0.05,
    true_xs=np.asarray(xs), verbose=True,
)
print(f"  EFGP-SING wall time: {time.time() - t0:.1f}s, "
      f"M={drift_efgp.cache.grid.M}")


# -----------------------------------------------------------------------------
# 4. Fit SING-SparseGP baseline (matched inducing-grid M ≈ EFGP M)
# -----------------------------------------------------------------------------
print("\n=== Fitting SING-SparseGP baseline ===")
quad = GaussHermiteQuadrature(D=D, n_quad=5)
zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)         # 64 inducing pts
sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
sparse_drift_params = dict(length_scales=jnp.full((D,), 0.7),
                            output_scale=jnp.asarray(1.0))
t0 = time.time()
mp_sp, _, _, _, _, op_sp, _, _ = fit_variational_em(
    key=jr.PRNGKey(33),
    fn=sparse_drift, likelihood=lik, t_grid=t_grid,
    drift_params=sparse_drift_params,
    init_params=init_params, output_params=output_params,
    sigma=sigma,
    rho_sched=jnp.linspace(0.2, 0.9, 8),
    n_iters=8, n_iters_e=10, n_iters_m=4,
    perform_m_step=True, learn_output_params=True,
    learning_rate=jnp.full((8,), 0.05),
    print_interval=1,
)
print(f"  SparseGP wall time: {time.time() - t0:.1f}s, n_inducing=64")


# -----------------------------------------------------------------------------
# 5. Diagnostics: predictive observation RMSE + drift recovery + lengthscale
# -----------------------------------------------------------------------------
def predict_obs_rmse(mp, output_params, ys):
    """Reconstructed-observation RMSE using the inferred latent posterior mean.

    Basis-invariant: invariant under any (C, d, x) transformation that
    preserves Cx + d.
    """
    m = np.asarray(mp['m'][0])
    C = np.asarray(output_params['C']); d = np.asarray(output_params['d'])
    y_hat = m @ C.T + d
    return float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))


obs_rmse_efgp = predict_obs_rmse(mp_efgp, op_efgp, ys)
obs_rmse_sp = predict_obs_rmse(mp_sp, op_sp, ys)
print(f"\n=== Predictive observation RMSE ===")
print(f"  EFGP-SING : {obs_rmse_efgp:.4f}")
print(f"  SparseGP  : {obs_rmse_sp:.4f}")
print(f"  trivial (mean obs): "
      f"{float(np.sqrt(np.mean((ys - ys.mean(axis=0))**2))):.4f}")


# -----------------------------------------------------------------------------
# 6. Drift posterior on a grid (in the latent posterior basis)
# -----------------------------------------------------------------------------
print("\n=== Drift recovery diagnostic ===")
m_efgp_np = np.asarray(mp_efgp['m'][0])
grid_lo = m_efgp_np.min(axis=0) - 0.5
grid_hi = m_efgp_np.max(axis=0) + 0.5
xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
GX, GY = np.meshgrid(xs_g, ys_g, indexing="ij")
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)

# EFGP posterior drift on the grid
X_eval = torch.tensor(grid_pts, dtype=drift_efgp.backend.rdtype)
f_efgp = drift_efgp._eval_mean_torch(X_eval).cpu().numpy()
f_efgp = f_efgp.reshape(GX.shape + (D,))

# True drift on the grid (in TRUTH basis — the EFGP posterior is in the
# inferred basis, so absolute drift values won't match; pattern should).
f_true = drift_true_np(grid_pts).reshape(GX.shape + (D,))


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].quiver(GX, GY, f_true[..., 0], f_true[..., 1], color="k")
axes[0].plot(np.asarray(xs)[:, 0], np.asarray(xs)[:, 1], "r-", alpha=0.5,
             label="true latent path")
axes[0].set_title("True drift (truth basis)")
axes[0].legend()
axes[0].set_aspect("equal")

axes[1].quiver(GX, GY, f_efgp[..., 0], f_efgp[..., 1], color="b")
axes[1].plot(m_efgp_np[:, 0], m_efgp_np[:, 1], "g-", alpha=0.7,
             label="inferred posterior mean")
axes[1].set_title(f"EFGP-SING drift (M={drift_efgp.cache.grid.M})")
axes[1].legend()
axes[1].set_aspect("equal")

# SparseGP drift on the same grid (its update_dynamics_params expects
# inputs with a leading batch dim that matches the marginal_params batch).
gp_post_sp = sparse_drift.update_dynamics_params(
    jr.PRNGKey(0), t_grid, mp_sp, jnp.ones((1, T), dtype=bool),
    sparse_drift_params,
    jnp.zeros((1, T, 1)),                  # (B, T, n_inputs)
    jnp.zeros((D, 1)), sigma)
f_sp = np.asarray(sparse_drift.get_posterior_f_mean(
    gp_post_sp, sparse_drift_params, jnp.asarray(grid_pts))).reshape(GX.shape + (D,))
axes[2].quiver(GX, GY, f_sp[..., 0], f_sp[..., 1], color="g")
m_sp_np = np.asarray(mp_sp['m'][0])
axes[2].plot(m_sp_np[:, 0], m_sp_np[:, 1], "g-", alpha=0.7)
axes[2].set_title(f"SparseGP drift (M=64 inducing)")
axes[2].set_aspect("equal")

fig.tight_layout()
fig.savefig(OUT_DIR / "drift_recovery.png", dpi=120)
print(f"  saved drift_recovery.png")


# -----------------------------------------------------------------------------
# 7. EM trajectory plots
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist_efgp.lengthscale, "o-", label="EFGP")
ax[0].set_xlabel("EM iter"); ax[0].set_ylabel("lengthscale ℓ")
ax[0].set_title("Kernel hyperparameter learning")
ax[0].legend()
ax[1].plot(hist_efgp.latent_rmse, "o-", label="latent RMSE (raw, basis-uncorrected)")
ax[1].set_xlabel("EM iter"); ax[1].set_ylabel("RMSE")
ax[1].set_title("Latent recovery (raw — note PCA-basis ambiguity)")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", dpi=120)
print(f"  saved training_curves.png")


# -----------------------------------------------------------------------------
# 8. Sanity: EFGP-SING is at least competitive with SparseGP on observation RMSE
# -----------------------------------------------------------------------------
print("\n=== PASS criteria ===")
trivial = float(np.sqrt(np.mean((ys - ys.mean(axis=0))**2)))
ratio = obs_rmse_efgp / max(obs_rmse_sp, 1e-9)
ok_efgp = obs_rmse_efgp < trivial
ok_sp = obs_rmse_sp < trivial
ok_match = 0.5 < ratio < 2.0
print(f"  EFGP beats trivial:       {ok_efgp}  ({obs_rmse_efgp:.3f} < {trivial:.3f})")
print(f"  SparseGP beats trivial:   {ok_sp}  ({obs_rmse_sp:.3f} < {trivial:.3f})")
print(f"  EFGP/SparseGP ratio:      {ratio:.2f}  (within [0.5, 2.0])")

if ok_efgp and ok_sp and ok_match:
    print("\n  ALL CHECKS PASSED")
else:
    print("\n  WARNING: some checks failed.  Inspect the plots in", OUT_DIR)
