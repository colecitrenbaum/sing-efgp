"""
EFGP-SING (JAX path) vs. SING-SparseGP head-to-head on the d=2 synthetic
linear-drift problem.

Critical: this script must NOT import torch.  We use only the JAX path
(``fit_efgp_sing_jax``) plus SING's stock SparseGP (pure JAX/TFP).
``pytorch_finufft`` and ``jax_finufft`` both wrap libfinufft and segfault
if both are loaded in the same process; the torch demo is in
``efgp_synthetic_2d.py`` and must run in its own process.

To run:

    python demos/efgp_synthetic_2d_jax.py

Outputs go to ``demos/_efgp_demo_out_jax/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import math
import numpy as np

# IMPORTANT: load the JAX path first so jax_finufft loads before any
# possible (transitive) pytorch_finufft import.  efgp_em.py forces
# jax_finufft to load via efgp_jax_primitives.
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs

OUT_DIR = Path(__file__).resolve().parent / "_efgp_demo_out_jax"
OUT_DIR.mkdir(exist_ok=True)
print(f"writing diagnostics to {OUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Synthetic data: 2D linear-drift latent SDE
# -----------------------------------------------------------------------------
D = 2
T = 60
sigma = 0.4
A_true = jnp.array([[-1.5, 0.0], [0.0, -1.5]])


def drift_true_jax(x, t):
    return A_true @ x


def drift_true_np(x):
    return x @ np.asarray(A_true).T   # x: (N, D)


def sigma_fn(x, t):
    return sigma * jnp.eye(D)


print("Simulating SDE...")
xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.2, -0.8]),
                  f=drift_true_jax, t_max=2.0, n_timesteps=T,
                  sigma=sigma_fn)
print(f"  xs shape={xs.shape}, range=({float(xs.min()):.2f}, "
      f"{float(xs.max()):.2f})")

N = 8
rng_np = np.random.default_rng(0)
C_true_np = rng_np.standard_normal((N, D)) * 0.5
out_true = dict(C=jnp.asarray(C_true_np), d=jnp.zeros(N),
                R=jnp.full((N,), 0.05))
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
output_params_init = dict(C=C_init, d=d_init, R=jnp.full((N,), 0.1))
init_params = jax.tree_util.tree_map(
    lambda x: x[None],
    dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1)
)
t_grid = jnp.linspace(0., 2.0, T)


# -----------------------------------------------------------------------------
# 3. Fit EFGP-SING (JAX path)
# -----------------------------------------------------------------------------
print("\n=== Fitting EFGP-SING (JAX) ===")
t0 = time.time()
mp_efgp, _, op_efgp, _, hist_efgp = em.fit_efgp_sing_jax(
    likelihood=lik, t_grid=t_grid,
    output_params=output_params_init, init_params=init_params,
    latent_dim=D, lengthscale=0.7, variance=1.0,
    sigma=sigma, sigma_drift_sq=sigma ** 2,
    eps_grid=1e-2, S_marginal=2,
    n_em_iters=8, n_estep_iters=10,
    rho_sched=jnp.linspace(0.2, 0.9, 8),
    learn_emissions=True, update_R=False,
    learn_kernel=False,                          # see README for details
    true_xs=np.asarray(xs), verbose=True,
)
t_efgp = time.time() - t0
print(f"  EFGP-SING (JAX) wall time: {t_efgp:.1f}s")


# -----------------------------------------------------------------------------
# 4. Fit SING-SparseGP baseline (matched grid)
# -----------------------------------------------------------------------------
print("\n=== Fitting SING-SparseGP baseline ===")
quad = GaussHermiteQuadrature(D=D, n_quad=5)
zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)         # 64 inducing pts
sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
sparse_drift_params = dict(
    length_scales=jnp.full((D,), 0.7),
    output_scale=jnp.asarray(1.0),
)
t0 = time.time()
mp_sp, _, _, _, _, op_sp, _, _ = fit_variational_em(
    key=jr.PRNGKey(33),
    fn=sparse_drift, likelihood=lik, t_grid=t_grid,
    drift_params=sparse_drift_params,
    init_params=init_params, output_params=output_params_init,
    sigma=sigma,
    rho_sched=jnp.linspace(0.2, 0.9, 8),
    n_iters=8, n_iters_e=10, n_iters_m=4,
    perform_m_step=True, learn_output_params=True,
    learning_rate=jnp.full((8,), 0.05),
    print_interval=1,
)
t_sp = time.time() - t0
print(f"  SparseGP wall time: {t_sp:.1f}s, n_inducing=64")


# -----------------------------------------------------------------------------
# 5. Diagnostics: predictive observation RMSE
# -----------------------------------------------------------------------------
def predict_obs_rmse(mp, output_params, ys):
    m = np.asarray(mp['m'][0])
    C = np.asarray(output_params['C'])
    d = np.asarray(output_params['d'])
    y_hat = m @ C.T + d
    return float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))


obs_rmse_efgp = predict_obs_rmse(mp_efgp, op_efgp, ys)
obs_rmse_sp = predict_obs_rmse(mp_sp, op_sp, ys)
trivial = float(np.sqrt(np.mean((ys - ys.mean(axis=0)) ** 2)))
print("\n=== Predictive observation RMSE ===")
print(f"  EFGP-SING (JAX) : {obs_rmse_efgp:.4f}   wall={t_efgp:.1f}s")
print(f"  SING-SparseGP   : {obs_rmse_sp:.4f}   wall={t_sp:.1f}s")
print(f"  trivial baseline: {trivial:.4f}")


# -----------------------------------------------------------------------------
# 6. Drift-recovery diagnostic: posterior mean drift on a grid
# -----------------------------------------------------------------------------
print("\n=== Drift recovery diagnostic ===")
m_efgp_np = np.asarray(mp_efgp['m'][0])
grid_lo = m_efgp_np.min(axis=0) - 0.5
grid_hi = m_efgp_np.max(axis=0) + 0.5
xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
GX, GY = np.meshgrid(xs_g, ys_g, indexing="ij")
grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)


# JAX EFGP drift on the grid: rebuild the cache, then run drift_moments_jax
import sing.efgp_jax_drift as jpd
ls = 0.7; var = 1.0
X_template = (jnp.linspace(-2.5, 2.5, 16)[:, None]
              * jnp.ones((1, D)))
grid_jax = jp.spectral_grid_se(ls, var, X_template, eps=1e-2)

# Re-build q(f) at the *final* posterior marginals so the drift on the
# diagnostic grid reflects what was learned.
ms_eval = jnp.asarray(m_efgp_np)
Ss_eval = jnp.asarray(np.asarray(mp_efgp['S'][0]))
SSs_eval = jnp.asarray(np.asarray(mp_efgp['SS'][0]))
del_t = t_grid[1:] - t_grid[:-1]
mu_r, _, _ = jpd.compute_mu_r_jax(
    ms_eval, Ss_eval, SSs_eval, del_t, grid_jax, jr.PRNGKey(99),
    sigma_drift_sq=sigma ** 2, S_marginal=2, D_lat=D, D_out=D,
)
# Evaluate Φ μ at the grid points
grid_pts_j = jnp.asarray(grid_pts, dtype=jnp.float32)
Ef_grid, _, _ = jpd.drift_moments_jax(mu_r, grid_jax, grid_pts_j,
                                       D_lat=D, D_out=D)
f_efgp = np.asarray(Ef_grid).reshape(GX.shape + (D,))

# True drift on the grid (in TRUTH basis; absolute orientations differ
# from EFGP basis due to PCA basis ambiguity)
f_true = drift_true_np(grid_pts).reshape(GX.shape + (D,))

# SparseGP drift
gp_post_sp = sparse_drift.update_dynamics_params(
    jr.PRNGKey(0), t_grid, mp_sp, jnp.ones((1, T), dtype=bool),
    sparse_drift_params,
    jnp.zeros((1, T, 1)),
    jnp.zeros((D, 1)), sigma)
f_sp = np.asarray(sparse_drift.get_posterior_f_mean(
    gp_post_sp, sparse_drift_params, jnp.asarray(grid_pts)
)).reshape(GX.shape + (D,))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].quiver(GX, GY, f_true[..., 0], f_true[..., 1], color="k")
axes[0].plot(np.asarray(xs)[:, 0], np.asarray(xs)[:, 1], "r-", alpha=0.5,
             label="true latent path")
axes[0].set_title("True drift (truth basis)")
axes[0].legend(); axes[0].set_aspect("equal")

axes[1].quiver(GX, GY, f_efgp[..., 0], f_efgp[..., 1], color="b")
axes[1].plot(m_efgp_np[:, 0], m_efgp_np[:, 1], "g-", alpha=0.7,
             label="EFGP inferred mean")
axes[1].set_title(f"EFGP-SING (JAX) drift  M={grid_jax.M}")
axes[1].legend(); axes[1].set_aspect("equal")

m_sp_np = np.asarray(mp_sp['m'][0])
axes[2].quiver(GX, GY, f_sp[..., 0], f_sp[..., 1], color="g")
axes[2].plot(m_sp_np[:, 0], m_sp_np[:, 1], "g-", alpha=0.7)
axes[2].set_title("SparseGP drift  64 inducing pts")
axes[2].set_aspect("equal")

fig.tight_layout()
fig.savefig(OUT_DIR / "drift_recovery.png", dpi=120)
print(f"  saved drift_recovery.png")


# -----------------------------------------------------------------------------
# 7. PASS criteria
# -----------------------------------------------------------------------------
print("\n=== PASS criteria ===")
ratio = obs_rmse_efgp / max(obs_rmse_sp, 1e-9)
ok_efgp = obs_rmse_efgp < trivial
ok_sp = obs_rmse_sp < trivial
ok_match = 0.5 < ratio < 2.0
print(f"  EFGP beats trivial:        {ok_efgp}  ({obs_rmse_efgp:.3f} < {trivial:.3f})")
print(f"  SparseGP beats trivial:    {ok_sp}   ({obs_rmse_sp:.3f} < {trivial:.3f})")
print(f"  EFGP/SparseGP ratio:       {ratio:.2f}  (within [0.5, 2.0])")
print(f"  EFGP wall-time vs SparseGP: {t_efgp / t_sp:.2f}x")
if ok_efgp and ok_sp and ok_match:
    print("\n  ALL CHECKS PASSED")
else:
    print("\n  WARNING: see plots in", OUT_DIR)
