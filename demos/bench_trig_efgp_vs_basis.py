"""Trigonometric-ripple drift, K=10, T=500: EFGP-gmix vs PolyBasis (deg-3).

True drift (NOT in the polynomial basis):
    f(x, y) = ( sin(π x) − 0.3 x,
                sin(π y) − 0.3 y )

Two stable fixed points per axis at x ≈ ±0.95 (where sin(πx) = 0.3x with
cos(πx) ≈ −1), so the 2D field has FOUR stable equilibria arranged in a
square.  sin(πx) is far from any degree-3 polynomial over x ∈ [-3, 3]
(three half-cycles), so the 16-monomial PolyBasis can't approximate it
without massive aliasing.  EFGP's nonparametric SE-kernel prior, by
contrast, can fit sin smoothly given enough data.

Both methods share the SING natural-grad E-step and the IDENTICAL outer
schedule (EFGP CLAUDE.md defaults):

    n_em_iters=50, n_estep=10, n_mstep=4
    rho_sched=linspace(0.05, 0.7, 50)
    mstep_lr=0.01

Only the drift parametrisation differs.  Emissions held at truth.

Outputs (in _bench_trig_efgp_vs_basis_out/):
    bench.npz, bench_summary.png, bench_latents.png, bench_drift.png

Usage:  /Users/colecitrenbaum/myenv/bin/python demos/bench_trig_efgp_vs_basis.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import BasisSDE
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em


# 16-term polynomial basis up to degree 3 (matches inference_and_learning.ipynb)
def poly_basis_2d(x, t):
    return jnp.array([1., x[0], x[0]**2, x[0]**3, x[1], x[1]**2, x[1]**3,
                       x[0]*x[1], (x[0]**2)*(x[1]**2), (x[0]**2)*x[1],
                       x[0]*(x[1]**2), (x[0]**3)*(x[1]**3),
                       (x[0]**3)*x[1], (x[0]**3)*(x[1]**2),
                       x[0]*(x[1]**3), (x[0]**2)*(x[1]**3)])

N_BASIS = 16


# ---------------------------------------------------------------------------
# Problem config
# ---------------------------------------------------------------------------
D = 2
SIGMA = 0.4
LS_INIT = 0.3        # EFGP kernel lengthscale init
VAR_INIT = 1.0       # EFGP kernel variance init
K = 10
T = 500
T_MAX = 25.0
N_OBS = 8

# Shared SING schedule — applies to BOTH methods identically.
N_EM = 50
N_ESTEP = 10
N_MSTEP_INNER = 4
MSTEP_LR = 0.01

OUT_DIR = Path(__file__).resolve().parent / "_bench_trig_efgp_vs_basis_out"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Drift / data
# ---------------------------------------------------------------------------
def trig_drift_jax(x, t):
    return jnp.array([jnp.sin(jnp.pi * x[0]) - 0.3 * x[0],
                       jnp.sin(jnp.pi * x[1]) - 0.3 * x[1]])


def trig_drift_np(X):
    X = np.asarray(X)
    return np.column_stack([np.sin(np.pi * X[:, 0]) - 0.3 * X[:, 0],
                             np.sin(np.pi * X[:, 1]) - 0.3 * X[:, 1]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def simulate_K_trials(K_, seed=42):
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    rng = np.random.default_rng(seed)
    # Spread initial conditions widely so all four basins get sampled.
    x0_K = jnp.asarray(rng.normal(0.0, 1.5, size=(K_, D)).astype(np.float64))

    xs_K = []
    for k in range(K_):
        xs_k = simulate_sde(jr.PRNGKey(7000 + k), x0=x0_K[k],
                             f=trig_drift_jax, t_max=T_MAX,
                             n_timesteps=T, sigma=sigma_fn)
        xs_K.append(jnp.clip(xs_k, -3.0, 3.0))
    xs_K = jnp.stack(xs_K, axis=0)

    C_true = rng.standard_normal((N_OBS, D)).astype(np.float64) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys_K = jnp.stack([
        simulate_gaussian_obs(jr.PRNGKey(8000 + k), xs_K[k], out_true)
        for k in range(K_)
    ], axis=0)
    return xs_K, ys_K, x0_K, out_true


# ---------------------------------------------------------------------------
# EFGP
# ---------------------------------------------------------------------------
def fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.3 * jnp.eye(D), (K_, 1, 1)))

    rho = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=out_true, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
        estep_method='gmix', S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True,
        n_mstep_iters=N_MSTEP_INNER, mstep_lr=MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        pin_grid=True, pin_grid_lengthscale=0.5,
        verbose=False,
        true_xs=np.asarray(xs_K),
    )
    wall = time.time() - t0
    return mp, hist, wall


def eval_efgp_drift(mp, hist, t_grid, X_eval, trial_mask):
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    ms = jnp.asarray(np.asarray(mp['m']))
    Ss = jnp.asarray(np.asarray(mp['S']))
    SSs = jnp.asarray(np.asarray(mp['SS']))
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(0.5, var, X_template, eps=1e-3)
    ws_final = jpd._ws_real_se(jnp.log(ls), jnp.log(var),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval, dtype=jnp.float32)[None],
        D_lat=D, D_out=D)
    return np.asarray(Ef[0])


# ---------------------------------------------------------------------------
# PolyBasis
# ---------------------------------------------------------------------------
def fit_basis(ys_K, xs_K, x0_K, out_true, t_grid):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.3 * jnp.eye(D), (K_, 1, 1)))

    w_init = 0.1 * jr.normal(jr.PRNGKey(4), (D, N_BASIS))
    drift_params = {'w': w_init}

    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    fn = BasisSDE(basis_set=poly_basis_2d, expectation=quad, latent_dim=D)

    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    lr = MSTEP_LR * jnp.ones(N_EM)
    history = []
    t0 = time.time()
    mp, _, _, dp, _, _, _, elbos = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=fn, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params,
        init_params=ip, output_params=dict(out_true),
        sigma=SIGMA, rho_sched=rho_sched,
        n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=N_MSTEP_INNER,
        perform_m_step=True, learn_output_params=False,
        learning_rate=lr, print_interval=999,
        drift_params_history=history)
    wall = time.time() - t0
    return mp, fn, dp, elbos, wall


def eval_basis_drift(fn, dp, X_eval):
    from jax import vmap as _vmap
    w = dp['w']
    Phi = _vmap(lambda x: fn.basis_set(x, jnp.zeros(())))(jnp.asarray(X_eval))
    return np.asarray(Phi @ w.T)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def latent_mse(m_pred, xs_true):
    err = np.asarray(m_pred) - np.asarray(xs_true)
    return float(np.mean(err ** 2))


def per_trial_latent_mse(m_pred, xs_true):
    err = np.asarray(m_pred) - np.asarray(xs_true)
    return np.mean(err ** 2, axis=(1, 2))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"\n=== Trig-ripple: EFGP vs PolyBasis  (K={K}, T={T}, n_em={N_EM}) ===")
    print(f"    drift: f(x,y) = (sin(pi*x) - 0.3*x,  sin(pi*y) - 0.3*y)")

    xs_K, ys_K, x0_K, out_true = simulate_K_trials(K)
    t_grid = jnp.linspace(0., T_MAX, T)
    trial_mask = jnp.ones((K, T), dtype=bool)

    xs_flat = np.asarray(xs_K).reshape(-1, D)
    f_true_flat = trig_drift_np(xs_flat)
    drift_scale = float(np.sqrt(np.mean(f_true_flat ** 2)))

    xs_np = np.asarray(xs_K)
    lo = xs_np.min(axis=(0, 1)) - 0.3
    hi = xs_np.max(axis=(0, 1)) + 0.3
    n_quiver = 16
    gx = np.linspace(lo[0], hi[0], n_quiver)
    gy = np.linspace(lo[1], hi[1], n_quiver)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true_grid = trig_drift_np(grid_pts).reshape(n_quiver, n_quiver, D)

    # ---- EFGP ----
    print("  fitting EFGP-gmix...")
    mp_e, hist_e, wall_e = fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid)
    lat_mse_e = latent_mse(mp_e['m'], xs_K)
    lat_mse_e_per = per_trial_latent_mse(mp_e['m'], xs_K)
    f_e_along = eval_efgp_drift(mp_e, hist_e, t_grid, xs_flat, trial_mask)
    rel_drift_e = float(np.sqrt(np.mean((f_e_along - f_true_flat) ** 2))
                        / drift_scale)
    f_e_grid = eval_efgp_drift(mp_e, hist_e, t_grid, grid_pts,
                                trial_mask).reshape(n_quiver, n_quiver, D)
    print(f"    wall {wall_e:.1f}s  "
          f"ℓ_final={float(hist_e.lengthscale[-1]):.3f}  "
          f"σ²_final={float(hist_e.variance[-1]):.3f}  "
          f"latent_MSE={lat_mse_e:.4f}  rel_drift_rmse={rel_drift_e:.3f}")

    # ---- PolyBasis ----
    print("  fitting PolyBasis (16 monomials, 32 weights)...")
    mp_b, fn_b, dp_b, elbos_b, wall_b = fit_basis(
        ys_K, xs_K, x0_K, out_true, t_grid)
    lat_mse_b = latent_mse(mp_b['m'], xs_K)
    lat_mse_b_per = per_trial_latent_mse(mp_b['m'], xs_K)
    f_b_along = eval_basis_drift(fn_b, dp_b, xs_flat)
    rel_drift_b = float(np.sqrt(np.mean((f_b_along - f_true_flat) ** 2))
                        / drift_scale)
    f_b_grid = eval_basis_drift(fn_b, dp_b, grid_pts).reshape(
        n_quiver, n_quiver, D)
    print(f"    wall {wall_b:.1f}s  "
          f"latent_MSE={lat_mse_b:.4f}  rel_drift_rmse={rel_drift_b:.3f}")

    # ---- Save ----
    np.savez(OUT_DIR / "bench.npz",
              K=K, T=T,
              wall_efgp=wall_e, wall_basis=wall_b,
              latent_mse_efgp=lat_mse_e, latent_mse_basis=lat_mse_b,
              latent_mse_per_trial_efgp=lat_mse_e_per,
              latent_mse_per_trial_basis=lat_mse_b_per,
              rel_drift_rmse_efgp=rel_drift_e,
              rel_drift_rmse_basis=rel_drift_b,
              ls_hist_efgp=np.asarray(hist_e.lengthscale),
              var_hist_efgp=np.asarray(hist_e.variance),
              elbos_basis=np.asarray(elbos_b),
              xs_true=np.asarray(xs_K),
              m_efgp=np.asarray(mp_e['m']),
              m_basis=np.asarray(mp_b['m']),
              f_true_grid=f_true_grid, f_e_grid=f_e_grid, f_b_grid=f_b_grid,
              w_basis=np.asarray(dp_b['w']),
              quiver_GX=GX, quiver_GY=GY)

    # ---- Summary figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = ['EFGP', 'PolyBasis']
    colors = ['tab:blue', 'tab:green']
    walls = [wall_e, wall_b]
    lat_mses = [lat_mse_e, lat_mse_b]
    rel_drifts = [rel_drift_e, rel_drift_b]

    axes[0].bar(methods, walls, color=colors)
    axes[0].set_ylabel("wall (s)")
    axes[0].set_title(f"wall time  (K={K}, T={T})")
    for i, w in enumerate(walls):
        axes[0].text(i, w, f"{w:.0f}s", ha='center', va='bottom')

    axes[1].bar(methods, lat_mses, color=colors)
    axes[1].set_ylabel("MSE  (||m - x_true||² / KTD)")
    axes[1].set_title("latent recovery MSE")
    for i, v in enumerate(lat_mses):
        axes[1].text(i, v, f"{v:.4f}", ha='center', va='bottom')

    axes[2].bar(methods, rel_drifts, color=colors)
    axes[2].set_ylabel("rel drift RMSE  / ‖f_true‖")
    axes[2].set_title("drift RMSE along true paths")
    for i, v in enumerate(rel_drifts):
        axes[2].text(i, v, f"{v:.3f}", ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bench_summary.png", dpi=120)
    plt.close(fig)

    # ---- Latent overlays ----
    K_plot = min(K, 4)
    fig, axes = plt.subplots(K_plot, D, figsize=(10, 2.2 * K_plot),
                              sharex=True)
    if K_plot == 1:
        axes = axes[None, :]
    for k in range(K_plot):
        for d in range(D):
            ax = axes[k, d]
            ax.plot(t_grid, xs_K[k, :, d], 'k', lw=1.5, label='truth')
            ax.plot(t_grid, mp_e['m'][k, :, d], color=colors[0], lw=1.0,
                     label='EFGP')
            ax.plot(t_grid, mp_b['m'][k, :, d], color=colors[1], lw=1.0,
                     label='PolyBasis')
            if k == 0 and d == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.set_ylabel(f"trial {k}, dim {d}")
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    fig.suptitle(f"Latent recovery  (K={K}, T={T})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bench_latents.png", dpi=120)
    plt.close(fig)

    # ---- Drift quiver ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, F, name in zip(axes,
                            [f_true_grid, f_e_grid, f_b_grid],
                            ['truth', 'EFGP', 'PolyBasis']):
        ax.quiver(GX, GY, F[..., 0], F[..., 1],
                   np.linalg.norm(F, axis=-1),
                   cmap='viridis', scale=30)
        ax.set_title(name)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        for k in range(min(K, 6)):
            ax.plot(xs_K[k, :, 0], xs_K[k, :, 1], 'w-', lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bench_drift.png", dpi=120)
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/bench.npz, bench_summary.png, "
          f"bench_latents.png, bench_drift.png")


if __name__ == "__main__":
    main()
