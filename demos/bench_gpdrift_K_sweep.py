"""GP-drift K-sweep: EFGP-gmix vs SparseGP {M=25, M=49} at T=1000, fp64.

In-model companion to ``bench_2well_K_sweep.py``: drift is a SINGLE fixed
sample from the SE-kernel GP prior at known ``ℓ_true=0.6, σ²_true=1.0``.
Both methods are fitting an RBF GP — no kernel-vs-cubic misspecification,
so any drift-recovery gap reflects approximation quality + smoother bias
ONLY.

For each K ∈ {1, 10, 30}:
  * Fit EFGP-gmix, SparseGP M=25, SparseGP M=49.
  * Per-trial drift RMSE along the TRUE latent paths (= along the GP
    sample evaluated at the true paths).
  * Wall clock.
  * ℓ and σ² trajectory per EM iter.
  * Drift field comparison (GT vs each method) on a 2D quiver.
  * Saves ``bench_gpdrift_K{K}.png`` and ``K{K}.npz``.

Settings match ``bench_2well_K_sweep.py``: n_em=50, lr=0.01, n_iters_m=4,
rho_sched=linspace(0.05, 0.7), eps_grid=1e-3, kernel_warmup_iters=8,
output_scale=sqrt(σ²) for SparseGP.

Usage:  python demos/bench_gpdrift_K_sweep.py
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
# fp64 is REQUIRED — see CLAUDE.md.
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
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em


# ---------- problem constants ----------
D = 2
SIGMA = 0.4                     # SDE noise
LS_TRUE = 0.6
VAR_TRUE = 1.0
LS_INIT = 0.3                   # deliberately wrong init, matches 2-well bench
VAR_INIT = 1.0
T = 1000
T_MAX = 15.0                    # per-trial integration window (long enough for trajectories to explore full bbox)
N_OBS = 6
ALPHA_RESTORE = 0.05            # weak linear restoring (timescale 1/α=20, > T_MAX): keeps tails bounded without dominating

K_LIST = [1, 10, 30]
N_EM = 50
N_ESTEP = 10
N_MSTEP_INNER = 4
MSTEP_LR = 0.01

# SparseGP M=49 at K=30 fp64 is ~3 hrs; M=25 alone gives the comparison.
# Set to True if you want M=49 included.
INCLUDE_SP49 = False

DRIFT_SEED = 42                 # one fixed GP-drift sample for the whole sweep

OUT_DIR = Path(__file__).resolve().parent / "_bench_gpdrift_K_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


# ---------------------------------------------------------------------------
# GP drift sampler — random-features draw from the SE-kernel GP prior.
# Same machinery as `_gp_drift_factory` in tests/test_efgp_jax_recovery.py.
# Returns ``drift_x(x)`` for SDE simulation and ``drift_grid(X)`` for
# vectorised evaluation on a grid of test points.
# ---------------------------------------------------------------------------
def gp_drift_factory(ls, var, key, extent=4.0, eps_grid=1e-2):
    X_template = jnp.linspace(-extent, extent, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
        / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps                  # (D, M)

    def drift_grid(X_eval):
        """X_eval: (N, D) -> (N, D) drift values."""
        return jnp.stack([
            jp.nufft2(X_eval, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real
            for r in range(D)
        ], axis=-1)

    def drift_x(x, t):
        # SDE-friendly signature: f(x, t) -> (D,).  Add gentle linear
        # restoring -ALPHA_RESTORE * x so trajectories don't blow up.
        return drift_grid(x[None, :])[0] - ALPHA_RESTORE * x

    return drift_x, drift_grid


# ---------------------------------------------------------------------------
# Data simulation (one-time across the whole sweep)
# ---------------------------------------------------------------------------
def simulate_K_trials(K_max, seed=DRIFT_SEED):
    """Sample one GP drift, then K_max independent trials with diverse x0."""
    drift_fn, drift_grid_fn = gp_drift_factory(LS_TRUE, VAR_TRUE,
                                                 jr.PRNGKey(seed))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)

    rng = np.random.default_rng(seed + 1)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, size=(K_max, D)).astype(np.float64))

    xs_K = []
    for k in range(K_max):
        xs_k = simulate_sde(jr.PRNGKey(seed + 100 + k), x0=x0_K[k],
                             f=drift_fn, t_max=T_MAX,
                             n_timesteps=T, sigma=sigma_fn)
        xs_K.append(jnp.clip(xs_k, -3.0, 3.0))
    xs_K = jnp.stack(xs_K, axis=0)                   # (K_max, T, D)

    rng2 = np.random.default_rng(seed + 2)
    C_true = rng2.standard_normal((N_OBS, D)).astype(np.float64) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys_K = jnp.stack([
        simulate_gaussian_obs(jr.PRNGKey(seed + 200 + k), xs_K[k], out_true)
        for k in range(K_max)
    ], axis=0)
    return drift_grid_fn, xs_K, ys_K, x0_K, out_true


# ---------------------------------------------------------------------------
# EFGP / SparseGP fits — same patterns as bench_2well_K_sweep
# ---------------------------------------------------------------------------
def fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K_, 1, 1)))
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
        pin_grid=True, pin_grid_lengthscale=LS_TRUE * 0.75,
        verbose=False,
        true_xs=np.asarray(xs_K),
    )
    return mp, hist, time.time() - t0


def eval_efgp_drift_along(mp, hist, t_grid, X_eval, trial_mask):
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    ms = jnp.asarray(np.asarray(mp['m']))
    Ss = jnp.asarray(np.asarray(mp['S']))
    SSs = jnp.asarray(np.asarray(mp['SS']))
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(LS_TRUE * 0.75, var, X_template, eps=1e-3)
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
        mu_r, grid, jnp.asarray(X_eval, dtype=jnp.float64)[None],
        D_lat=D, D_out=D)
    return np.asarray(Ef[0])


def _data_aware_zs(num_per_dim, xs_np, pad=0.4):
    lo = xs_np.min(axis=(0, 1)) - pad
    hi = xs_np.max(axis=(0, 1)) + pad
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim)
                for d in range(xs_np.shape[-1])]
    return jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'),
                      axis=-1).reshape(-1, xs_np.shape[-1])


def fit_sparsegp(ys_K, xs_K, x0_K, out_true, t_grid, num_per_dim):
    K_, T_, _ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K_, 1, 1)))
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = _data_aware_zs(num_per_dim=num_per_dim, xs_np=np.asarray(xs_K))
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                                output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho = jnp.linspace(0.05, 0.7, N_EM)
    lr = jnp.full((N_EM,), MSTEP_LR)
    history = []
    t0 = time.time()
    mp, _, _, dp, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=dict(out_true),
        sigma=SIGMA, rho_sched=rho,
        n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=N_MSTEP_INNER,
        perform_m_step=True, learn_output_params=False,
        learning_rate=lr, print_interval=999,
        drift_params_history=history)
    wall = time.time() - t0
    ls_hist = [float(jnp.exp(jnp.mean(jnp.log(h['length_scales']))))
               for h in history]
    var_hist = [float(h['output_scale']) ** 2 for h in history]
    return mp, dp, sparse_drift, ls_hist, var_hist, wall


def eval_sparsegp_drift_along(sparse_drift, dp, mp, t_grid, trial_mask, X_eval):
    gp_post = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp, trial_mask, dp,
        jnp.zeros((trial_mask.shape[0], trial_mask.shape[1], 1)),
        jnp.zeros((D, 1)), SIGMA)
    return np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post, dp, jnp.asarray(X_eval)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"\n=== GP-drift K-sweep (T={T}, K∈{K_LIST}, n_em={N_EM}, fp64) ===")
    print(f"  GP truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}; "
          f"init ℓ={LS_INIT}, σ²={VAR_INIT}; SDE σ={SIGMA}")
    K_max = max(K_LIST)
    drift_grid_fn, xs_full, ys_full, x0_full, out_true = simulate_K_trials(K_max)
    t_grid = jnp.linspace(0., T_MAX, T)

    summary_rows = []
    for K_ in K_LIST:
        print(f"\n  ----- K = {K_} -----")
        xs_K = xs_full[:K_]
        ys_K = ys_full[:K_]
        x0_K = x0_full[:K_]
        trial_mask = jnp.ones((K_, T), dtype=bool)

        xs_flat = np.asarray(xs_K).reshape(-1, D)
        # The SDE drift used for simulation is `f_GP(x) - ALPHA_RESTORE * x`.
        # The inference methods are fitting the FULL drift (a stationary RBF
        # GP can't represent the non-stationary linear restoring, so it gets
        # absorbed into the inferred f).  We must compare against the full
        # truth, not just the GP component.
        f_true_flat = np.asarray(
            drift_grid_fn(jnp.asarray(xs_flat, dtype=jnp.float64))
        ) - ALPHA_RESTORE * xs_flat
        drift_scale = float(np.sqrt(np.mean(f_true_flat ** 2)))

        # Drift evaluation grid for field plots
        xs_np = np.asarray(xs_K)
        lo = xs_np.min(axis=(0, 1)) - 0.3
        hi = xs_np.max(axis=(0, 1)) + 0.3
        n_quiver = 16
        gx = np.linspace(lo[0], hi[0], n_quiver)
        gy = np.linspace(lo[1], hi[1], n_quiver)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
        # Same correction for the field plot: include the linear restoring.
        f_true_grid_gp = np.asarray(drift_grid_fn(
            jnp.asarray(grid_pts, dtype=jnp.float64)))
        f_true_grid = (f_true_grid_gp - ALPHA_RESTORE * grid_pts
                       ).reshape(n_quiver, n_quiver, D)

        # ---- EFGP ----
        print(f"    fitting EFGP-gmix...")
        mp_e, hist_e, wall_e = fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid)
        f_e = eval_efgp_drift_along(mp_e, hist_e, t_grid, xs_flat, trial_mask)
        rmse_e_per = np.sqrt(np.mean(
            (f_e.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
            axis=(1, 2)))
        rel_e = float(np.sqrt(np.mean((f_e - f_true_flat) ** 2)) / drift_scale)
        ls_hist_e = list(hist_e.lengthscale)
        var_hist_e = list(hist_e.variance)
        f_e_grid = eval_efgp_drift_along(mp_e, hist_e, t_grid, grid_pts,
                                            trial_mask).reshape(n_quiver, n_quiver, D)
        print(f"      wall {wall_e:.0f}s  ℓ_final={ls_hist_e[-1]:.3f}  "
              f"σ²_final={var_hist_e[-1]:.3f}  rel_drift={rel_e:.3f}")

        # ---- SparseGP M=25 ----
        print(f"    fitting SparseGP M=25...")
        mp_25, dp_25, sd_25, ls_h_25, var_h_25, wall_25 = fit_sparsegp(
            ys_K, xs_K, x0_K, out_true, t_grid, num_per_dim=5)
        f_25 = eval_sparsegp_drift_along(sd_25, dp_25, mp_25, t_grid,
                                            trial_mask, xs_flat)
        rmse_25_per = np.sqrt(np.mean(
            (f_25.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
            axis=(1, 2)))
        rel_25 = float(np.sqrt(np.mean((f_25 - f_true_flat) ** 2)) / drift_scale)
        f_25_grid = eval_sparsegp_drift_along(
            sd_25, dp_25, mp_25, t_grid, trial_mask, grid_pts
        ).reshape(n_quiver, n_quiver, D)
        print(f"      wall {wall_25:.0f}s  ℓ_final={ls_h_25[-1]:.3f}  "
              f"σ²_final={var_h_25[-1]:.3f}  rel_drift={rel_25:.3f}")

        # ---- SparseGP M=49 (optional; expensive at K=30 fp64) ----
        if INCLUDE_SP49:
            print(f"    fitting SparseGP M=49...")
            mp_49, dp_49, sd_49, ls_h_49, var_h_49, wall_49 = fit_sparsegp(
                ys_K, xs_K, x0_K, out_true, t_grid, num_per_dim=7)
            f_49 = eval_sparsegp_drift_along(sd_49, dp_49, mp_49, t_grid,
                                                trial_mask, xs_flat)
            rmse_49_per = np.sqrt(np.mean(
                (f_49.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
                axis=(1, 2)))
            rel_49 = float(np.sqrt(np.mean((f_49 - f_true_flat) ** 2)) / drift_scale)
            f_49_grid = eval_sparsegp_drift_along(
                sd_49, dp_49, mp_49, t_grid, trial_mask, grid_pts
            ).reshape(n_quiver, n_quiver, D)
            print(f"      wall {wall_49:.0f}s  ℓ_final={ls_h_49[-1]:.3f}  "
                  f"σ²_final={var_h_49[-1]:.3f}  rel_drift={rel_49:.3f}")
        else:
            wall_49 = 0.0
            ls_h_49 = [float('nan')] * N_EM
            var_h_49 = [float('nan')] * N_EM
            rmse_49_per = np.full(K_, np.nan)
            rel_49 = float('nan')
            f_49_grid = np.full((n_quiver, n_quiver, D), np.nan)
            print(f"    [skipped SparseGP M=49 — INCLUDE_SP49=False]")

        # Save (also include trajectory paths and field grids so we can
        # re-render figures without rerunning the fits).
        np.savez(OUT_DIR / f"K{K_}.npz",
                  K=K_, T=T,
                  wall_efgp=wall_e, wall_sp25=wall_25, wall_sp49=wall_49,
                  ls_hist_efgp=ls_hist_e, var_hist_efgp=var_hist_e,
                  ls_hist_sp25=ls_h_25, var_hist_sp25=var_h_25,
                  ls_hist_sp49=ls_h_49, var_hist_sp49=var_h_49,
                  rel_drift_rmse_efgp=rel_e,
                  rel_drift_rmse_sp25=rel_25,
                  rel_drift_rmse_sp49=rel_49,
                  per_trial_drift_rmse_efgp=rmse_e_per,
                  per_trial_drift_rmse_sp25=rmse_25_per,
                  per_trial_drift_rmse_sp49=rmse_49_per,
                  drift_scale=drift_scale,
                  ls_true=LS_TRUE, var_true=VAR_TRUE,
                  xs_K=xs_np,
                  GX=GX, GY=GY,
                  f_true_grid=f_true_grid,
                  f_e_grid=f_e_grid, f_25_grid=f_25_grid, f_49_grid=f_49_grid)
        summary_rows.append((K_, wall_e, wall_25, wall_49,
                              rel_e, rel_25, rel_49,
                              ls_hist_e[-1], ls_h_25[-1], ls_h_49[-1],
                              var_hist_e[-1], var_h_25[-1], var_h_49[-1]))
        render_K_figure(
            K_, T, drift_scale,
            wall_e, wall_25, wall_49,
            rmse_e_per, rmse_25_per, rmse_49_per,
            ls_hist_e, var_hist_e,
            ls_h_25, var_h_25, ls_h_49, var_h_49,
            GX, GY, f_true_grid, f_e_grid, f_25_grid, f_49_grid,
            xs_np)

    print(f"\n  ===== summary (T={T}, ℓ_true={LS_TRUE}, σ²_true={VAR_TRUE}) =====")
    print(f"  {'K':>3}  | {'wall_efgp':>10} {'wall_sp25':>10} {'wall_sp49':>10} "
          f"| {'rel_e':>6} {'rel_25':>6} {'rel_49':>6} "
          f"| {'ℓ_e':>6} {'ℓ_25':>6} {'ℓ_49':>6} "
          f"| {'σ²_e':>6} {'σ²_25':>6} {'σ²_49':>6}")
    for r in summary_rows:
        print(f"  {r[0]:>3}  | {r[1]:>9.0f}s {r[2]:>9.0f}s {r[3]:>9.0f}s "
              f"| {r[4]:>6.3f} {r[5]:>6.3f} {r[6]:>6.3f} "
              f"| {r[7]:>6.3f} {r[8]:>6.3f} {r[9]:>6.3f} "
              f"| {r[10]:>6.3f} {r[11]:>6.3f} {r[12]:>6.3f}")
    print(f"\n  saved: {OUT_DIR}")


def render_K_figure(K_, T, drift_scale,
                     wall_e, wall_25, wall_49,
                     rmse_e_per, rmse_25_per, rmse_49_per,
                     ls_hist_e, var_hist_e,
                     ls_h_25, var_h_25, ls_h_49, var_h_49,
                     GX, GY, f_true_grid, f_e_grid, f_25_grid, f_49_grid,
                     xs_np):
    import matplotlib.gridspec as gridspec
    has_49 = not (np.isnan(rmse_49_per).all()
                  or np.isnan(np.asarray(ls_h_49)).all())
    n_field_cols = 4 if has_49 else 3
    fig = plt.figure(figsize=(15 if has_49 else 12, 13))
    gs = gridspec.GridSpec(3, n_field_cols, figure=fig, hspace=0.35,
                            wspace=0.3, height_ratios=[1, 1, 1.15])
    ax00 = fig.add_subplot(gs[0, :n_field_cols // 2 + (n_field_cols % 2)])
    ax01 = fig.add_subplot(gs[0, n_field_cols // 2 + (n_field_cols % 2):])
    ax10 = fig.add_subplot(gs[1, :n_field_cols // 2 + (n_field_cols % 2)])
    ax11 = fig.add_subplot(gs[1, n_field_cols // 2 + (n_field_cols % 2):])
    field_axes = [fig.add_subplot(gs[2, c]) for c in range(n_field_cols)]

    # wall clock
    ax = ax00
    if has_49:
        methods = ['EFGP-gmix', 'SparseGP M=25', 'SparseGP M=49']
        walls = [wall_e, wall_25, wall_49]
        cols = ['C0', 'C1', 'C2']
    else:
        methods = ['EFGP-gmix', 'SparseGP M=25']
        walls = [wall_e, wall_25]
        cols = ['C0', 'C1']
    bars = ax.bar(methods, walls, color=cols, edgecolor='k')
    for b, w in zip(bars, walls):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f'{w:.0f}s', ha='center', va='bottom')
    ax.set_ylabel('wall clock (s)')
    ax.set_title(f'wall clock  (K={K_}, T={T}, fp64, n_em={N_EM})')
    ax.grid(alpha=0.3, axis='y')

    # drift RMSE per trial
    ax = ax01
    idx = np.arange(K_)
    if has_49:
        width = 0.27
        ax.bar(idx - width, rmse_e_per, width=width, label='EFGP-gmix',
                color='C0', edgecolor='k')
        ax.bar(idx, rmse_25_per, width=width, label='SparseGP M=25',
                color='C1', edgecolor='k')
        ax.bar(idx + width, rmse_49_per, width=width, label='SparseGP M=49',
                color='C2', edgecolor='k')
    else:
        width = 0.4
        ax.bar(idx - width / 2, rmse_e_per, width=width, label='EFGP-gmix',
                color='C0', edgecolor='k')
        ax.bar(idx + width / 2, rmse_25_per, width=width, label='SparseGP M=25',
                color='C1', edgecolor='k')
    ax.axhline(drift_scale, color='k', linestyle=':', alpha=0.5,
                label=f'drift scale = {drift_scale:.2f}')
    ax.set_xlabel('trial')
    if K_ > 12:
        ax.set_xticks(idx[::max(1, K_ // 6)])
    else:
        ax.set_xticks(idx)
    ax.set_ylabel('drift RMSE along path')
    ax.set_title(f'drift accuracy along true latent path  (per-trial)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # ℓ trajectory
    ax = ax10
    iters = np.arange(0, len(ls_hist_e) + 1)
    ax.plot(iters, [LS_INIT] + list(ls_hist_e), '-o',
             label='EFGP-gmix', color='C0', markersize=4)
    ax.plot(iters, [LS_INIT] + list(ls_h_25), '-s',
             label='SparseGP M=25', color='C1', markersize=4)
    if has_49:
        ax.plot(iters, [LS_INIT] + list(ls_h_49), '-^',
                 label='SparseGP M=49', color='C2', markersize=4)
    ax.axhline(LS_TRUE, color='k', linestyle='--', alpha=0.5,
                label=f'truth ℓ={LS_TRUE}')
    ax.set_xlabel('EM iter')
    ax.set_ylabel('ℓ')
    ax.set_title('lengthscale trajectory')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # σ² trajectory
    ax = ax11
    ax.plot(iters, [VAR_INIT] + list(var_hist_e), '-o',
             label='EFGP-gmix', color='C0', markersize=4)
    ax.plot(iters, [VAR_INIT] + list(var_h_25), '-s',
             label='SparseGP M=25', color='C1', markersize=4)
    if has_49:
        ax.plot(iters, [VAR_INIT] + list(var_h_49), '-^',
                 label='SparseGP M=49', color='C2', markersize=4)
    ax.axhline(VAR_TRUE, color='k', linestyle='--', alpha=0.5,
                label=f'truth σ²={VAR_TRUE}')
    ax.set_xlabel('EM iter')
    ax.set_ylabel('σ²')
    ax.set_yscale('log')
    ax.set_title('output-scale trajectory  (log y)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # drift fields
    step = max(1, GX.shape[0] // 12)
    SX = GX[::step, ::step]; SY = GY[::step, ::step]
    Tg = f_true_grid[::step, ::step]
    Eg = f_e_grid[::step, ::step]
    S1 = f_25_grid[::step, ::step]
    if has_49:
        S2 = f_49_grid[::step, ::step]
        all_grids = (Tg, Eg, S1, S2)
        titles = ['ground truth  (GP draw, ℓ=' + str(LS_TRUE) + ', σ²=' + str(VAR_TRUE) + ')',
                   'EFGP-gmix', 'SparseGP M=25', 'SparseGP M=49']
        plot_grids = [Tg, Eg, S1, S2]
    else:
        all_grids = (Tg, Eg, S1)
        titles = ['ground truth  (GP draw, ℓ=' + str(LS_TRUE) + ', σ²=' + str(VAR_TRUE) + ')',
                   'EFGP-gmix', 'SparseGP M=25']
        plot_grids = [Tg, Eg, S1]
    all_norms = np.concatenate([
        np.linalg.norm(g.reshape(-1, 2), axis=1)
        for g in all_grids
    ])
    scale = float(np.percentile(all_norms, 95)) * SX.shape[0] * 0.9
    scale = max(scale, 1e-6)
    for ax, ttl, F in zip(field_axes, titles, plot_grids):
        ax.quiver(SX, SY, F[..., 0], F[..., 1],
                   color='k', angles='xy', scale_units='xy',
                   scale=scale, width=0.005)
        if xs_np is not None:
            # Overlay ALL K trajectories so coverage is visible.
            for k in range(xs_np.shape[0]):
                ax.plot(xs_np[k, :, 0], xs_np[k, :, 1],
                         color='C3', alpha=min(0.4, 4.0 / xs_np.shape[0]),
                         linewidth=0.5)
        ax.set_xlim(GX.min(), GX.max())
        ax.set_ylim(GY.min(), GY.max())
        ax.set_aspect('equal')
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(f'GP-drift  K={K_}  T={T}  (fp64, ℓ_true={LS_TRUE}, σ²_true={VAR_TRUE})',
                  fontsize=12)
    out = OUT_DIR / f'bench_gpdrift_K{K_}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"      saved figure: {out}")


if __name__ == "__main__":
    main()
