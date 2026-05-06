"""2-well drift K-sweep: EFGP-gmix vs SparseGP {M=25, M=49} at T=1000.

Drift: f(x, y) = (x - x³, -y)   — bistable in x, restoring in y.
Same 2-well family as ``bench_2well_mc_vs_gmix.py`` (NOT a GP draw,
so there's no "true ℓ"; both methods chase their own MLE).

For each K ∈ {1, 10, 30}:
  * Fit all three methods.
  * Per-trial drift RMSE along the TRUE latent paths.
  * Wall clock.
  * ℓ and σ² trajectory per EM iter.
  * Save a figure ``bench_2well_K{K}.png``.

Emissions FIXED at truth so latents stay in truth coords (no
Procrustes drift between methods).

Usage:  python demos/bench_2well_K_sweep.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import numpy as np
import jax
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

D = 2
SIGMA = 0.4
LS_INIT = 0.3
VAR_INIT = 1.0
T = 1000
T_MAX = 50.0          # roughly the per-trial integration window
N_OBS = 8

K_LIST = [1, 10, 30]
N_EM = 50
N_ESTEP = 10
N_MSTEP_INNER = 4
MSTEP_LR = 0.01

OUT_DIR = Path(__file__).resolve().parent / "_bench_2well_K_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Drift / data
# ---------------------------------------------------------------------------
def two_well_jax(x, t):
    return jnp.array([x[0] - x[0] ** 3, -x[1]])


def two_well_np(X):
    X = np.asarray(X)
    return np.column_stack([X[:, 0] - X[:, 0] ** 3, -X[:, 1]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def simulate_K_trials(K_max, seed=42):
    """Generate K_max independent 2-well trajectories with diverse x0."""
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    rng = np.random.default_rng(seed)
    x0_K = jnp.asarray(rng.normal(0.0, 1.0, size=(K_max, D)).astype(np.float32))

    xs_K = []
    for k in range(K_max):
        xs_k = simulate_sde(jr.PRNGKey(7000 + k), x0=x0_K[k],
                             f=two_well_jax, t_max=T_MAX,
                             n_timesteps=T, sigma=sigma_fn)
        xs_K.append(jnp.clip(xs_k, -3.0, 3.0))
    xs_K = jnp.stack(xs_K, axis=0)                      # (K_max, T, D)

    C_true = rng.standard_normal((N_OBS, D)).astype(np.float32) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys_K = jnp.stack([
        simulate_gaussian_obs(jr.PRNGKey(8000 + k), xs_K[k], out_true)
        for k in range(K_max)
    ], axis=0)                                           # (K_max, T, N_OBS)
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


def eval_efgp_drift_along(mp, hist, t_grid, X_eval, trial_mask):
    """q(f) at final hypers, eval at X_eval (M, D)."""
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
# SparseGP
# ---------------------------------------------------------------------------
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
    ip = dict(mu0=x0_K, V0=jnp.tile(0.3 * jnp.eye(D), (K_, 1, 1)))

    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = _data_aware_zs(num_per_dim=num_per_dim, xs_np=np.asarray(xs_K))
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                                output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho = jnp.linspace(0.05, 0.7, N_EM)
    lr = jnp.full((N_EM,), MSTEP_LR)
    history = []
    t0 = time.time()
    mp, _, _, dp, _, _, _, elbos = fit_variational_em(
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


def eval_sparsegp_drift_along(sparse_drift, dp, mp, t_grid, trial_mask,
                                X_eval):
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
    print(f"\n=== 2-well K-sweep (T={T}, K∈{K_LIST}, n_em={N_EM}) ===")

    # Generate the full K=max trajectories once; subset for smaller K.
    K_max = max(K_LIST)
    xs_full, ys_full, x0_full, out_true = simulate_K_trials(K_max)
    t_grid = jnp.linspace(0., T_MAX, T)

    # Fixed evaluation set: drift accuracy along TRUE latent paths.
    # Use trial_mask of all-ones with the per-K subset.

    summary_rows = []
    for K_ in K_LIST:
        print(f"\n  ----- K = {K_} -----")
        xs_K = xs_full[:K_]
        ys_K = ys_full[:K_]
        x0_K = x0_full[:K_]
        trial_mask = jnp.ones((K_, T), dtype=bool)

        # Drift evaluation: along each trial's true path
        xs_flat = np.asarray(xs_K).reshape(-1, D)
        f_true_flat = two_well_np(xs_flat)
        drift_scale = float(np.sqrt(np.mean(f_true_flat ** 2)))

        # Drift evaluation grid for the field plots: span the data bbox
        xs_np = np.asarray(xs_K)
        lo = xs_np.min(axis=(0, 1)) - 0.3
        hi = xs_np.max(axis=(0, 1)) + 0.3
        n_quiver = 16
        gx = np.linspace(lo[0], hi[0], n_quiver)
        gy = np.linspace(lo[1], hi[1], n_quiver)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
        f_true_grid = two_well_np(grid_pts).reshape(n_quiver, n_quiver, D)

        print(f"    fitting EFGP-gmix...")
        mp_e, hist_e, wall_e = fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid)
        f_e = eval_efgp_drift_along(mp_e, hist_e, t_grid, xs_flat, trial_mask)
        rmse_e_per = np.sqrt(np.mean(
            (f_e.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
            axis=(1, 2)))
        rel_e = float(np.sqrt(np.mean((f_e - f_true_flat) ** 2)) / drift_scale)
        ls_hist_e = list(hist_e.lengthscale)
        var_hist_e = list(hist_e.variance)
        # Drift field on the eval grid
        f_e_grid = eval_efgp_drift_along(mp_e, hist_e, t_grid, grid_pts,
                                            trial_mask).reshape(n_quiver, n_quiver, D)
        print(f"      wall {wall_e:.1f}s  ℓ_final={ls_hist_e[-1]:.3f}  "
              f"σ²_final={var_hist_e[-1]:.3f}  rel_drift_rmse={rel_e:.3f}")

        print(f"    fitting SparseGP M=25...")
        mp_25, dp_25, sd_25, ls_h_25, var_h_25, wall_25 = fit_sparsegp(
            ys_K, xs_K, x0_K, out_true, t_grid, num_per_dim=5)
        f_25 = eval_sparsegp_drift_along(sd_25, dp_25, mp_25, t_grid,
                                            trial_mask, xs_flat)
        rmse_25_per = np.sqrt(np.mean(
            (f_25.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
            axis=(1, 2)))
        rel_25 = float(np.sqrt(np.mean((f_25 - f_true_flat) ** 2))
                        / drift_scale)
        f_25_grid = eval_sparsegp_drift_along(
            sd_25, dp_25, mp_25, t_grid, trial_mask, grid_pts
        ).reshape(n_quiver, n_quiver, D)
        print(f"      wall {wall_25:.1f}s  ℓ_final={ls_h_25[-1]:.3f}  "
              f"σ²_final={var_h_25[-1]:.3f}  rel_drift_rmse={rel_25:.3f}")

        print(f"    fitting SparseGP M=49...")
        mp_49, dp_49, sd_49, ls_h_49, var_h_49, wall_49 = fit_sparsegp(
            ys_K, xs_K, x0_K, out_true, t_grid, num_per_dim=7)
        f_49 = eval_sparsegp_drift_along(sd_49, dp_49, mp_49, t_grid,
                                            trial_mask, xs_flat)
        rmse_49_per = np.sqrt(np.mean(
            (f_49.reshape(K_, T, D) - f_true_flat.reshape(K_, T, D)) ** 2,
            axis=(1, 2)))
        rel_49 = float(np.sqrt(np.mean((f_49 - f_true_flat) ** 2))
                        / drift_scale)
        f_49_grid = eval_sparsegp_drift_along(
            sd_49, dp_49, mp_49, t_grid, trial_mask, grid_pts
        ).reshape(n_quiver, n_quiver, D)
        print(f"      wall {wall_49:.1f}s  ℓ_final={ls_h_49[-1]:.3f}  "
              f"σ²_final={var_h_49[-1]:.3f}  rel_drift_rmse={rel_49:.3f}")

        # ---- Save ----
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
                  drift_scale=drift_scale)
        summary_rows.append((K_, wall_e, wall_25, wall_49,
                              rel_e, rel_25, rel_49,
                              ls_hist_e[-1], ls_h_25[-1], ls_h_49[-1],
                              var_hist_e[-1], var_h_25[-1], var_h_49[-1]))

        # ---- Plot per-K figure ----
        render_K_figure(
            K_, T, drift_scale,
            wall_e, wall_25, wall_49,
            rmse_e_per, rmse_25_per, rmse_49_per,
            ls_hist_e, var_hist_e,
            ls_h_25, var_h_25, ls_h_49, var_h_49,
            GX, GY, f_true_grid, f_e_grid, f_25_grid, f_49_grid,
            xs_np,
        )

    # ---- Summary table ----
    print(f"\n  ===== summary (T={T}) =====")
    header = (f"  {'K':>3}  | {'wall_efgp':>10} {'wall_sp25':>10} "
              f"{'wall_sp49':>10} | {'rel_e':>6} {'rel_25':>6} "
              f"{'rel_49':>6} | {'ℓ_e':>6} {'ℓ_25':>6} {'ℓ_49':>6} "
              f"| {'σ²_e':>6} {'σ²_25':>6} {'σ²_49':>6}")
    print(header)
    for r in summary_rows:
        print(f"  {r[0]:>3}  | {r[1]:>9.1f}s {r[2]:>9.1f}s {r[3]:>9.1f}s "
              f"| {r[4]:>6.3f} {r[5]:>6.3f} {r[6]:>6.3f} "
              f"| {r[7]:>6.3f} {r[8]:>6.3f} {r[9]:>6.3f} "
              f"| {r[10]:>6.3f} {r[11]:>6.3f} {r[12]:>6.3f}")
    print(f"\n  saved: {OUT_DIR}")


def render_K_figure(K_, T, drift_scale,
                     wall_e, wall_25, wall_49,
                     rmse_e_per, rmse_25_per, rmse_49_per,
                     ls_hist_e, var_hist_e,
                     ls_h_25, var_h_25, ls_h_49, var_h_49,
                     GX=None, GY=None, f_true_grid=None,
                     f_e_grid=None, f_25_grid=None, f_49_grid=None,
                     xs_np=None):
    """Per-K figure: 3 rows × 4 cols.
      row 0:  wall (wide)            |  drift RMSE per trial (wide)
      row 1:  ℓ trajectory (wide)    |  σ² trajectory (wide)
      row 2:  GT field | EFGP field | SparseGP M=25 | SparseGP M=49
    """
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                            height_ratios=[1, 1, 1.15])
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0:2])
    axes[0, 1] = fig.add_subplot(gs[0, 2:4])
    axes[1, 0] = fig.add_subplot(gs[1, 0:2])
    axes[1, 1] = fig.add_subplot(gs[1, 2:4])
    field_axes = [fig.add_subplot(gs[2, c]) for c in range(4)]

    # ---- Wall clock bar chart (top-left) ----
    ax = axes[0, 0]
    methods = ['EFGP-gmix', 'SparseGP M=25', 'SparseGP M=49']
    walls = [wall_e, wall_25, wall_49]
    colors = ['C0', 'C1', 'C2']
    bars = ax.bar(methods, walls, color=colors, edgecolor='k')
    for b, w in zip(bars, walls):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f'{w:.1f}s', ha='center', va='bottom')
    ax.set_ylabel('wall clock (s)')
    ax.set_title(f'wall clock  (K={K_}, T={T}, n_em={N_EM})')
    ax.grid(alpha=0.3, axis='y')

    # ---- Drift RMSE along trajectory (top-right) ----
    ax = axes[0, 1]
    width = 0.27
    idx = np.arange(K_)
    ax.bar(idx - width, rmse_e_per, width=width, label='EFGP-gmix',
            color='C0', edgecolor='k')
    ax.bar(idx, rmse_25_per, width=width, label='SparseGP M=25',
            color='C1', edgecolor='k')
    ax.bar(idx + width, rmse_49_per, width=width, label='SparseGP M=49',
            color='C2', edgecolor='k')
    ax.axhline(drift_scale, color='k', linestyle=':', alpha=0.5,
                label=f'drift scale = {drift_scale:.2f}')
    ax.set_xlabel('trial')
    ax.set_xticks(idx)
    if K_ > 12:
        # Too many trials to label every one — just show every 5th
        ax.set_xticks(idx[::max(1, K_ // 6)])
    ax.set_ylabel('drift RMSE along path')
    ax.set_title(f'drift accuracy along true latent path  (per-trial)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # ---- ℓ trajectory (bottom-left) ----
    ax = axes[1, 0]
    iters = np.arange(0, len(ls_hist_e) + 1)
    ax.plot(iters, [LS_INIT] + list(ls_hist_e), '-o', label='EFGP-gmix',
             color='C0', markersize=4)
    ax.plot(iters, [LS_INIT] + list(ls_h_25), '-s', label='SparseGP M=25',
             color='C1', markersize=4)
    ax.plot(iters, [LS_INIT] + list(ls_h_49), '-^', label='SparseGP M=49',
             color='C2', markersize=4)
    ax.set_xlabel('EM iter')
    ax.set_ylabel('ℓ')
    ax.set_title('lengthscale trajectory')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # ---- σ² trajectory (bottom-right) ----
    ax = axes[1, 1]
    ax.plot(iters, [VAR_INIT] + list(var_hist_e), '-o', label='EFGP-gmix',
             color='C0', markersize=4)
    ax.plot(iters, [VAR_INIT] + list(var_h_25), '-s', label='SparseGP M=25',
             color='C1', markersize=4)
    ax.plot(iters, [VAR_INIT] + list(var_h_49), '-^', label='SparseGP M=49',
             color='C2', markersize=4)
    ax.set_xlabel('EM iter')
    ax.set_ylabel('σ²')
    ax.set_yscale('log')
    ax.set_title('output-scale trajectory  (log y)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # ---- Drift fields (bottom row) ----
    if f_true_grid is not None:
        # Render quiver: subsample to ~12x12 even from the 16x16 grid for
        # readability, with consistent arrow scale across all 4 panels.
        step = max(1, GX.shape[0] // 12)
        SX = GX[::step, ::step]; SY = GY[::step, ::step]
        Tg = f_true_grid[::step, ::step]
        Eg = f_e_grid[::step, ::step]
        S1 = f_25_grid[::step, ::step]
        S2 = f_49_grid[::step, ::step]
        # Shared scale so arrows are comparable
        all_norms = np.concatenate([
            np.linalg.norm(g.reshape(-1, 2), axis=1)
            for g in (Tg, Eg, S1, S2)
        ])
        scale = float(np.percentile(all_norms, 95)) * SX.shape[0] * 0.9
        scale = max(scale, 1e-6)
        titles = ['ground truth  f(x,y)=(x − x³, −y)',
                   'EFGP-gmix', 'SparseGP M=25', 'SparseGP M=49']
        fields = [Tg, Eg, S1, S2]
        for ax, ttl, F in zip(field_axes, titles, fields):
            ax.quiver(SX, SY, F[..., 0], F[..., 1],
                       color='k', angles='xy', scale_units='xy',
                       scale=scale, width=0.005)
            # Overlay one trial's path (lightly) for spatial context
            if xs_np is not None and xs_np.shape[0] >= 1:
                ax.plot(xs_np[0, :, 0], xs_np[0, :, 1],
                         color='C3', alpha=0.25, linewidth=0.8)
            ax.set_xlim(GX.min(), GX.max())
            ax.set_ylim(GY.min(), GY.max())
            ax.set_aspect('equal')
            ax.set_title(ttl, fontsize=9)
            ax.grid(alpha=0.25)

    fig.suptitle(f'2-well drift  K={K_}  T={T}', fontsize=12)
    out = OUT_DIR / f'bench_2well_K{K_}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"      saved figure: {out}")


if __name__ == "__main__":
    main()
