"""
bench_gpdrift_x64.py

Test of the misspecification hypothesis:
  Sample a single drift function f_true ~ GP(0, RBF(ℓ_true, σ²_true)),
  run an SDE with that drift, then ask EFGP and SparseGP to recover it.
  Now the data IS in the model class — no kernel misspecification.

Prediction: EFGP at θ ≈ MLE should now BEAT SparseGP on drift_pc, since
the MLE is consistent for the data-generating θ_true and SparseGP's
non-MLE hypers should generalize worse, not better.

Setup:
  ℓ_true = 1.0, σ²_true = 1.5  (matching Duffing MLE order of magnitude)
  T = 2000, n_em = 50, lr = 0.01, x64
  ls_init ∈ {0.7, 3.0}, methods = EFGP, SP M=25, SP M=64
  Drift: f_true(x) = K(x, X_grid) · α   where α = K_grid⁻¹ f_grid,
         f_grid ~ N(0, K_grid).  This places f_true in the RKHS of
         RBF(ℓ_true, σ²_true).

Run:
  ~/myenv/bin/python demos/bench_gpdrift_x64.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import sing.efgp_em as efgp_em

import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
T = 2000
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
VAR_INIT = 1.0

LS_TRUE = 1.0
VAR_TRUE = 1.5
SIGMA_SDE = 0.2

LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64]

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21


# Reproducible drift seed
DRIFT_SEED = 42
SDE_SEED = 137
EMISSION_SEED = 99
N_OBS = 8

# Grid for sampling the GP drift.  ~30×30 = 900 points in [-3,3]² gives
# faithful interpolation at the eval scales we care about.
GRID_NPER = 30
GRID_LIM = 3.0


def _rbf_K(X1, X2, ls, var):
    """RBF kernel matrix.  X1: (n1, D), X2: (n2, D)."""
    sq = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return var * jnp.exp(-0.5 * sq / (ls ** 2))


def sample_gp_drift(key, ls_true, var_true, lim=GRID_LIM, n_per=GRID_NPER):
    """Sample f_grid ~ N(0, K(ls_true, var_true)) on a uniform grid in
    [-lim, lim]^D, return (X_grid, alpha) where
        alpha = K(X_grid, X_grid)⁻¹ f_grid.
    The drift evaluator at any x is then
        f_true(x) = K(x, X_grid) @ alpha  ∈ R^D
    which is the conditional mean of a GP given f_grid as observations,
    and lives in the RKHS of RBF(ls_true, var_true)."""
    a = jnp.linspace(-lim, lim, n_per)
    GX, GY = jnp.meshgrid(a, a, indexing='ij')
    X_grid = jnp.stack([GX.ravel(), GY.ravel()], axis=-1)  # (M, D)
    M = X_grid.shape[0]
    K = _rbf_K(X_grid, X_grid, ls_true, var_true) + 1e-6 * jnp.eye(M)
    L = jnp.linalg.cholesky(K)
    # Two iid GP draws — one per output dim
    z = jr.normal(key, (M, D))
    f_grid = L @ z                            # (M, D), each col ~ N(0, K)
    alpha = jnp.linalg.solve(K, f_grid)        # (M, D)
    return X_grid, alpha, f_grid


def make_drift_fn(X_grid, alpha, ls_true, var_true):
    """Return a JAX-compatible f_true(x, t) closure."""
    X_grid = jnp.asarray(X_grid)
    alpha = jnp.asarray(alpha)
    def f_true(x, t):
        # x: (D,) → returns (D,)
        Kxg = _rbf_K(x[None, :], X_grid, ls_true, var_true)[0]  # (M,)
        return Kxg @ alpha
    return f_true


def make_data():
    """Simulate SDE driven by the GP-sampled drift.  Frozen oracle C, R."""
    key = jr.PRNGKey(DRIFT_SEED)
    X_grid, alpha, f_grid = sample_gp_drift(key, LS_TRUE, VAR_TRUE)
    f_true = make_drift_fn(X_grid, alpha, LS_TRUE, VAR_TRUE)

    # Run SDE.  Use a smaller t_max than Duffing since GP drift is smaller
    # in magnitude — keeps trajectory in [-3, 3]² without clipping.
    t_max = 8.0 * (T / 400.0)                        # 40
    sigma_fn = lambda x, t: SIGMA_SDE * jnp.eye(D)
    x0 = jnp.array([1.0, 0.0])
    xs = simulate_sde(jr.PRNGKey(SDE_SEED), x0=x0, f=f_true,
                      t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    # Defensively clip; trajectory should already stay bounded.
    xs = jnp.clip(xs, -3.0, 3.0)

    rng = np.random.default_rng(EMISSION_SEED)
    C_true = jnp.asarray(rng.standard_normal((N_OBS, D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(EMISSION_SEED + 1), xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(N_OBS),
              R=jnp.full((N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, SIGMA_SDE, f_true, X_grid, alpha


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def procrustes_align(m_inf, m_true):
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


def fit_efgp(lik, op, ip, t_grid, sigma, ls_init):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=N_M_INNER, mstep_lr=MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(mp=mp,
                ls_traj=np.array([ls_init] + list(hist.lengthscale)),
                var_traj=np.array([VAR_INIT] + list(hist.variance)),
                ls=float(hist.lengthscale[-1]),
                var=float(hist.variance[-1]),
                wall=wall)


def _data_aware_zs(num_per_dim, xs_np, pad=0.4):
    """Per-dim inducing grid in [min(x_d)-pad, max(x_d)+pad].  Replaces
    the symmetric ±zs_lim grid with one matched to the actual trajectory
    bbox.  ~all inducing points have data within ~ℓ for converged ℓ."""
    lo = xs_np.min(0) - pad
    hi = xs_np.max(0) + pad
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim)
                for d in range(xs_np.shape[1])]
    return jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'),
                      axis=-1).reshape(-1, xs_np.shape[1])


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init,
                   xs_np):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = _data_aware_zs(num_per_dim, xs_np)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((N_EM,), MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] +
                        [float(np.mean(h['length_scales']))
                         for h in history])
    var_traj = np.array([VAR_INIT] +
                         [float(h['output_scale']) ** 2
                          for h in history])
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls_traj=ls_traj, var_traj=var_traj,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2,
                wall=wall)


def eval_efgp_drift(mp, ls, var, t_grid, sigma, grid_pts):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(
        jnp.asarray(ls), jnp.asarray(var), X_template, eps=1e-6)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts), D_lat=D, D_out=D)
    return np.asarray(Ef)


def eval_sp_drift(s, grid_pts):
    return np.asarray(s['sd'].get_posterior_f_mean(
        s['gp_post'], s['dp'], jnp.asarray(grid_pts)))


def compute_drift_metrics(mp, xs_np, f_eval_fn, drift_fn):
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in grid_pts])
    f_pred_raw = f_eval_fn(grid_pts)
    mse_raw = float(np.mean((f_pred_raw - f_true) ** 2))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))
    rel_mse_raw = mse_raw / max(var_f, 1e-12)

    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    grid_inf = (grid_pts - b) @ np.linalg.inv(A).T
    f_pred_pc = f_eval_fn(grid_inf) @ A.T
    mse_pc = float(np.mean((f_pred_pc - f_true) ** 2))
    rel_mse_pc = mse_pc / max(var_f, 1e-12)
    return dict(rel_mse=rel_mse_pc, rel_mse_raw=rel_mse_raw,
                var_f=var_f, A=A, b=b)


def latent_recovery_rmse(mp, xs_np):
    m_inf = np.asarray(mp['m'][0])
    raw = float(np.sqrt(np.mean((m_inf - xs_np) ** 2)))
    A, b = procrustes_align(m_inf, xs_np)
    m_aligned = m_inf @ A.T + b
    pc = float(np.sqrt(np.mean((m_aligned - xs_np) ** 2)))
    return dict(raw=raw, pc=pc, A=A, b=b)


def gt_landscape(xs, sigma_drift, t_grid, LOG_LS, LOG_VAR):
    """Same GT landscape computation as Duffing bench — pseudo-velocity
    log marginal."""
    import scipy.linalg
    xs_np = np.asarray(xs)
    T_ = xs_np.shape[0]
    inputs = xs_np[:-1]
    dt = float(np.asarray(t_grid[1] - t_grid[0]))
    velocities = (xs_np[1:] - xs_np[:-1]) / dt
    n_obs = T_ - 1
    D_out = velocities.shape[1]
    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs ** 2).sum(-1)
    noise_var = sigma_drift ** 2 / dt
    eye_n = np.eye(n_obs)
    L_grid = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        ell = math.exp(float(ll))
        K_unsc = np.exp(-0.5 * sq_dists / (ell ** 2))
        for j, lv in enumerate(LOG_VAR):
            var_ = math.exp(float(lv))
            A_ = var_ * K_unsc + noise_var * eye_n
            try:
                L_chol = np.linalg.cholesky(A_)
            except np.linalg.LinAlgError:
                L_grid[i, j] = np.nan
                continue
            logdet = 2.0 * np.log(np.diag(L_chol)).sum()
            quad = 0.0
            for d in range(D_out):
                z = scipy.linalg.solve_triangular(
                    L_chol, velocities[:, d], lower=True)
                quad += float(z @ z)
            L_grid[i, j] = 0.5 * quad + 0.5 * D_out * logdet
    return L_grid


def render_landscape(T_data, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LOG_LS = T_data['LOG_LS']; LOG_VAR = T_data['LOG_VAR']
    L_gt = T_data['L_gt']; gb_ll = T_data['gb_ll']; gb_lv = T_data['gb_lv']
    results = T_data['results']

    n = len(LS_INIT_LIST)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]

    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        levels = list(np.logspace(np.log10(max(lo, hi/1000)),
                                    np.log10(hi), 10))
    else:
        levels = [1, 10, 100]

    for col, ls_init in enumerate(LS_INIT_LIST):
        ax = axes[col]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0]+levels,
                     cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                         colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        cell = results.get(ls_init, {})
        e_state = cell.get('efgp', {}).get('state')
        if e_state is not None:
            elr = cell['efgp']['latent_rmse']['pc']
            ax.plot(np.log(e_state['ls_traj']),
                     np.log(e_state['var_traj']),
                     '-o', color='C0', markersize=3, linewidth=1.6,
                     label=f"EFGP [{e_state['wall']:.0f}s, "
                            f"drift={cell['efgp']['drift']['rel_mse']:.2f}, "
                            f"lat={elr:.3f}]")
        cmap = ['C3', 'C1']
        for i, M in enumerate(M_LIST):
            s_state = cell.get('sparse', {}).get(M, {}).get('state')
            if s_state is not None:
                slr = cell['sparse'][M]['latent_rmse']['pc']
                ax.plot(np.log(s_state['ls_traj']),
                         np.log(s_state['var_traj']),
                         '-s', color=cmap[i], markersize=3, linewidth=1.4,
                         label=f"SP M={M} [{s_state['wall']:.0f}s, "
                                f"drift={cell['sparse'][M]['drift']['rel_mse']:.2f}, "
                                f"lat={slr:.3f}]")
        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6)
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                    edgecolor='k', zorder=7,
                    label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                           f"σ²={math.exp(gb_lv):.2f}]")
        # True (data-generating) θ
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                    marker='X', s=240, color='magenta',
                    edgecolor='k', zorder=8,
                    label=f"θ_true [ℓ={LS_TRUE}, σ²={VAR_TRUE}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}", fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(f"GP-drift SDE T={T} (float64) — landscape + EM "
                  f"trajectories  [θ_true=({LS_TRUE}, {VAR_TRUE})]",
                  fontsize=13)
    import matplotlib.pyplot as plt2
    plt2.tight_layout(rect=[0, 0, 1, 0.95])
    plt2.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def render_scatter(T_data, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    init_colors = {0.7: 'tab:blue', 3.0: 'tab:orange'}
    sp_marker = {25: 'o', 64: 's'}
    results = T_data['results']

    for ax, metric in zip(axes, ['drift', 'latent']):
        for ls_init in LS_INIT_LIST:
            cell = results.get(ls_init, {})
            color = init_colors[ls_init]
            e_state = cell.get('efgp', {}).get('state')
            if e_state is not None:
                v = (cell['efgp']['drift']['rel_mse'] if metric == 'drift'
                      else cell['efgp']['latent_rmse']['pc'])
                if math.isfinite(v):
                    ax.scatter(e_state['wall'], v, marker='*', s=220,
                                color=color, edgecolor='k',
                                linewidth=0.8, zorder=5)
            for M in M_LIST:
                s_state = cell.get('sparse', {}).get(M, {}).get('state')
                if s_state is None:
                    continue
                v = (cell['sparse'][M]['drift']['rel_mse']
                     if metric == 'drift'
                     else cell['sparse'][M]['latent_rmse']['pc'])
                if math.isfinite(v):
                    face = color if M == 64 else 'none'
                    ax.scatter(s_state['wall'], v, marker=sp_marker[M],
                                s=130, edgecolor=color, facecolor=face,
                                linewidths=1.5, zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('wall time (s)', fontsize=11)
        ax.set_ylabel('drift rmse / Var(f)' if metric == 'drift'
                       else 'Procrustes-aligned latent RMSE',
                       fontsize=11)
        ax.set_title(metric.title(), fontsize=12)
        ax.grid(alpha=0.3, which='both')

    legend_elements = [
        Line2D([], [], marker='*', color='w', markerfacecolor='gray',
                markersize=15, markeredgecolor='k', label='EFGP'),
        Line2D([], [], marker='o', color='gray', markerfacecolor='none',
                markersize=10, linestyle='', markeredgecolor='gray',
                label='SP M=25'),
        Line2D([], [], marker='s', color='gray', markerfacecolor='gray',
                markersize=10, linestyle='', markeredgecolor='gray',
                label='SP M=64'),
    ]
    for ls_init in LS_INIT_LIST:
        legend_elements.append(
            Line2D([], [], marker='o', color=init_colors[ls_init],
                    markersize=10, linestyle='',
                    label=f'ls_init={ls_init}'))
    fig.legend(handles=legend_elements, loc='center right',
                bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle(f"GP-drift SDE T={T} (float64) — wall vs accuracy",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def main():
    print(f"bench_gpdrift_x64: T={T} ls_init={LS_INIT_LIST} "
          f"M={M_LIST} θ_true=(ℓ={LS_TRUE}, σ²={VAR_TRUE})  x64="
          f"{jax.config.read('jax_enable_x64')}", flush=True)

    print("Sampling drift function f_true ~ GP(0, RBF(ℓ_true, σ²_true))...",
          flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = make_data()
    xs_np = np.asarray(xs)
    print(f"  Trajectory range: x[0]∈[{xs_np[:,0].min():.2f}, "
          f"{xs_np[:,0].max():.2f}], x[1]∈[{xs_np[:,1].min():.2f}, "
          f"{xs_np[:,1].max():.2f}]", flush=True)

    LOG_LS = np.linspace(LOG_LS_RANGE[0], LOG_LS_RANGE[1], N_GRID)
    LOG_VAR = np.linspace(LOG_VAR_RANGE[0], LOG_VAR_RANGE[1], N_GRID)
    print(f"Computing GT landscape ({N_GRID}×{N_GRID})...", flush=True)
    t0 = time.perf_counter()
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"(true: ℓ={LS_TRUE}, σ²={VAR_TRUE})  ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    results = {}
    for ls_init in LS_INIT_LIST:
        print(f"\n--- ls_init = {ls_init} ---", flush=True)
        cell = dict(ls_init=ls_init)

        print(f"  EFGP fit...", flush=True)
        e = fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
        ed = compute_drift_metrics(
            e['mp'], xs_np,
            f_eval_fn=lambda g: eval_efgp_drift(
                e['mp'], e['ls'], e['var'], t_grid, sigma, g),
            drift_fn=drift_fn)
        elr = latent_recovery_rmse(e['mp'], xs_np)
        A_off = float(np.linalg.norm(ed['A'] - np.eye(D)))
        b_off = float(np.linalg.norm(ed['b']))
        print(f"    ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
              f"wall={e['wall']:.1f}s", flush=True)
        print(f"    drift_raw={ed['rel_mse_raw']:.4f}  "
              f"drift_pc={ed['rel_mse']:.4f}  "
              f"lat_raw={elr['raw']:.4f}  lat_pc={elr['pc']:.4f}  "
              f"||A-I||={A_off:.3f}, ||b||={b_off:.3f}",
              flush=True)
        cell['efgp'] = dict(state=e, drift=ed, latent_rmse=elr)

        cell['sparse'] = {}
        for M in M_LIST:
            n_per = int(round(math.sqrt(M)))
            print(f"  SP M={M} ({n_per}×{n_per}) fit...", flush=True)
            s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, ls_init,
                               xs_np)
            sd = compute_drift_metrics(
                s['mp'], xs_np,
                f_eval_fn=lambda g, s_=s: eval_sp_drift(s_, g),
                drift_fn=drift_fn)
            slr = latent_recovery_rmse(s['mp'], xs_np)
            A_off = float(np.linalg.norm(sd['A'] - np.eye(D)))
            b_off = float(np.linalg.norm(sd['b']))
            print(f"    ℓ={s['ls']:.3f}, σ²={s['var']:.3f}, "
                  f"drift_raw={sd['rel_mse_raw']:.4f}  "
                  f"drift_pc={sd['rel_mse']:.4f}  "
                  f"lat_raw={slr['raw']:.4f}, lat_pc={slr['pc']:.4f}, "
                  f"||A-I||={A_off:.3f}, ||b||={b_off:.3f}, "
                  f"wall={s['wall']:.1f}s", flush=True)
            cell['sparse'][M] = dict(state=s, drift=sd, latent_rmse=slr)
        results[ls_init] = cell

        # Save partial
        save_kwargs = dict(ls_init=ls_init, T=T, xs_np=xs_np,
                            ls_true=LS_TRUE, var_true=VAR_TRUE,
                            X_grid=np.asarray(X_grid),
                            alpha=np.asarray(alpha),
                            e_ls_traj=e['ls_traj'],
                            e_var_traj=e['var_traj'],
                            e_wall=e['wall'],
                            e_m=np.asarray(e['mp']['m'][0]),
                            e_rmse_pc=ed['rel_mse'],
                            e_rmse_raw=ed['rel_mse_raw'],
                            e_lat_raw=elr['raw'],
                            e_lat_pc=elr['pc'])
        for M in M_LIST:
            sm = cell['sparse'][M]
            save_kwargs[f's{M}_ls_traj'] = sm['state']['ls_traj']
            save_kwargs[f's{M}_var_traj'] = sm['state']['var_traj']
            save_kwargs[f's{M}_wall'] = sm['state']['wall']
            save_kwargs[f's{M}_m'] = np.asarray(sm['state']['mp']['m'][0])
            save_kwargs[f's{M}_rmse_pc'] = sm['drift']['rel_mse']
            save_kwargs[f's{M}_rmse_raw'] = sm['drift']['rel_mse_raw']
            save_kwargs[f's{M}_lat_raw'] = sm['latent_rmse']['raw']
            save_kwargs[f's{M}_lat_pc'] = sm['latent_rmse']['pc']
        np.savez(f"/tmp/gpdrift_T{T}_ls{ls_init}.npz", **save_kwargs)

    T_data = dict(T=T, xs_np=xs_np, t_grid=t_grid, sigma=sigma,
                  LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                  gb_ll=gb_ll, gb_lv=gb_lv, results=results)
    render_landscape(T_data, '/tmp/gpdrift_landscape.png')
    render_scatter(T_data, '/tmp/gpdrift_scatter.png')

    # Final table
    print(f"\n{'='*100}")
    print(f"GP-drift SDE: well-specified test  "
          f"(ℓ_true={LS_TRUE}, σ²_true={VAR_TRUE}, T={T})")
    print(f"{'='*100}")
    print(f"  {'ls_init':>7s}  {'method':>9s}  {'ℓ_final':>8s}  "
          f"{'σ²_final':>9s}  {'drift_pc':>9s}  {'drift_raw':>10s}  "
          f"{'lat_pc':>7s}  {'wall':>7s}")
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e_state = cell['efgp']['state']
        d = cell['efgp']['drift']['rel_mse']
        d_r = cell['efgp']['drift']['rel_mse_raw']
        lr = cell['efgp']['latent_rmse']['pc']
        print(f"  {ls_init:>7.1f}  {'EFGP':>9s}  "
              f"{e_state['ls']:>8.3f}  {e_state['var']:>9.3f}  "
              f"{d:>9.4f}  {d_r:>10.4f}  {lr:>7.4f}  "
              f"{e_state['wall']:>6.1f}s")
        for M in M_LIST:
            s_state = cell['sparse'][M]['state']
            d = cell['sparse'][M]['drift']['rel_mse']
            d_r = cell['sparse'][M]['drift']['rel_mse_raw']
            lr = cell['sparse'][M]['latent_rmse']['pc']
            print(f"  {ls_init:>7.1f}  {f'SP M={M}':>9s}  "
                  f"{s_state['ls']:>8.3f}  {s_state['var']:>9.3f}  "
                  f"{d:>9.4f}  {d_r:>10.4f}  {lr:>7.4f}  "
                  f"{s_state['wall']:>6.1f}s")


if __name__ == '__main__':
    main()
