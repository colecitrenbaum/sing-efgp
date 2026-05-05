"""
bench_overnight_sweep.py

Overnight sweep: 3 benchmarks × 3 T's × (1 EFGP + 3 SparseGP-num_inducing).

For each (T, benchmark), produce:
  - GT log-marginal landscape with EFGP + SparseGP-{25,64,144} trajectories
    overlaid, legend includes wall-time
  - Drift fields (truth | EFGP | SparseGP variants)

And one global scatter:
  - x = wall-time, y = relative drift error (MSE(f) / Var(f))
  - color/marker by method × num_inducing × T

Run:
  ~/myenv/bin/python demos/bench_overnight_sweep.py
"""
from __future__ import annotations

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

import jax
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
LS_INIT = 0.7
VAR_INIT = 1.0
MSTEP_LR = 0.01
N_M_INNER = 4

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

# Sweep matrix
T_LIST = [400, 2000, 5000]
M_LIST = [25, 64, 144]      # total inducing pts (5², 8², 12²)
N_EM_LIST = [50, 100]       # separate fits per n_em — honest walls + metrics

# Base benchmarks (T and t_max get scaled per T_LIST entry)
BENCHMARKS_BASE = [
    dict(name='Damped rotation (linear)', short='linear',
          drift_fn=lambda x, t: _A_rot @ x,
          x0=jnp.array([2.0, 0.0]),
          T_base=400, t_max_base=8.0, sigma=0.3, N_obs=8, seed=7),
    dict(name='Duffing double-well', short='duffing',
          drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
          x0=jnp.array([1.2, 0.0]),
          T_base=400, t_max_base=15.0, sigma=0.2, N_obs=8, seed=13),
    dict(name='Anharmonic oscillator', short='anharmonic',
          drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3]),
          x0=jnp.array([1.5, 0.0]),
          T_base=400, t_max_base=10.0, sigma=0.3, N_obs=8, seed=21),
]


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg, T):
    """Frozen oracle emissions; T scales t_max proportionally."""
    t_max = cfg['t_max_base'] * (T / cfg['T_base'])
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=t_max,
                      n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def procrustes_align(m_inf, m_true):
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


def fit_efgp(lik, op, ip, t_grid, sigma, n_em):
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=N_M_INNER, mstep_lr=MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        mp=mp, ls_traj=np.array([LS_INIT] + list(hist.lengthscale)),
        var_traj=np.array([VAR_INIT] + list(hist.variance)),
        ls=float(hist.lengthscale[-1]), var=float(hist.variance[-1]),
        wall=wall,
    )


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, n_em):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    history = []
    t0 = time.perf_counter()
    mp, nat_p, gp_post, dp_final, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=n_em, n_iters_e=10,
        n_iters_m=N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((n_em,), MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT] +
                        [float(np.mean(h['length_scales']))
                         for h in history])
    var_traj = np.array([VAR_INIT] +
                         [float(h['output_scale']) ** 2
                          for h in history])
    return dict(
        mp=mp, sd=sparse, gp_post=gp_post, dp=dp_final,
        ls_traj=ls_traj, var_traj=var_traj,
        ls=float(jnp.mean(dp_final['length_scales'])),
        var=float(dp_final['output_scale']) ** 2,
        wall=wall,
    )


def gt_landscape(xs, sigma_drift, t_grid, LOG_LS, LOG_VAR):
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


def eval_efgp_drift(mp, ls, var, t_grid, sigma, grid_pts):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(
        jnp.asarray(ls, dtype=jnp.float32),
        jnp.asarray(var, dtype=jnp.float32),
        X_template, eps=1e-6)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    return np.asarray(Ef)


def eval_sp_drift(s, grid_pts):
    return np.asarray(s['sd'].get_posterior_f_mean(
        s['gp_post'], s['dp'],
        jnp.asarray(grid_pts, dtype=jnp.float32)))


def compute_drift_metrics(mp, xs_np, f_eval_fn, drift_fn, lo_hi=None):
    """Return relative MSE = MSE(f_pred, f_true) / Var(f_true) on a grid
    from xs's bounding box.  Procrustes-align inferred latent before
    pulling the grid back into latent frame."""
    if lo_hi is None:
        lo = xs_np.min(0) - 0.4
        hi = xs_np.max(0) + 0.4
    else:
        lo, hi = lo_hi
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in grid_pts])
    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    grid_inf = (grid_pts - b) @ np.linalg.inv(A).T
    f_pred = f_eval_fn(grid_inf) @ A.T

    mse = float(np.mean((f_pred - f_true) ** 2))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))
    rel_mse = mse / max(var_f, 1e-12)

    return dict(
        f_true=f_true, f_pred=f_pred, GX=GX, GY=GY,
        mse=mse, var_f=var_f, rel_mse=rel_mse,
        zero_baseline_rmse=float(np.sqrt(np.mean(f_true ** 2))),
    )


def run_one(cfg, T, M_list, n_em):
    print(f"\n[bench] === {cfg['name']} | T={T} | n_em={n_em} ===",
          flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(cfg, T)
    sigma = cfg['sigma']
    xs_np = np.asarray(xs)

    # EFGP (one fit per T, n_em)
    print(f"  EFGP fit...", flush=True)
    e = fit_efgp(lik, op, ip, t_grid, sigma, n_em)
    print(f"    final ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
          f"wall={e['wall']:.1f}s", flush=True)
    e_metrics = compute_drift_metrics(
        e['mp'], xs_np,
        f_eval_fn=lambda g: eval_efgp_drift(e['mp'], e['ls'], e['var'],
                                              t_grid, sigma, g),
        drift_fn=cfg['drift_fn'])
    print(f"    drift rel_mse={e_metrics['rel_mse']:.3f}", flush=True)

    # SparseGP per total-M (M ∈ {25, 64, 144} mapped to 5×5, 8×8, 12×12 grid)
    sp_results = {}
    for M in M_list:
        n_per_dim_2d = int(round(math.sqrt(M)))
        assert n_per_dim_2d ** 2 == M, f"M={M} not a perfect square"
        print(f"  SparseGP fit (M={M} = {n_per_dim_2d}×{n_per_dim_2d})...",
              flush=True)
        s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per_dim_2d, n_em)
        print(f"    final ℓ={s['ls']:.3f}, σ²={s['var']:.3f}, "
              f"wall={s['wall']:.1f}s", flush=True)
        s_metrics = compute_drift_metrics(
            s['mp'], xs_np,
            f_eval_fn=lambda g, s_=s: eval_sp_drift(s_, g),
            drift_fn=cfg['drift_fn'])
        print(f"    drift rel_mse={s_metrics['rel_mse']:.3f}", flush=True)
        sp_results[M] = dict(state=s, metrics=s_metrics)

    # GT landscape (one per T, benchmark)
    LOG_LS = np.linspace(LOG_LS_RANGE[0], LOG_LS_RANGE[1], N_GRID)
    LOG_VAR = np.linspace(LOG_VAR_RANGE[0], LOG_VAR_RANGE[1], N_GRID)
    print(f"  GT landscape (T={T})...", flush=True)
    t0 = time.perf_counter()
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"    GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    out = dict(
        cfg=cfg, T=T, n_em=n_em, xs_np=xs_np, t_grid=t_grid, sigma=sigma,
        LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
        L_gt_min=float(L_gt[gb]), gb_ll=gb_ll, gb_lv=gb_lv,
        efgp=dict(state=e, metrics=e_metrics),
        sparse=sp_results,
    )
    np.savez(f"/tmp/sweep_{cfg['short']}_T{T}_nem{n_em}.npz",
              T=T, name=cfg['name'],
              LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
              L_gt_min=float(L_gt[gb]), gb_ll=gb_ll, gb_lv=gb_lv,
              e_ls_traj=e['ls_traj'], e_var_traj=e['var_traj'],
              e_wall=e['wall'], e_rel_mse=e_metrics['rel_mse'],
              **{f"s{M}_ls_traj": sp_results[M]['state']['ls_traj']
                  for M in M_list},
              **{f"s{M}_var_traj": sp_results[M]['state']['var_traj']
                  for M in M_list},
              **{f"s{M}_wall": sp_results[M]['state']['wall']
                  for M in M_list},
              **{f"s{M}_rel_mse": sp_results[M]['metrics']['rel_mse']
                  for M in M_list},
              )
    return out


def render_landscape_overlay(results_for_T, T, M_list, n_em,
                                out_png):
    """One figure per (T, n_em) with 1 row × 3 cols (one per benchmark).
    Each panel is the GT log-marginal landscape with EFGP + SparseGP-{M}
    trajectories overlaid.  Legend includes wall time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for col, r in enumerate(results_for_T):
        ax = axes[col]
        LOG_LS = r['LOG_LS']; LOG_VAR = r['LOG_VAR']
        L_norm = r['L_gt'] - r['L_gt_min']
        finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            levels = list(np.logspace(np.log10(max(lo, hi/1000)),
                                        np.log10(hi), 10))
        else:
            levels = [1, 10, 100]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0]+levels,
                     cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                         colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        # EFGP traj
        e = r['efgp']
        e_ls = e['state']['ls_traj']
        e_var = e['state']['var_traj']
        ax.plot(np.log(e_ls), np.log(e_var), '-o', color='C0',
                 markersize=3, linewidth=1.6,
                 label=f"EFGP [{e['state']['wall']:.0f}s, "
                        f"rmse²/var={e['metrics']['rel_mse']:.2f}]")

        # SparseGP per total-M
        cmap = ['C3', 'C1', 'C2']
        for i, M in enumerate(M_list):
            s = r['sparse'][M]
            s_ls = s['state']['ls_traj']
            s_var = s['state']['var_traj']
            ax.plot(np.log(s_ls), np.log(s_var), '-s',
                     color=cmap[i], markersize=3, linewidth=1.4,
                     label=f"SparseGP M={M} "
                            f"[{s['state']['wall']:.0f}s, "
                            f"rmse²/var={s['metrics']['rel_mse']:.2f}]")

        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6)
        ax.scatter([r['gb_ll']], [r['gb_lv']],
                    marker='*', s=240, color='gold', edgecolor='k',
                    zorder=7,
                    label=f"GT MLE [ℓ={math.exp(r['gb_ll']):.2f}, "
                            f"σ²={math.exp(r['gb_lv']):.2f}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0: ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"{r['cfg']['name']}", fontsize=11)
        ax.legend(loc='lower right', fontsize=7)

    fig.suptitle(f"GT log-marginal landscape (T={T}, n_em={n_em})  "
                  "EFGP + SparseGP overlay  [oracle C frozen, lr=0.01, "
                  f"n_iters_m={N_M_INNER}]", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"[bench] Saved {out_png}", flush=True)


def render_scatter(all_results, M_list, n_em, out_png):
    """Time vs relative-accuracy scatter for one n_em value."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bench_names = list({r['cfg']['short']: r['cfg']['name']
                         for results in all_results.values() for r in results}.values())
    bench_shorts = [r['cfg']['short'] for r in next(iter(all_results.values()))]

    sp_markers = {25: 'o', 64: 's', 144: '^'}
    T_colors = {400: 'tab:blue', 2000: 'tab:orange', 5000: 'tab:red'}

    for col, short in enumerate(bench_shorts):
        ax = axes[col]
        for T, results in all_results.items():
            r = next(rr for rr in results if rr['cfg']['short'] == short)
            # EFGP point
            ax.scatter(r['efgp']['state']['wall'],
                        r['efgp']['metrics']['rel_mse'],
                        marker='*', s=200, color=T_colors[T],
                        edgecolor='k', zorder=5,
                        label=f"EFGP T={T}")
            # SparseGP points (one per M)
            for M in M_list:
                s = r['sparse'][M]
                ax.scatter(s['state']['wall'], s['metrics']['rel_mse'],
                            marker=sp_markers[M], s=120,
                            color=T_colors[T], edgecolor='k',
                            facecolor='none' if M != 64 else T_colors[T],
                            linewidths=1.5, zorder=4,
                            label=f"SparseGP M={M} T={T}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('wall time (s)', fontsize=10)
        if col == 0:
            ax.set_ylabel('relative drift error  MSE(f) / Var(f)', fontsize=10)
        ax.set_title(bench_names[col], fontsize=11)
        ax.grid(alpha=0.3, which='both')
        # Custom legend: list T in colors, M in markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([], [], marker='*', color='w', markerfacecolor='gray',
                    markersize=14, markeredgecolor='k', label='EFGP'),
        ]
        for M in M_list:
            face = 'gray' if M == 64 else 'none'
            legend_elements.append(
                Line2D([], [], marker=sp_markers[M], color='gray',
                        markerfacecolor=face, markersize=10,
                        markeredgecolor='gray', linestyle='', label=f"SparseGP M={M}"))
        for T in [400, 2000, 5000]:
            legend_elements.append(
                Line2D([], [], marker='o', color=T_colors[T], markersize=10,
                        linestyle='', label=f'T={T}'))
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
    fig.suptitle(f"Wall-time vs relative-drift-error  (n_em={n_em})  "
                  "[oracle C frozen, matched lr/iters]", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"[bench] Saved {out_png}", flush=True)


def main():
    print(f"[bench] Overnight sweep: T={T_LIST}, M_total={M_LIST}, "
          f"n_em={N_EM_LIST}", flush=True)
    print(f"[bench] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), "
          f"lr={MSTEP_LR}, n_m_inner={N_M_INNER}", flush=True)

    all_results_by_nem = {n_em: {} for n_em in N_EM_LIST}
    for n_em in N_EM_LIST:
        for T in T_LIST:
            results = []
            for cfg in BENCHMARKS_BASE:
                r = run_one(cfg, T, M_LIST, n_em)
                results.append(r)
            all_results_by_nem[n_em][T] = results
            # Render landscape overlay per (T, n_em) immediately so partial
            # progress is preserved if the script dies overnight.
            render_landscape_overlay(
                results, T, M_LIST, n_em,
                out_png=f'/tmp/sweep_landscape_T{T}_nem{n_em}.png')
            # Re-render scatter with whatever n_em-specific data is in so far.
            render_scatter(
                all_results_by_nem[n_em], M_LIST, n_em,
                out_png=f'/tmp/sweep_time_vs_acc_nem{n_em}.png')

    print(f"\n[bench] Final summary:", flush=True)
    print(f"  {'cfg':25s}  {'T':>5s}  {'n_em':>5s}  {'method':>14s}  "
          f"{'wall':>8s}  {'rel_mse':>10s}  {'ℓ_final':>8s}", flush=True)
    for n_em in N_EM_LIST:
        for T, results in all_results_by_nem[n_em].items():
            for r in results:
                cfg_name = r['cfg']['name'][:24]
                e = r['efgp']
                print(f"  {cfg_name:25s}  {T:>5d}  {n_em:>5d}  "
                      f"{'EFGP':>14s}  "
                      f"{e['state']['wall']:>7.1f}s  "
                      f"{e['metrics']['rel_mse']:>10.4f}  "
                      f"{e['state']['ls']:>8.3f}", flush=True)
                for M in M_LIST:
                    s = r['sparse'][M]
                    print(f"  {cfg_name:25s}  {T:>5d}  {n_em:>5d}  "
                          f"{f'SP M={M}':>14s}  "
                          f"{s['state']['wall']:>7.1f}s  "
                          f"{s['metrics']['rel_mse']:>10.4f}  "
                          f"{s['state']['ls']:>8.3f}", flush=True)


if __name__ == '__main__':
    main()
