"""
bench_duffing_lsinit_x64.py

Duffing double-well at T=2000 and T=5000, sweeping ls_init across
{0.7, 1.5, 3.0, 5.0} for EFGP + SparseGP M={25, 64}.  Float64 enabled
globally so the backward-Cholesky NaN issue doesn't fire.

Renders incrementally:
  - after T=2000 finishes: landscape overlay + per-T scatter + table
  - after T=5000 finishes: same again with both T's

Saves .npz per cell so partial progress survives an interrupt.

Run:
  ~/myenv/bin/python demos/bench_duffing_lsinit_x64.py
"""
from __future__ import annotations

# Enable x64 BEFORE any jax import that allocates an array.
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
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
VAR_INIT = 1.0

T_LIST = [5000]  # T=2000 already done; re-running just T=5000
LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64]                  # skip M=144 — keeps wall manageable

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21


CFG = dict(name='Duffing double-well', short='duffing',
            drift_fn=lambda x, t: jnp.stack(
                [x[1], x[0] - x[0]**3 - 0.5*x[1]]),
            x0=jnp.array([1.2, 0.0]),
            T_base=400, t_max_base=15.0, sigma=0.2, N_obs=8, seed=13)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(T):
    t_max = CFG['t_max_base'] * (T / CFG['T_base'])
    sigma = CFG['sigma']
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(CFG['seed']), x0=CFG['x0'],
                      f=CFG['drift_fn'], t_max=t_max,
                      n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(CFG['seed'])
    C_true = jnp.asarray(rng.standard_normal((CFG['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(CFG['N_obs']),
                    R=jnp.full((CFG['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(CFG['seed'] + 1), xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(CFG['N_obs']),
              R=jnp.full((CFG['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid, sigma


def procrustes_align(m_inf, m_true):
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


def fit_efgp(lik, op, ip, t_grid, sigma, ls_init):
    """Single EM at eps_grid=1e-3.  (Refinement at eps=1e-4 was checked
    against the un-refined smoother and gave Δdrift ≈ 0.0002 at +10.8s
    cost — not worth keeping.)"""
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


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=num_per_dim)
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
        s['gp_post'], s['dp'],
        jnp.asarray(grid_pts)))


def compute_drift_metrics(mp, xs_np, f_eval_fn, drift_fn):
    """Both Procrustes-aligned and raw drift rmse, plus the alignment
    A/b so we can verify Procrustes is doing nothing under oracle C."""
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in grid_pts])

    # Raw (no Procrustes — eval directly in oracle frame)
    f_pred_raw = f_eval_fn(grid_pts)
    mse_raw = float(np.mean((f_pred_raw - f_true) ** 2))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))
    rel_mse_raw = mse_raw / max(var_f, 1e-12)

    # Procrustes-aligned (legacy)
    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    grid_inf = (grid_pts - b) @ np.linalg.inv(A).T
    f_pred_pc = f_eval_fn(grid_inf) @ A.T
    mse_pc = float(np.mean((f_pred_pc - f_true) ** 2))
    rel_mse_pc = mse_pc / max(var_f, 1e-12)

    return dict(rel_mse=rel_mse_pc, rel_mse_raw=rel_mse_raw,
                var_f=var_f, A=A, b=b)


def latent_recovery_rmse(mp, xs_np):
    """Both raw and Procrustes-aligned latent recovery RMSE."""
    m_inf = np.asarray(mp['m'][0])
    raw = float(np.sqrt(np.mean((m_inf - xs_np) ** 2)))
    A, b = procrustes_align(m_inf, xs_np)
    m_aligned = m_inf @ A.T + b
    pc = float(np.sqrt(np.mean((m_aligned - xs_np) ** 2)))
    return dict(raw=raw, pc=pc, A=A, b=b)


def run_T(T):
    print(f"\n{'='*70}\nT = {T}\n{'='*70}", flush=True)
    xs, ys, lik, op, ip, t_grid, sigma = make_data(T)
    xs_np = np.asarray(xs)

    LOG_LS = np.linspace(LOG_LS_RANGE[0], LOG_LS_RANGE[1], N_GRID)
    LOG_VAR = np.linspace(LOG_VAR_RANGE[0], LOG_VAR_RANGE[1], N_GRID)
    print(f"  Computing GT landscape (one-time per T)...", flush=True)
    t0 = time.perf_counter()
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    results = {}  # ls_init -> dict
    for ls_init in LS_INIT_LIST:
        print(f"\n--- ls_init = {ls_init} ---", flush=True)
        cell = dict(ls_init=ls_init)

        # EFGP
        print(f"  EFGP fit (ls_init={ls_init})...", flush=True)
        try:
            e = fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
            ed = compute_drift_metrics(
                e['mp'], xs_np,
                f_eval_fn=lambda g: eval_efgp_drift(
                    e['mp'], e['ls'], e['var'], t_grid, sigma, g),
                drift_fn=CFG['drift_fn'])
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
        except Exception as ex:
            print(f"    EXC: {type(ex).__name__}: {ex}", flush=True)
            cell['efgp'] = dict(error=str(ex))

        # SparseGP
        cell['sparse'] = {}
        for M in M_LIST:
            n_per = int(round(math.sqrt(M)))
            print(f"  SP M={M} ({n_per}×{n_per}) fit...", flush=True)
            try:
                s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, ls_init)
                sd = compute_drift_metrics(
                    s['mp'], xs_np,
                    f_eval_fn=lambda g, s_=s: eval_sp_drift(s_, g),
                    drift_fn=CFG['drift_fn'])
                slr = latent_recovery_rmse(s['mp'], xs_np)
                A_off = float(np.linalg.norm(sd['A'] - np.eye(D)))
                b_off = float(np.linalg.norm(sd['b']))
                ls_str = (f"{s['ls']:.3f}"
                           if math.isfinite(s['ls']) else 'NaN')
                draw = (f"{sd['rel_mse_raw']:.3f}"
                        if math.isfinite(sd['rel_mse_raw']) else 'NaN')
                dpc = (f"{sd['rel_mse']:.3f}"
                       if math.isfinite(sd['rel_mse']) else 'NaN')
                print(f"    ℓ={ls_str}, σ²={s['var']:.3f}, "
                      f"drift_raw={draw}, drift_pc={dpc}, "
                      f"lat_raw={slr['raw']:.4f}, lat_pc={slr['pc']:.4f}, "
                      f"||A-I||={A_off:.3f}, ||b||={b_off:.3f}, "
                      f"wall={s['wall']:.1f}s", flush=True)
                cell['sparse'][M] = dict(state=s, drift=sd, latent_rmse=slr)
            except Exception as ex:
                print(f"    EXC: {type(ex).__name__}: {ex}", flush=True)
                cell['sparse'][M] = dict(error=str(ex))
        results[ls_init] = cell

        # Save partial after each ls_init.  Include m_inferred per method
        # so we can recompute alignment-free metrics post-hoc.
        def _g(d, *keys, default=np.nan):
            for k in keys:
                if not isinstance(d, dict): return default
                d = d.get(k, default)
                if d is default: return default
            return d
        save_kwargs = dict(ls_init=ls_init, T=T, xs_np=xs_np)
        e_state = _g(cell, 'efgp', 'state')
        if isinstance(e_state, dict):
            save_kwargs.update(
                e_ls_traj=e_state.get('ls_traj', np.array([np.nan])),
                e_var_traj=e_state.get('var_traj', np.array([np.nan])),
                e_wall=e_state.get('wall', np.nan),
                e_m=np.asarray(e_state.get('mp', {}).get(
                    'm', np.zeros((1, 1, D)))[0]),
            )
        save_kwargs.update(
            e_rmse_pc=_g(cell, 'efgp', 'drift', 'rel_mse'),
            e_rmse_raw=_g(cell, 'efgp', 'drift', 'rel_mse_raw'),
            e_lat_raw=_g(cell, 'efgp', 'latent_rmse', 'raw'),
            e_lat_pc=_g(cell, 'efgp', 'latent_rmse', 'pc'),
        )
        for M in M_LIST:
            s_state = _g(cell, 'sparse', M, 'state')
            if isinstance(s_state, dict):
                save_kwargs[f's{M}_ls_traj'] = s_state.get(
                    'ls_traj', np.array([np.nan]))
                save_kwargs[f's{M}_var_traj'] = s_state.get(
                    'var_traj', np.array([np.nan]))
                save_kwargs[f's{M}_wall'] = s_state.get('wall', np.nan)
                save_kwargs[f's{M}_m'] = np.asarray(s_state.get(
                    'mp', {}).get('m', np.zeros((1, 1, D)))[0])
            save_kwargs[f's{M}_rmse_pc'] = _g(
                cell, 'sparse', M, 'drift', 'rel_mse')
            save_kwargs[f's{M}_rmse_raw'] = _g(
                cell, 'sparse', M, 'drift', 'rel_mse_raw')
            save_kwargs[f's{M}_lat_raw'] = _g(
                cell, 'sparse', M, 'latent_rmse', 'raw')
            save_kwargs[f's{M}_lat_pc'] = _g(
                cell, 'sparse', M, 'latent_rmse', 'pc')
        np.savez(f"/tmp/duffx64_T{T}_ls{ls_init}.npz", **save_kwargs)

    return dict(T=T, xs_np=xs_np, t_grid=t_grid, sigma=sigma,
                LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                gb_ll=gb_ll, gb_lv=gb_lv, results=results)


def render_landscape(T_data, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = T_data['T']
    LOG_LS = T_data['LOG_LS']; LOG_VAR = T_data['LOG_VAR']
    L_gt = T_data['L_gt']; gb_ll = T_data['gb_ll']; gb_lv = T_data['gb_lv']
    results = T_data['results']

    n = len(LS_INIT_LIST)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), sharey=True)
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

        cell = results[ls_init]
        # EFGP
        e_state = cell.get('efgp', {}).get('state')
        if e_state is not None:
            elr = cell['efgp']['latent_rmse']
            ax.plot(np.log(e_state['ls_traj']),
                     np.log(e_state['var_traj']),
                     '-o', color='C0', markersize=3, linewidth=1.6,
                     label=f"EFGP [{e_state['wall']:.0f}s, "
                            f"drift={cell['efgp']['drift']['rel_mse']:.2f}, "
                            f"lat={elr['pc']:.3f}]")
        # SparseGP per-M
        cmap = ['C3', 'C1']
        for i, M in enumerate(M_LIST):
            s_state = cell.get('sparse', {}).get(M, {}).get('state')
            if s_state is not None:
                slr = cell['sparse'][M]['latent_rmse']
                ax.plot(np.log(s_state['ls_traj']),
                         np.log(s_state['var_traj']),
                         '-s', color=cmap[i], markersize=3, linewidth=1.4,
                         label=f"SP M={M} [{s_state['wall']:.0f}s, "
                                f"drift={cell['sparse'][M]['drift']['rel_mse']:.2f}, "
                                f"lat={slr['pc']:.3f}]")
        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6)
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                    edgecolor='k', zorder=7,
                    label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                           f"σ²={math.exp(gb_lv):.2f}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}", fontsize=11)
        ax.legend(loc='lower right', fontsize=7)
    fig.suptitle(f"Duffing T={T} (float64) — landscape + trajectories per "
                  "ls_init", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def render_scatter(all_T, out_png):
    """Wall vs (drift, latent) scatter, one row per metric, one col per T."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Ts = sorted(all_T.keys())
    fig, axes = plt.subplots(2, len(Ts), figsize=(7 * len(Ts), 11),
                              squeeze=False)

    init_colors = {0.7: 'tab:blue', 1.5: 'tab:green',
                    3.0: 'tab:orange', 5.0: 'tab:red'}
    sp_marker = {25: 'o', 64: 's'}

    for col, T in enumerate(Ts):
        T_data = all_T[T]
        for row, metric_name, key in [(0, 'drift rmse / Var(f)', 'drift_rmse'),
                                         (1, 'latent recovery RMSE', 'latent_rmse')]:
            ax = axes[row, col]
            for ls_init in LS_INIT_LIST:
                if ls_init not in T_data['results']:
                    continue
                cell = T_data['results'][ls_init]
                color = init_colors.get(ls_init, 'gray')

                # EFGP
                e_state = cell.get('efgp', {}).get('state')
                if e_state is not None:
                    if key == 'drift_rmse':
                        v = cell['efgp']['drift']['rel_mse']
                    else:
                        v = cell['efgp']['latent_rmse']['pc']
                    if math.isfinite(v):
                        ax.scatter(e_state['wall'], v, marker='*', s=220,
                                    color=color, edgecolor='k',
                                    linewidth=0.8, zorder=5)

                # SP per-M
                for M in M_LIST:
                    s_state = cell.get('sparse', {}).get(M, {}).get('state')
                    if s_state is not None:
                        if key == 'drift_rmse':
                            v = cell['sparse'][M]['drift']['rel_mse']
                        else:
                            v = cell['sparse'][M]['latent_rmse']['pc']
                        if math.isfinite(v):
                            face = color if M == 64 else 'none'
                            ax.scatter(s_state['wall'], v,
                                        marker=sp_marker[M], s=120,
                                        edgecolor=color, facecolor=face,
                                        linewidths=1.5, zorder=4)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('wall time (s)', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f"T = {T}", fontsize=11)
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
            Line2D([], [], marker='o',
                    color=init_colors.get(ls_init, 'gray'),
                    markersize=10, linestyle='',
                    label=f'ls_init = {ls_init}'))
    fig.legend(handles=legend_elements, loc='center right',
                bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle(f"Duffing — wall time vs accuracy (float64)",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def print_table(all_T):
    """Tabular summary across all T's."""
    print(f"\n{'='*100}\nFINAL TABLE\n{'='*100}")
    print(f"  {'T':>5s}  {'ls_init':>7s}  {'method':>9s}  "
          f"{'ℓ_final':>8s}  {'σ²_final':>9s}  {'drift':>7s}  "
          f"{'latent':>7s}  {'wall':>7s}")
    for T in sorted(all_T.keys()):
        T_data = all_T[T]
        for ls_init in LS_INIT_LIST:
            if ls_init not in T_data['results']:
                continue
            cell = T_data['results'][ls_init]
            e_state = cell.get('efgp', {}).get('state')
            if e_state is not None:
                d = cell['efgp']['drift']['rel_mse']
                lr = cell['efgp']['latent_rmse']['pc']
                print(f"  {T:>5d}  {ls_init:>7.1f}  {'EFGP':>9s}  "
                      f"{e_state['ls']:>8.3f}  {e_state['var']:>9.3f}  "
                      f"{d:>7.3f}  {lr:>7.4f}  {e_state['wall']:>6.0f}s")
            for M in M_LIST:
                s_state = cell.get('sparse', {}).get(M, {}).get('state')
                if s_state is not None:
                    d = cell['sparse'][M]['drift']['rel_mse']
                    lr = cell['sparse'][M]['latent_rmse']['pc']
                    ls_s = (f"{s_state['ls']:>8.3f}"
                             if math.isfinite(s_state['ls']) else f"{'NaN':>8s}")
                    var_s = (f"{s_state['var']:>9.3f}"
                              if math.isfinite(s_state['var']) else f"{'NaN':>9s}")
                    d_s = (f"{d:>7.3f}" if math.isfinite(d)
                           else f"{'NaN':>7s}")
                    lr_s = (f"{lr:>7.4f}" if math.isfinite(lr)
                            else f"{'NaN':>7s}")
                    print(f"  {T:>5d}  {ls_init:>7.1f}  {f'SP M={M}':>9s}  "
                          f"{ls_s}  {var_s}  {d_s}  {lr_s}  "
                          f"{s_state['wall']:>6.0f}s")


def main():
    print(f"bench_duffing_lsinit_x64: T={T_LIST}, ls_init={LS_INIT_LIST}, "
          f"M={M_LIST}, x64={jax.config.read('jax_enable_x64')}", flush=True)

    all_T = {}
    for T in T_LIST:
        T_data = run_T(T)
        all_T[T] = T_data
        # Render after this T finishes — partial result safe.
        print(f"\n[render] After T={T} ...", flush=True)
        render_landscape(T_data,
                          f'/tmp/duffx64_landscape_T{T}.png')
        render_scatter(all_T,
                        f'/tmp/duffx64_scatter_through_T{T}.png')
        print_table(all_T)

    # Final combined render
    render_scatter(all_T, '/tmp/duffx64_scatter_final.png')
    print_table(all_T)


if __name__ == '__main__':
    main()
