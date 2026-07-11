"""
bench_gpdrift_inducing_sweep_x64.py

Does inducing-point SparseGP converge to EFGP (and to the ORACLE MLE) as the
number of inducing points grows, on a WELL-SPECIFIED synthetic problem?

Setup (reused verbatim from bench_gpdrift_x64):
  Draw a single drift  f_true ~ GP(0, RBF(ℓ_true=1.0, σ²_true=1.5)),  run an SDE
  with it, then ask EFGP and SparseGP to recover it.  Because the data IS in the
  model class, there is a genuine oracle: the pseudo-velocity log-marginal MLE on
  the true latent paths (`gt_landscape`).  EFGP is known to sit at that MLE; the
  open question is whether SparseGP marches to it as M grows.

  This repeats the neural-data inducing sweep
  (demos/bench_neural_inducing_sweep_x64.py) on well-specified synthetic data,
  where — unlike real neural data — there is a ground-truth θ and an oracle MLE
  to converge to.

Training hypers are MATCHED across methods (N_EM=50, rho_sched=linspace(0.05,0.7,50),
mstep_lr=0.01, n_estep=10, n_mstep=4) — inherited by importing bench_gpdrift_x64's
fit_efgp / fit_sparsegp — so the ONLY difference between methods is the GP-fit method.

Sweep: M ∈ {25, 64, 144, 256}  (num_per_dim 5,8,12,16), T=2000, ls_init ∈ {0.7, 3.0}.
Inducing grid is data-aware (bench._data_aware_zs): it densifies within the data
bbox as M grows, which is the correct way to test M-convergence.

Deliverables (in demos/_bench_gpdrift_inducing_sweep_out/):
  landscape_paths.png   PRIMARY — GT oracle contours + EM learning path per method
                        (EFGP bold blue; SparseGP one path per M, light→dark viridis)
  convergence.png       recovered ℓ / σ² / wall vs #inducing, with EFGP + oracle-MLE
                        reference lines
  sweep.npz             raw results

Run under Slurm (demos/bench_gpdrift_inducing_sweep.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax.* use (CLAUDE.md)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp

# Reuse ALL data / fit / metric / landscape machinery from the head-to-head demo.
# This guarantees identical data and identical matched training hypers — the only
# thing that varies here is the inducing count.
import demos.bench_gpdrift_x64 as bench
import sing.efgp_em as efgp_em

D = bench.D
T = bench.T
LS_TRUE = bench.LS_TRUE
VAR_TRUE = bench.VAR_TRUE
VAR_INIT = bench.VAR_INIT
LOG_LS_RANGE = bench.LOG_LS_RANGE
LOG_VAR_RANGE = bench.LOG_VAR_RANGE
N_GRID = bench.N_GRID

LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64, 144, 256]        # 5×5, 8×8, 12×12, 16×16

OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


def fit_efgp(lik, op, ip, t_grid, sigma, ls_init):
    """EFGP fit with the SAME matched training hypers as bench.fit_efgp, but
    additionally returns the EM history so drift can be evaluated from the
    EM's ACTUAL converged q(f) via efgp_em.posterior_drift_mean (bench's own
    eval_efgp_drift recomputes mu_r through a now-stale compute_mu_r_jax
    signature)."""
    rho_sched = jnp.linspace(0.05, 0.7, bench.N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=bench.VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=bench.N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=bench.N_M_INNER, mstep_lr=bench.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(mp=mp, hist=hist,
                ls_traj=np.array([ls_init] + list(hist.lengthscale)),
                var_traj=np.array([bench.VAR_INIT] + list(hist.variance)),
                ls=float(hist.lengthscale[-1]),
                var=float(hist.variance[-1]),
                wall=wall)


def eval_efgp_drift(hist, grid_pts):
    """Posterior drift mean from the EM's converged q(f) (current API)."""
    return efgp_em.posterior_drift_mean(hist, grid_pts)


def render_landscape(T_data, out_png):
    """GT oracle log-marginal contours with EM learning paths overlaid.

    One panel per ls_init.  EFGP path bold blue; one SparseGP path per inducing
    count coloured light→dark (viridis) so the eye tracks how the endpoints march
    with M.  Markers: init (+), oracle GT MLE (gold *), θ_true (magenta X)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

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
        levels = list(np.logspace(np.log10(max(lo, hi / 1000)),
                                  np.log10(hi), 10))
    else:
        levels = [1, 10, 100]

    # light→dark as M grows
    sp_colors = cm.viridis(np.linspace(0.15, 0.9, len(M_LIST)))

    for col, ls_init in enumerate(LS_INIT_LIST):
        ax = axes[col]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                    cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                        colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        cell = results.get(ls_init, {})

        # SparseGP paths (draw first so EFGP sits on top)
        for i, M in enumerate(M_LIST):
            sm = cell.get('sparse', {}).get(M)
            if sm is None or np.any(np.isnan(sm['state']['ls_traj'])):
                continue
            st = sm['state']
            ax.plot(np.log(st['ls_traj']), np.log(st['var_traj']),
                    '-s', color=sp_colors[i], markersize=3, linewidth=1.4,
                    alpha=0.9,
                    label=f"SP M={M} [ℓ={st['ls']:.2f}, σ²={st['var']:.2f}, "
                          f"{st['wall']:.0f}s, drift={sm['drift']['rel_mse']:.2f}]")

        # EFGP path (bold, on top)
        e_state = cell.get('efgp', {}).get('state')
        if e_state is not None:
            ed = cell['efgp']['drift']['rel_mse']
            ax.plot(np.log(e_state['ls_traj']), np.log(e_state['var_traj']),
                    '-o', color='C0', markersize=3.5, linewidth=2.2, zorder=8,
                    label=f"EFGP [ℓ={e_state['ls']:.2f}, σ²={e_state['var']:.2f}, "
                          f"{e_state['wall']:.0f}s, drift={ed:.2f}]")

        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                   marker='+', s=200, color='black', zorder=9, label='init')
        ax.scatter([gb_ll], [gb_lv], marker='*', s=260, color='gold',
                   edgecolor='k', zorder=10,
                   label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                         f"σ²={math.exp(gb_lv):.2f}]")
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                   marker='X', s=240, color='magenta', edgecolor='k', zorder=11,
                   label=f"θ_true [ℓ={LS_TRUE}, σ²={VAR_TRUE}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}", fontsize=11)
        ax.legend(loc='lower right', fontsize=7)

    fig.suptitle(f"Well-specified GP-drift SDE T={T} (float64) — hyperparameter "
                 f"EM trajectories vs #inducing\n"
                 f"[θ_true=({LS_TRUE}, {VAR_TRUE})]  "
                 f"SparseGP endpoints should march toward the gold GT MLE / "
                 f"blue EFGP endpoint as M grows", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {out_png}", flush=True)


def render_convergence(T_data, out_png):
    """Recovered ℓ, σ², wall vs #inducing (one line per ls_init), with EFGP-final
    and oracle-MLE horizontal reference lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gb_ll = T_data['gb_ll']; gb_lv = T_data['gb_lv']
    results = T_data['results']
    mle_ls = math.exp(gb_ll); mle_var = math.exp(gb_lv)
    n_inds = np.asarray(M_LIST)
    init_colors = {0.7: 'tab:blue', 3.0: 'tab:orange'}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for ls_init in LS_INIT_LIST:
        cell = results.get(ls_init, {})
        c = init_colors.get(ls_init, 'tab:red')
        sp_ls = [cell['sparse'][M]['state']['ls'] for M in M_LIST]
        sp_var = [cell['sparse'][M]['state']['var'] for M in M_LIST]
        sp_wall = [cell['sparse'][M]['state']['wall'] for M in M_LIST]
        e = cell.get('efgp', {}).get('state')

        ax[0].plot(n_inds, sp_ls, '-s', color=c,
                   label=f'SparseGP (init {ls_init})')
        ax[1].plot(n_inds, sp_var, '-s', color=c,
                   label=f'SparseGP (init {ls_init})')
        ax[2].semilogy(n_inds, sp_wall, '-s', color=c,
                       label=f'SparseGP (init {ls_init})')
        if e is not None:
            ax[0].axhline(e['ls'], color=c, ls='--', lw=1.0, alpha=0.8,
                          label=f'EFGP (init {ls_init}) {e["ls"]:.2f}')
            ax[1].axhline(e['var'], color=c, ls='--', lw=1.0, alpha=0.8,
                          label=f'EFGP (init {ls_init}) {e["var"]:.2f}')
            ax[2].axhline(e['wall'], color=c, ls='--', lw=1.0, alpha=0.8)

    ax[0].axhline(mle_ls, color='gold', ls='-', lw=2.0,
                  label=f'oracle MLE ({mle_ls:.2f})')
    ax[0].axhline(LS_TRUE, color='magenta', ls=':', lw=1.5,
                  label=f'θ_true ({LS_TRUE})')
    ax[1].axhline(mle_var, color='gold', ls='-', lw=2.0,
                  label=f'oracle MLE ({mle_var:.2f})')
    ax[1].axhline(VAR_TRUE, color='magenta', ls=':', lw=1.5,
                  label=f'θ_true ({VAR_TRUE})')

    ax[0].set_xlabel('# inducing'); ax[0].set_ylabel(r'recovered $\ell$')
    ax[0].set_title(r'lengthscale $\ell$'); ax[0].legend(fontsize=7)
    ax[1].set_xlabel('# inducing'); ax[1].set_ylabel(r'recovered $\sigma_f^2$')
    ax[1].set_title(r'variance $\sigma_f^2$'); ax[1].legend(fontsize=7)
    ax[2].set_xlabel('# inducing'); ax[2].set_ylabel('wall (s)')
    ax[2].set_title('wall-clock (EFGP = dashed)'); ax[2].legend(fontsize=7)
    fig.suptitle('SparseGP → EFGP / oracle-MLE convergence vs # inducing points '
                 '(well-specified GP drift)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {out_png}", flush=True)


def main():
    print(f"bench_gpdrift_inducing_sweep: T={T} ls_init={LS_INIT_LIST} "
          f"M={M_LIST} θ_true=(ℓ={LS_TRUE}, σ²={VAR_TRUE})  "
          f"x64={jax.config.read('jax_enable_x64')}  devices={jax.devices()}",
          flush=True)

    print("Sampling drift f_true ~ GP(0, RBF(ℓ_true, σ²_true))...", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)
    print(f"  trajectory range: x[0]∈[{xs_np[:,0].min():.2f}, "
          f"{xs_np[:,0].max():.2f}], x[1]∈[{xs_np[:,1].min():.2f}, "
          f"{xs_np[:,1].max():.2f}]", flush=True)

    LOG_LS = np.linspace(LOG_LS_RANGE[0], LOG_LS_RANGE[1], N_GRID)
    LOG_VAR = np.linspace(LOG_VAR_RANGE[0], LOG_VAR_RANGE[1], N_GRID)
    print(f"Computing GT oracle landscape ({N_GRID}×{N_GRID})...", flush=True)
    t0 = time.perf_counter()
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"(true: ℓ={LS_TRUE}, σ²={VAR_TRUE})  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    results = {}
    for ls_init in LS_INIT_LIST:
        print(f"\n--- ls_init = {ls_init} ---", flush=True)
        cell = dict(ls_init=ls_init, sparse={})

        print("  EFGP fit...", flush=True)
        e = fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
        try:
            ed = bench.compute_drift_metrics(
                e['mp'], xs_np,
                f_eval_fn=lambda g: eval_efgp_drift(e['hist'], g),
                drift_fn=drift_fn)
        except Exception as ex:
            print(f"    (EFGP drift metric skipped: {type(ex).__name__}: {ex})",
                  flush=True)
            ed = dict(rel_mse=np.nan, rel_mse_raw=np.nan)
        elr = bench.latent_recovery_rmse(e['mp'], xs_np)
        print(f"    ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
              f"drift_pc={ed['rel_mse']:.4f}, lat_pc={elr['pc']:.4f}, "
              f"wall={e['wall']:.1f}s", flush=True)
        cell['efgp'] = dict(state=e, drift=ed, latent_rmse=elr)

        for M in M_LIST:
            n_per = int(round(math.sqrt(M)))
            print(f"  SP M={M} ({n_per}×{n_per}) fit...", flush=True)
            try:
                s = bench.fit_sparsegp(lik, op, ip, t_grid, sigma, n_per,
                                       ls_init, xs_np)
            except Exception as ex:
                print(f"    EXC (fit): {type(ex).__name__}: {ex}", flush=True)
                nan_state = dict(ls_traj=np.array([np.nan]),
                                 var_traj=np.array([np.nan]),
                                 ls=np.nan, var=np.nan, wall=np.nan)
                cell['sparse'][M] = dict(
                    state=nan_state,
                    drift=dict(rel_mse=np.nan, rel_mse_raw=np.nan),
                    latent_rmse=dict(pc=np.nan, raw=np.nan))
                continue
            try:
                sd = bench.compute_drift_metrics(
                    s['mp'], xs_np,
                    f_eval_fn=lambda g, s_=s: bench.eval_sp_drift(s_, g),
                    drift_fn=drift_fn)
            except Exception as ex:
                print(f"    (SP drift metric skipped: {type(ex).__name__}: {ex})",
                      flush=True)
                sd = dict(rel_mse=np.nan, rel_mse_raw=np.nan)
            slr = bench.latent_recovery_rmse(s['mp'], xs_np)
            print(f"    ℓ={s['ls']:.3f}, σ²={s['var']:.3f}, "
                  f"drift_pc={sd['rel_mse']:.4f}, lat_pc={slr['pc']:.4f}, "
                  f"wall={s['wall']:.1f}s", flush=True)
            cell['sparse'][M] = dict(state=s, drift=sd, latent_rmse=slr)
        results[ls_init] = cell

    T_data = dict(T=T, xs_np=xs_np, t_grid=t_grid, sigma=sigma,
                  LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                  gb_ll=gb_ll, gb_lv=gb_lv, results=results)

    render_landscape(T_data, OUT_DIR / 'landscape_paths.png')
    render_convergence(T_data, OUT_DIR / 'convergence.png')

    # ---- save raw ----
    save_kwargs = dict(T=T, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                       gb_ll=gb_ll, gb_lv=gb_lv, xs_np=xs_np,
                       ls_true=LS_TRUE, var_true=VAR_TRUE,
                       M_list=np.asarray(M_LIST),
                       ls_init_list=np.asarray(LS_INIT_LIST))
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e = cell['efgp']['state']
        save_kwargs[f'lsinit{ls_init}_efgp_ls_traj'] = e['ls_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_var_traj'] = e['var_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_ls'] = e['ls']
        save_kwargs[f'lsinit{ls_init}_efgp_var'] = e['var']
        save_kwargs[f'lsinit{ls_init}_efgp_wall'] = e['wall']
        save_kwargs[f'lsinit{ls_init}_efgp_drift_pc'] = \
            cell['efgp']['drift']['rel_mse']
        for M in M_LIST:
            st = cell['sparse'][M]['state']
            save_kwargs[f'lsinit{ls_init}_sp{M}_ls_traj'] = st['ls_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_var_traj'] = st['var_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_ls'] = st['ls']
            save_kwargs[f'lsinit{ls_init}_sp{M}_var'] = st['var']
            save_kwargs[f'lsinit{ls_init}_sp{M}_wall'] = st['wall']
            save_kwargs[f'lsinit{ls_init}_sp{M}_drift_pc'] = \
                cell['sparse'][M]['drift']['rel_mse']
    np.savez(OUT_DIR / 'sweep.npz', **save_kwargs)
    print(f"  saved {OUT_DIR / 'sweep.npz'}", flush=True)

    # ---- summary table ----
    print(f"\n{'='*90}")
    print(f"SUMMARY — well-specified GP drift T={T}  "
          f"(θ_true: ℓ={LS_TRUE}, σ²={VAR_TRUE}; "
          f"oracle MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f})")
    print(f"{'='*90}")
    print(f"  {'ls_init':>7s}  {'method':>9s}  {'ℓ_final':>8s}  "
          f"{'σ²_final':>9s}  {'drift_pc':>9s}  {'wall':>7s}")
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e = cell['efgp']['state']
        print(f"  {ls_init:>7.1f}  {'EFGP':>9s}  {e['ls']:>8.3f}  "
              f"{e['var']:>9.3f}  {cell['efgp']['drift']['rel_mse']:>9.4f}  "
              f"{e['wall']:>6.1f}s")
        for M in M_LIST:
            st = cell['sparse'][M]['state']
            dp = cell['sparse'][M]['drift']['rel_mse']
            ls_s = f"{st['ls']:>8.3f}" if math.isfinite(st['ls']) else f"{'NaN':>8s}"
            var_s = f"{st['var']:>9.3f}" if math.isfinite(st['var']) else f"{'NaN':>9s}"
            dp_s = f"{dp:>9.4f}" if math.isfinite(dp) else f"{'NaN':>9s}"
            w_s = f"{st['wall']:>6.1f}s" if math.isfinite(st['wall']) else f"{'NaN':>7s}"
            print(f"  {ls_init:>7.1f}  {f'SP M={M}':>9s}  {ls_s}  {var_s}  "
                  f"{dp_s}  {w_s}")


if __name__ == '__main__':
    main()
