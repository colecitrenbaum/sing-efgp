"""
bench_dense_M.py

Test whether SparseGP improves with denser inducing grids on the
trajectory.  Lower T (=400) makes the per-iter M-step cheap so we can
push M to 256.  Inducing-point grid `zs_lim=1.8` concentrates on the
Duffing data extent (vs the default 2.5).

Plot all M-step trajectories on the GT log-marginal landscape so we
can see directly how the SparseGP hyperparameter trajectory depends
on M.

Run:
  ~/myenv/bin/python demos/bench_dense_M.py
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
T = 400
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
VAR_INIT = 1.0

LS_INIT_LIST = [0.7, 3.0]
M_LIST = [25, 64, 144, 256]   # 5×5, 8×8, 12×12, 16×16
ZS_LIM = 1.8                  # matches Duffing data extent

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID = 21


CFG = dict(
    name='Duffing double-well',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    T_base=400, t_max_base=15.0, sigma=0.2, N_obs=8, seed=13,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data():
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
    return xs, lik, op, ip, t_grid, sigma


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
        n_hutchinson_mstep=4, kernel_warmup_iters=8, verbose=False)
    return dict(ls_traj=np.array([ls_init] + list(hist.lengthscale)),
                var_traj=np.array([VAR_INIT] + list(hist.variance)),
                ls=float(hist.lengthscale[-1]),
                var=float(hist.variance[-1]),
                wall=time.perf_counter() - t0)


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init,
                   zs_lim):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=zs_lim, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, _, dp, *_ = fit_variational_em(
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
    return dict(ls_traj=ls_traj, var_traj=var_traj,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2,
                wall=wall)


def render(L_gt, LOG_LS, LOG_VAR, gb_ll, gb_lv, results, out_png,
           xs_np):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        levels = list(np.logspace(np.log10(max(lo, hi/1000)),
                                    np.log10(hi), 10))
    else:
        levels = [1, 10, 100]

    n = len(LS_INIT_LIST)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), sharey=True)
    if n == 1:
        axes = [axes]

    cmap_M = {25: 'C3', 64: 'C1', 144: 'C2', 256: 'C4'}

    for col, ls_init in enumerate(LS_INIT_LIST):
        ax = axes[col]
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0]+levels,
                     cmap='viridis_r', extend='max')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                         colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

        cell = results.get(ls_init, {})
        e = cell.get('efgp')
        if e is not None:
            ax.plot(np.log(e['ls_traj']), np.log(e['var_traj']),
                     '-o', color='C0', markersize=3.5, linewidth=2.0,
                     label=f"EFGP [{e['wall']:.0f}s]", zorder=8)
        for M in M_LIST:
            sm = cell.get('sparse', {}).get(M)
            if sm is not None and not np.any(np.isnan(sm['ls_traj'])):
                ax.plot(np.log(sm['ls_traj']), np.log(sm['var_traj']),
                         '-s', color=cmap_M[M], markersize=3,
                         linewidth=1.4, alpha=0.85,
                         label=f"SP M={M} [{sm['wall']:.0f}s]")
        ax.scatter([math.log(ls_init)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=9)
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                    edgecolor='k', zorder=10,
                    label=f"GT MLE [ℓ={math.exp(gb_ll):.2f}, "
                           f"σ²={math.exp(gb_lv):.2f}]")
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_xlabel('log ℓ', fontsize=10)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)
        ax.set_title(f"ls_init = {ls_init}", fontsize=11)
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(f"Duffing T={T}, zs_lim={ZS_LIM} — M-step trajectories "
                  f"per inducing-grid density (float64)",
                  fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"  saved {out_png}", flush=True)


def main():
    print(f"bench_dense_M: T={T} ls_init={LS_INIT_LIST} M={M_LIST} "
          f"zs_lim={ZS_LIM} x64={jax.config.read('jax_enable_x64')}",
          flush=True)

    xs, lik, op, ip, t_grid, sigma = make_data()
    xs_np = np.asarray(xs)
    print(f"  trajectory range: x[0]∈[{xs_np[:,0].min():.2f}, "
          f"{xs_np[:,0].max():.2f}], x[1]∈[{xs_np[:,1].min():.2f}, "
          f"{xs_np[:,1].max():.2f}]", flush=True)

    LOG_LS = np.linspace(LOG_LS_RANGE[0], LOG_LS_RANGE[1], N_GRID)
    LOG_VAR = np.linspace(LOG_VAR_RANGE[0], LOG_VAR_RANGE[1], N_GRID)
    print(f"  computing GT landscape...", flush=True)
    t0 = time.perf_counter()
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    results = {}
    for ls_init in LS_INIT_LIST:
        print(f"\n--- ls_init = {ls_init} ---", flush=True)
        cell = dict(sparse={})

        print(f"  EFGP fit...", flush=True)
        e = fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
        print(f"    final ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
              f"wall={e['wall']:.1f}s", flush=True)
        cell['efgp'] = e

        for M in M_LIST:
            n_per = int(round(math.sqrt(M)))
            print(f"  SP M={M} ({n_per}×{n_per}) at zs_lim={ZS_LIM}...",
                  flush=True)
            try:
                s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per,
                                  ls_init, ZS_LIM)
                ls_str = (f"{s['ls']:.3f}"
                           if math.isfinite(s['ls']) else 'NaN')
                var_str = (f"{s['var']:.3f}"
                            if math.isfinite(s['var']) else 'NaN')
                print(f"    final ℓ={ls_str}, σ²={var_str}, "
                      f"wall={s['wall']:.1f}s", flush=True)
                cell['sparse'][M] = s
            except Exception as ex:
                print(f"    EXC: {type(ex).__name__}: {ex}", flush=True)
                cell['sparse'][M] = dict(ls_traj=np.array([np.nan]),
                                          var_traj=np.array([np.nan]),
                                          ls=np.nan, var=np.nan,
                                          wall=np.nan)
        results[ls_init] = cell

    render(L_gt, LOG_LS, LOG_VAR, gb_ll, gb_lv, results,
           '/tmp/dense_M_landscape.png', xs_np)

    # Save raw data
    save_kwargs = dict(LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                        gb_ll=gb_ll, gb_lv=gb_lv, xs_np=xs_np,
                        T=T, zs_lim=ZS_LIM)
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        save_kwargs[f'lsinit{ls_init}_efgp_ls_traj'] = cell['efgp']['ls_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_var_traj'] = cell['efgp']['var_traj']
        save_kwargs[f'lsinit{ls_init}_efgp_wall'] = cell['efgp']['wall']
        for M in M_LIST:
            sm = cell['sparse'][M]
            save_kwargs[f'lsinit{ls_init}_sp{M}_ls_traj'] = sm['ls_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_var_traj'] = sm['var_traj']
            save_kwargs[f'lsinit{ls_init}_sp{M}_wall'] = sm['wall']
    np.savez('/tmp/dense_M_data.npz', **save_kwargs)
    print(f"  saved /tmp/dense_M_data.npz", flush=True)

    # Summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY — Duffing T={T} zs_lim={ZS_LIM}")
    print(f"{'='*100}")
    print(f"  {'ls_init':>7s}  {'method':>9s}  {'ℓ_final':>8s}  "
          f"{'σ²_final':>9s}  {'wall':>7s}")
    for ls_init in LS_INIT_LIST:
        cell = results[ls_init]
        e = cell['efgp']
        print(f"  {ls_init:>7.1f}  {'EFGP':>9s}  {e['ls']:>8.3f}  "
              f"{e['var']:>9.3f}  {e['wall']:>6.1f}s")
        for M in M_LIST:
            sm = cell['sparse'][M]
            ls_s = (f"{sm['ls']:>8.3f}"
                     if math.isfinite(sm['ls']) else f"{'NaN':>8s}")
            var_s = (f"{sm['var']:>9.3f}"
                      if math.isfinite(sm['var']) else f"{'NaN':>9s}")
            wall_s = (f"{sm['wall']:>6.1f}s"
                       if math.isfinite(sm['wall']) else f"{'NaN':>7s}")
            print(f"  {ls_init:>7.1f}  {f'SP M={M}':>9s}  {ls_s}  "
                  f"{var_s}  {wall_s}")


if __name__ == '__main__':
    main()
