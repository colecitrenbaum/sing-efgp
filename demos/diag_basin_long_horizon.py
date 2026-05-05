"""
diag_basin_long_horizon.py

Confirm the two-basin structure on Duffing by running EFGP EM for n_em=200
from FOUR initializations spanning both basins and the boundary between them.

Conditions (all K=8, baseline settings otherwise):
  L1. init=(0.3, 1.0)  — should land in small-ℓ basin (gap_GT < 0.3 expected)
  L2. init=(0.5, 1.0)  — boundary case
  L3. init=(0.7, 1.0)  — production default; expected upper basin
  L4. init=(1.5, 1.0)  — well above boundary; should reach upper attractor

Key check: at n_em=200, do L1 and L2 stay near GT MLE, or do they drift up to
(2.285, 3.420)?  If they stay → real two-basin structure; if they drift →
slow but-eventual single-attractor.

Run:
  ~/myenv/bin/python demos/diag_basin_long_horizon.py
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

import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
N_EM = 200  # long horizon
K_FIXED = 8


CFG = dict(
    name='Duffing double-well',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    T=400, t_max=15.0, sigma=0.2, N_obs=8, seed=13,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg):
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=cfg['t_max'],
                      n_timesteps=cfg['T'], sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_efgp(lik, op, ip, t_grid, sigma, *, ls_init, var_init):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=var_init, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        K_per_dim=K_FIXED,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        ls_traj=np.array([ls_init] + list(hist.lengthscale)),
        var_traj=np.array([var_init] + list(hist.variance)),
        wall=wall,
    )


CONDS = [
    dict(label='L1: init=(0.3, 1.0)', ls=0.3, var=1.0),
    dict(label='L2: init=(0.5, 1.0)', ls=0.5, var=1.0),
    dict(label='L3: init=(0.7, 1.0) [prod default]', ls=0.7, var=1.0),
    dict(label='L4: init=(1.5, 1.0)', ls=1.5, var=1.0),
]


def main():
    print(f"[basin] Long-horizon basin test on {CFG['name']} (K={K_FIXED}, "
          f"N_EM={N_EM})", flush=True)
    print(f"[basin] GT MLE: ℓ=1.10, σ²=0.97   |   "
          f"upper attractor: ℓ=2.285, σ²=3.420", flush=True)
    print()

    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    results = {}
    for c in CONDS:
        print(f"[basin] {c['label']}...", flush=True)
        r = fit_efgp(lik, op, ip, t_grid, CFG['sigma'],
                      ls_init=c['ls'], var_init=c['var'])
        ls_f = float(r['ls_traj'][-1])
        var_f = float(r['var_traj'][-1])
        # Slope estimate over last 20 iters
        last20_ls = r['ls_traj'][-20:]
        last20_var = r['var_traj'][-20:]
        slope_ls = (last20_ls[-1] - last20_ls[0]) / 19
        slope_var = (last20_var[-1] - last20_var[0]) / 19
        gap_gt = math.hypot(math.log(ls_f) - math.log(1.10),
                              math.log(var_f) - math.log(0.97))
        gap_lk = math.hypot(math.log(ls_f) - math.log(2.285),
                              math.log(var_f) - math.log(3.420))
        print(f"    final ℓ={ls_f:.3f}, σ²={var_f:.3f},   "
              f"Δ_GT={gap_gt:.3f}, Δ_lock={gap_lk:.3f},   "
              f"slope ℓ={slope_ls:+.4f}/iter, σ²={slope_var:+.4f}/iter   "
              f"({r['wall']:.0f}s)", flush=True)
        results[c['label']] = r

    # Plot per-condition mini-panels: ℓ(t), σ²(t), trajectory
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nC = len(CONDS)
    fig, axes = plt.subplots(3, nC, figsize=(5 * nC, 11))

    for col, c in enumerate(CONDS):
        r = results[c['label']]
        iters = np.arange(len(r['ls_traj']))

        # row 0: ℓ vs iter
        ax = axes[0, col]
        ax.plot(iters, r['ls_traj'], '-', color='C0', lw=1.5)
        ax.axhline(1.10, color='gold', linestyle='--',
                    label='GT MLE 1.10', linewidth=1.5)
        ax.axhline(2.285, color='red', linestyle=':',
                    label='upper 2.285', linewidth=1.5)
        ax.set_xlabel('EM iter'); ax.set_ylabel('ℓ')
        ax.set_yscale('log')
        ax.set_title(f"{c['label']}\nfinal ℓ={r['ls_traj'][-1]:.3f}")
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

        # row 1: σ² vs iter
        ax = axes[1, col]
        ax.plot(iters, r['var_traj'], '-', color='C1', lw=1.5)
        ax.axhline(0.97, color='gold', linestyle='--',
                    label='GT MLE 0.97', linewidth=1.5)
        ax.axhline(3.420, color='red', linestyle=':',
                    label='upper 3.42', linewidth=1.5)
        ax.set_xlabel('EM iter'); ax.set_ylabel('σ²')
        ax.set_yscale('log')
        ax.set_title(f"final σ²={r['var_traj'][-1]:.3f}")
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)

        # row 2: trajectory in (log ℓ, log σ²)
        ax = axes[2, col]
        ax.plot(np.log(r['ls_traj']), np.log(r['var_traj']),
                 '-o', markersize=2, color='C2', lw=1.0)
        ax.scatter([math.log(c['ls'])], [math.log(c['var'])],
                    marker='+', s=200, color='black', zorder=5,
                    label='init')
        ax.scatter([math.log(1.10)], [math.log(0.97)],
                    marker='*', s=300, color='gold', edgecolor='k',
                    zorder=10, label='GT MLE')
        ax.scatter([math.log(2.285)], [math.log(3.420)],
                    marker='X', s=200, color='red', edgecolor='k',
                    zorder=10, label='upper attractor')
        ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 2.5)
        ax.set_title('(log ℓ, log σ²) path')
        ax.legend(loc='best', fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"Long-horizon basin test on {CFG['name']} "
                 f"(K={K_FIXED}, N_EM={N_EM})\n"
                 f"slope of last-20 iters tells us if traj is still moving",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/tmp/diag_basin_long_horizon.png', dpi=110)
    print('[basin] Saved /tmp/diag_basin_long_horizon.png', flush=True)

    np.savez('/tmp/diag_basin_long_horizon.npz',
              **{f'ls_{c["label"][:2]}': results[c['label']]['ls_traj']
                  for c in CONDS},
              **{f'var_{c["label"][:2]}': results[c['label']]['var_traj']
                  for c in CONDS})

    print('\n[basin] Summary at iter 50, 100, 200:')
    print(f"  {'cond':36s}  {'@50':>20s}  {'@100':>20s}  {'@200':>20s}",
          flush=True)
    for c in CONDS:
        r = results[c['label']]
        s50 = (float(r['ls_traj'][50]), float(r['var_traj'][50]))
        s100 = (float(r['ls_traj'][100]), float(r['var_traj'][100]))
        s200 = (float(r['ls_traj'][-1]), float(r['var_traj'][-1]))
        print(f"  {c['label']:36s}  "
              f"({s50[0]:.3f}, {s50[1]:.3f})    "
              f"({s100[0]:.3f}, {s100[1]:.3f})    "
              f"({s200[0]:.3f}, {s200[1]:.3f})", flush=True)


if __name__ == '__main__':
    main()
