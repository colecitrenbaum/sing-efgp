"""
diag_K_production_sweep.py

Killer experiment: re-run production EFGP fit on Duffing with various K_per_dim
to see if the converged hypers shift toward the GT MLE (ℓ=1.10) as K grows.

The K-bias landscape diagnostic showed K=5 (production default) misrepresents
the small-ℓ region of the loss surface so badly that the M-step minimum lies
at (ℓ=2.65, σ²=4.71) — far from GT (ℓ=1.10, σ²=0.97). At K=12+, the loss min
collapses to (ℓ=0.23, σ²=7.39+) — different but small. The question: does the
production EM at K=24 actually move there?

Run:
  ~/myenv/bin/python demos/diag_K_production_sweep.py
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
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
N_EM = 50
LS_INIT = 0.7
VAR_INIT = 1.0


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

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
    C_true = rng.standard_normal((cfg['N_obs'], D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(cfg['N_obs']),
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


def fit_efgp_with_K(lik, op, ip, t_grid, sigma, K_per_dim, mstep_lr=0.01):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        K_per_dim=K_per_dim,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        ls_traj=np.array([LS_INIT] + list(hist.lengthscale)),
        var_traj=np.array([VAR_INIT] + list(hist.variance)),
        wall=wall,
    )


def main():
    print(f"[diag] EFGP production K sweep on {CFG['name']}", flush=True)
    print(f"[diag] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), n_em={N_EM}",
          flush=True)
    print(f"[diag] GT MLE (from bench_two_obj_duffing): ℓ=1.10, σ²=0.97",
          flush=True)
    print()

    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    K_LIST = [5, 8, 12, 16, 24]
    results = {}

    for K in K_LIST:
        print(f"[diag] K_per_dim={K}...", flush=True)
        r = fit_efgp_with_K(lik, op, ip, t_grid, CFG['sigma'], K_per_dim=K)
        ls_f = float(r['ls_traj'][-1])
        var_f = float(r['var_traj'][-1])
        print(f"    final ℓ={ls_f:.3f}, σ²={var_f:.3f}, wall={r['wall']:.1f}s",
              flush=True)
        results[K] = r

    # Plot trajectories
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for K in K_LIST:
        r = results[K]
        iters = np.arange(len(r['ls_traj']))
        axes[0].plot(iters, r['ls_traj'], '-o', markersize=3, label=f"K={K}")
        axes[1].plot(iters, r['var_traj'], '-o', markersize=3, label=f"K={K}")
        axes[2].plot(np.log(r['ls_traj']), np.log(r['var_traj']),
                       '-o', markersize=3, label=f"K={K}")

    # GT MLE marker
    axes[0].axhline(1.10, color='gold', linestyle='--',
                    label='GT MLE ℓ=1.10', linewidth=2)
    axes[1].axhline(0.97, color='gold', linestyle='--',
                    label='GT MLE σ²=0.97', linewidth=2)
    axes[2].scatter([math.log(1.10)], [math.log(0.97)],
                    marker='*', s=300, color='gold', edgecolor='k',
                    zorder=10, label='GT MLE')

    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel('ℓ')
    axes[0].set_yscale('log'); axes[0].set_title('ℓ trajectory by K')
    axes[0].legend(loc='best', fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_yscale('log'); axes[1].set_title('σ² trajectory by K')
    axes[1].legend(loc='best', fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('log ℓ'); axes[2].set_ylabel('log σ²')
    axes[2].set_title('Trajectory in (log ℓ, log σ²) space')
    axes[2].legend(loc='best', fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle(f"EFGP production EM: K_per_dim sweep on {CFG['name']}\n"
                 f"K=5 is the production default. GT MLE ℓ=1.10, σ²=0.97 (gold).",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/tmp/diag_K_production_sweep.png', dpi=110)
    print('[diag] Saved /tmp/diag_K_production_sweep.png', flush=True)

    np.savez('/tmp/diag_K_production_sweep.npz',
              K_list=np.array(K_LIST),
              **{f'ls_traj_K{K}': results[K]['ls_traj'] for K in K_LIST},
              **{f'var_traj_K{K}': results[K]['var_traj'] for K in K_LIST},
              walls=np.array([results[K]['wall'] for K in K_LIST]))
    print('[diag] Saved /tmp/diag_K_production_sweep.npz', flush=True)

    print('\n[diag] Summary: production-EM converged hypers vs K')
    print(f"  {'K':>4s}  {'ℓ_final':>10s}  {'σ²_final':>10s}  {'wall':>8s}",
          flush=True)
    for K in K_LIST:
        r = results[K]
        print(f"  {K:>4d}  {float(r['ls_traj'][-1]):>10.3f}  "
              f"{float(r['var_traj'][-1]):>10.3f}  {r['wall']:>7.1f}s",
              flush=True)


if __name__ == '__main__':
    main()
