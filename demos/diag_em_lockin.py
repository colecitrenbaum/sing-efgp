"""
diag_em_lockin.py

Test the EM-lockin hypothesis on Duffing: production EM at any K converges to
(ℓ=2.285, σ²=3.420) — neither GT MLE (1.10, 0.97) nor the K=24 frozen-smoother
loss min (0.23, 7.39).  Hypothesis: a smoother-bias positive feedback loop
(over-smooth q(x) ↔ large-ℓ drift) traps EM at a self-consistent attractor.

Test conditions (all at K_per_dim=8 fixed):
  A. Baseline (kernel_warmup_iters=8, S_marginal=2, init=(0.7, 1.0))
  B. Drop warmup: kernel_warmup_iters=0
  C. More MC: S_marginal=8 (E-step + M-step)
  D. High init: init=(3.0, 1.0)
  E. Low init:  init=(0.3, 1.0)
  F. Combine: warmup=0, S=8, low init

GT MLE on Duffing (from bench): ℓ=1.10, σ²=0.97.

Run:
  ~/myenv/bin/python demos/diag_em_lockin.py
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
N_EM = 50
K_FIXED = 8


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
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((cfg['N_obs'],), 0.1))
    op_oracle = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                     R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid, op_oracle


def fit_efgp(lik, op, ip, t_grid, sigma, *,
              ls_init, var_init, kernel_warmup_iters, S_marginal,
              learn_emissions=True, K_per_dim=K_FIXED):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=var_init, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=S_marginal,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=learn_emissions, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=kernel_warmup_iters,
        K_per_dim=K_per_dim,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        ls_traj=np.array([ls_init] + list(hist.lengthscale)),
        var_traj=np.array([var_init] + list(hist.variance)),
        wall=wall,
    )


CONDS = [
    dict(label='A: baseline (warmup=8, S=2, init=0.7,1.0)',
          ls_init=0.7, var_init=1.0, kernel_warmup_iters=8, S_marginal=2),
    dict(label='B: warmup=0',
          ls_init=0.7, var_init=1.0, kernel_warmup_iters=0, S_marginal=2),
    dict(label='C: S=8',
          ls_init=0.7, var_init=1.0, kernel_warmup_iters=8, S_marginal=8),
    dict(label='D: init=(3.0,1.0)',
          ls_init=3.0, var_init=1.0, kernel_warmup_iters=8, S_marginal=2),
    dict(label='E: init=(0.3,1.0)',
          ls_init=0.3, var_init=1.0, kernel_warmup_iters=8, S_marginal=2),
    dict(label='F: warmup=0+S=8+init=(0.3,1.0)',
          ls_init=0.3, var_init=1.0, kernel_warmup_iters=0, S_marginal=8),
]


def main():
    print(f"[lockin] EM lock-in test on {CFG['name']}", flush=True)
    print(f"[lockin] K_per_dim={K_FIXED}, N_EM={N_EM}", flush=True)
    print(f"[lockin] GT MLE: ℓ=1.10, σ²=0.97", flush=True)
    print(f"[lockin] Production attractor (any K): ℓ=2.285, σ²=3.420", flush=True)
    print()

    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    results = {}
    for c in CONDS:
        print(f"[lockin] {c['label']}...", flush=True)
        r = fit_efgp(
            lik, op, ip, t_grid, CFG['sigma'],
            ls_init=c['ls_init'], var_init=c['var_init'],
            kernel_warmup_iters=c['kernel_warmup_iters'],
            S_marginal=c['S_marginal'])
        ls_f = float(r['ls_traj'][-1])
        var_f = float(r['var_traj'][-1])
        gap_to_gt = math.hypot(math.log(ls_f) - math.log(1.10),
                                 math.log(var_f) - math.log(0.97))
        gap_to_lockin = math.hypot(math.log(ls_f) - math.log(2.285),
                                     math.log(var_f) - math.log(3.420))
        print(f"    final ℓ={ls_f:.3f}, σ²={var_f:.3f}, "
              f"gap_to_GT={gap_to_gt:.3f}, gap_to_lockin={gap_to_lockin:.3f}, "
              f"wall={r['wall']:.1f}s",
              flush=True)
        results[c['label']] = r

    # Plot trajectories
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i, c in enumerate(CONDS):
        r = results[c['label']]
        iters = np.arange(len(r['ls_traj']))
        axes[0].plot(iters, r['ls_traj'], '-o', markersize=3,
                       color=colors[i], label=c['label'])
        axes[1].plot(iters, r['var_traj'], '-o', markersize=3,
                       color=colors[i], label=c['label'])
        axes[2].plot(np.log(r['ls_traj']), np.log(r['var_traj']),
                       '-o', markersize=3, color=colors[i], label=c['label'])
        axes[2].plot(math.log(r['ls_traj'][0]),
                       math.log(r['var_traj'][0]), '+', markersize=12,
                       color=colors[i], markeredgewidth=2)

    # GT MLE + lockin markers
    axes[0].axhline(1.10, color='gold', linestyle='--',
                    label='GT MLE ℓ=1.10', linewidth=2)
    axes[0].axhline(2.285, color='red', linestyle=':',
                    label='lockin ℓ=2.285', linewidth=2)
    axes[1].axhline(0.97, color='gold', linestyle='--',
                    label='GT MLE σ²=0.97', linewidth=2)
    axes[1].axhline(3.420, color='red', linestyle=':',
                    label='lockin σ²=3.42', linewidth=2)
    axes[2].scatter([math.log(1.10)], [math.log(0.97)],
                    marker='*', s=300, color='gold', edgecolor='k',
                    zorder=10, label='GT MLE')
    axes[2].scatter([math.log(2.285)], [math.log(3.420)],
                    marker='X', s=200, color='red', edgecolor='k',
                    zorder=10, label='lockin attractor')

    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel('ℓ')
    axes[0].set_yscale('log'); axes[0].set_title('ℓ trajectory')
    axes[0].legend(loc='best', fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_yscale('log'); axes[1].set_title('σ² trajectory')
    axes[1].legend(loc='best', fontsize=7); axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('log ℓ'); axes[2].set_ylabel('log σ²')
    axes[2].set_title('Trajectory in (log ℓ, log σ²) space\n(+ marks init)')
    axes[2].legend(loc='best', fontsize=7); axes[2].grid(alpha=0.3)

    fig.suptitle(f"EFGP EM lock-in test on {CFG['name']} (K={K_FIXED})\n"
                 'GT MLE = gold, lock-in attractor = red X',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('/tmp/diag_em_lockin.png', dpi=110)
    print('[lockin] Saved /tmp/diag_em_lockin.png', flush=True)

    np.savez('/tmp/diag_em_lockin.npz',
              **{f"ls_{c['label'][0]}": results[c['label']]['ls_traj']
                  for c in CONDS},
              **{f"var_{c['label'][0]}": results[c['label']]['var_traj']
                  for c in CONDS})

    print('\n[lockin] Summary:')
    print(f"  {'cond':40s}  {'ℓ_f':>8s}  {'σ²_f':>8s}  "
          f"{'Δ_GT':>8s}  {'Δ_lock':>8s}", flush=True)
    for c in CONDS:
        r = results[c['label']]
        ls_f = float(r['ls_traj'][-1])
        var_f = float(r['var_traj'][-1])
        gap_gt = math.hypot(math.log(ls_f) - math.log(1.10),
                              math.log(var_f) - math.log(0.97))
        gap_lk = math.hypot(math.log(ls_f) - math.log(2.285),
                              math.log(var_f) - math.log(3.420))
        print(f"  {c['label']:40s}  {ls_f:>8.3f}  {var_f:>8.3f}  "
              f"{gap_gt:>8.3f}  {gap_lk:>8.3f}", flush=True)


if __name__ == '__main__':
    main()
