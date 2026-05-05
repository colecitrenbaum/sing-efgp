"""
diag_emissions_oracle.py

Test whether the EM lock-in at (ℓ=2.285, σ²=3.420) on Duffing is partly an
emissions-learning artifact.  Compare:

  G. SVD-init C (data-derived) + learn_emissions=False (frozen at SVD)
  H. Oracle C = C_true (synthetic ground truth) + learn_emissions=False

If H lands near GT MLE (ℓ=1.10, σ²=0.97), then emissions learning is part
of the lock-in loop.  If H also locks in at (2.285, 3.420), the bias is
purely on the drift side.

Run:
  ~/myenv/bin/python demos/diag_emissions_oracle.py
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
LS_INIT = 0.7
VAR_INIT = 1.0


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
    op_svd = dict(C=vt[:D].T, d=ys.mean(0),
                   R=jnp.full((cfg['N_obs'],), 0.1))
    op_oracle = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                       R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op_svd, op_oracle, ip, t_grid


def fit_efgp(lik, op, ip, t_grid, sigma, *, learn_emissions):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, op_final, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=learn_emissions, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        K_per_dim=K_FIXED,
        verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        ls_traj=np.array([LS_INIT] + list(hist.lengthscale)),
        var_traj=np.array([VAR_INIT] + list(hist.variance)),
        op_final=op_final, wall=wall,
    )


CONDS = [
    dict(label='G: SVD init C, frozen', op_key='svd', learn=False),
    dict(label='H: oracle C=C_true, frozen', op_key='oracle', learn=False),
    dict(label='I: SVD init C, learned (baseline)', op_key='svd', learn=True),
    dict(label='J: oracle C=C_true, learned', op_key='oracle', learn=True),
]


def main():
    print(f"[oracle] Emissions oracle test on {CFG['name']} (K={K_FIXED})",
          flush=True)
    print(f"[oracle] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), N_EM={N_EM}",
          flush=True)
    print(f"[oracle] GT MLE: ℓ=1.10, σ²=0.97   |   "
          f"baseline lockin: ℓ=2.285, σ²=3.420", flush=True)
    print()

    xs, ys, lik, op_svd, op_oracle, ip, t_grid = make_data(CFG)

    print(f"[oracle] || op_svd C - op_oracle C ||_F / ||C||_F = "
          f"{float(jnp.linalg.norm(op_svd['C'] - op_oracle['C'])) / float(jnp.linalg.norm(op_oracle['C'])):.3f}",
          flush=True)
    print()

    results = {}
    for c in CONDS:
        print(f"[oracle] {c['label']}...", flush=True)
        op = op_svd if c['op_key'] == 'svd' else op_oracle
        r = fit_efgp(lik, op, ip, t_grid, CFG['sigma'],
                      learn_emissions=c['learn'])
        ls_f = float(r['ls_traj'][-1])
        var_f = float(r['var_traj'][-1])
        gap_gt = math.hypot(math.log(ls_f) - math.log(1.10),
                              math.log(var_f) - math.log(0.97))
        gap_lk = math.hypot(math.log(ls_f) - math.log(2.285),
                              math.log(var_f) - math.log(3.420))
        print(f"    final ℓ={ls_f:.3f}, σ²={var_f:.3f}, "
              f"Δ_GT={gap_gt:.3f}, Δ_lock={gap_lk:.3f}, "
              f"wall={r['wall']:.1f}s",
              flush=True)
        results[c['label']] = r

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    colors = ['C0', 'C1', 'C2', 'C3']
    for i, c in enumerate(CONDS):
        r = results[c['label']]
        iters = np.arange(len(r['ls_traj']))
        axes[0].plot(iters, r['ls_traj'], '-o', markersize=3,
                       color=colors[i], label=c['label'])
        axes[1].plot(iters, r['var_traj'], '-o', markersize=3,
                       color=colors[i], label=c['label'])
        axes[2].plot(np.log(r['ls_traj']), np.log(r['var_traj']),
                       '-o', markersize=3, color=colors[i], label=c['label'])

    for ax in axes[:2]:
        ax.set_yscale('log')
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
    axes[0].set_title('ℓ trajectory')
    axes[0].legend(loc='best', fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_title('σ² trajectory')
    axes[1].legend(loc='best', fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('log ℓ'); axes[2].set_ylabel('log σ²')
    axes[2].set_title('Trajectory in (log ℓ, log σ²) space')
    axes[2].legend(loc='best', fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle(f"EFGP emissions-oracle test on {CFG['name']} (K={K_FIXED})",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('/tmp/diag_emissions_oracle.png', dpi=110)
    print('[oracle] Saved /tmp/diag_emissions_oracle.png', flush=True)

    print('\n[oracle] Summary:')
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
