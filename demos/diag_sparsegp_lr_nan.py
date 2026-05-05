"""
diag_sparsegp_lr_nan.py

Find where SparseGP NaNs out as LR is increased toward EFGP-comparable rates,
and identify the failing variable.

For each LR in {3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}, run the production fit
with oracle C frozen and per-iter logging of (ℓ, σ²).  Stop at first NaN and
print which param went bad.

Run:
  ~/myenv/bin/python demos/diag_sparsegp_lr_nan.py
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
N_EM = 50
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
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_with_lr(lik, op, ip, t_grid, sigma, lr):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    try:
        mp, nat_p, *_ = fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
            n_iters_m=4, perform_m_step=True, learn_output_params=False,
            learning_rate=jnp.full((N_EM,), lr),
            print_interval=999, drift_params_history=history)
        wall = time.perf_counter() - t0
        ls_traj = np.array([LS_INIT] +
                            [float(np.mean(np.asarray(d['length_scales'])))
                             for d in history])
        var_traj = np.array([VAR_INIT] +
                             [float(np.asarray(d['output_scale'])) ** 2
                              for d in history])
        # Per-component length scales
        ls_per_comp = np.stack([np.asarray(d['length_scales'])
                                  for d in history])  # (N_EM, D)
        return dict(ok=True, ls_traj=ls_traj, var_traj=var_traj,
                     ls_per=ls_per_comp, wall=wall)
    except Exception as e:
        return dict(ok=False, err=str(e), history=history,
                     wall=time.perf_counter() - t0)


def main():
    print(f"[diag] SparseGP LR NaN sweep on {CFG['name']} (oracle C, frozen)",
          flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    LRS = [3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    results = {}
    for lr in LRS:
        print(f"\n[diag] lr={lr:.0e}...", flush=True)
        r = fit_with_lr(lik, op, ip, t_grid, CFG['sigma'], lr)
        if r['ok']:
            ls_f = float(r['ls_traj'][-1])
            var_f = float(r['var_traj'][-1])
            has_nan = (np.any(np.isnan(r['ls_traj']))
                        or np.any(np.isnan(r['var_traj'])))
            first_nan = -1
            if has_nan:
                for t in range(len(r['ls_traj'])):
                    if np.isnan(r['ls_traj'][t]) or np.isnan(r['var_traj'][t]):
                        first_nan = t
                        break
            min_ls = float(np.nanmin(r['ls_per']))
            min_var = float(np.nanmin(r['var_traj']))
            print(f"  final ℓ={ls_f:.3f}, σ²={var_f:.3f},  "
                  f"min ℓ_per={min_ls:.4f}, min σ²={min_var:.4f}",
                  flush=True)
            print(f"  has_nan={has_nan} (first at iter {first_nan if has_nan else 'n/a'})",
                  flush=True)
            results[lr] = r
        else:
            print(f"  EXCEPTION: {r['err'][:200]}", flush=True)
            results[lr] = r

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i, lr in enumerate(LRS):
        r = results[lr]
        if not r['ok']:
            continue
        iters = np.arange(len(r['ls_traj']))
        axes[0].plot(iters, r['ls_traj'], '-o', color=colors[i],
                       markersize=2, label=f"lr={lr:.0e}")
        axes[1].plot(iters, r['var_traj'], '-o', color=colors[i],
                       markersize=2, label=f"lr={lr:.0e}")
        axes[2].plot(iters[1:], np.min(r['ls_per'], axis=1),
                       '-o', color=colors[i], markersize=2,
                       label=f"lr={lr:.0e}: min_ℓ_dim")
    for ax in axes[:3]:
        ax.set_yscale('symlog', linthresh=1e-3)
    axes[0].axhline(1.10, color='gold', linestyle='--',
                    label='GT MLE 1.10', linewidth=1.5)
    axes[1].axhline(0.97, color='gold', linestyle='--',
                    label='GT MLE 0.97', linewidth=1.5)
    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel('ℓ (mean)')
    axes[0].set_title('mean ℓ across dims')
    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_title('output variance')
    axes[2].set_xlabel('EM iter'); axes[2].set_ylabel('min ℓ across dims')
    axes[2].set_title('per-dim minimum ℓ\n(NaN onset typically follows ℓ→0)')
    axes[2].axhline(0.0, color='red', linestyle=':',
                    label='ℓ=0 (NaN risk)', linewidth=1.5)
    for ax in axes:
        ax.legend(loc='best', fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"SparseGP LR sweep on {CFG['name']} (oracle C frozen)\n"
                 'Identifies where NaN sets in and which variable fails first',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('/tmp/diag_sparsegp_lr_nan.png', dpi=110)
    print('\n[diag] Saved /tmp/diag_sparsegp_lr_nan.png', flush=True)

    print('\n[diag] Summary:')
    print(f"  {'lr':>10s}  {'final ℓ':>10s}  {'final σ²':>10s}  "
          f"{'min ℓ_dim':>12s}  {'NaN':>5s}", flush=True)
    for lr in LRS:
        r = results[lr]
        if not r['ok']:
            print(f"  {lr:>10.0e}  EXCEPTION", flush=True)
            continue
        nan = (np.any(np.isnan(r['ls_traj']))
                or np.any(np.isnan(r['var_traj'])))
        min_ls = float(np.nanmin(r['ls_per']))
        print(f"  {lr:>10.0e}  {r['ls_traj'][-1]:>10.3f}  "
              f"{r['var_traj'][-1]:>10.3f}  {min_ls:>12.5f}  "
              f"{'YES' if nan else 'no':>5s}", flush=True)


if __name__ == '__main__':
    main()
