"""
diag_sparsegp_T2k_lr.py

LR + clip_norm sweep on Duffing T=2000 (the tightest case) with oracle C
frozen. Find the most aggressive (lr, clip) combination that converges
without NaN. Goal: a robust SparseGP config for T=2000 head-to-head.

Run:
  ~/myenv/bin/python demos/diag_sparsegp_T2k_lr.py
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
T_BIG = 2000


CFG = dict(
    name='Duffing double-well',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    t_max=15.0 * (T_BIG / 400.0), sigma=0.2, N_obs=8, seed=13,
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
                      n_timesteps=T_BIG, sigma=sigma_fn)
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
    t_grid = jnp.linspace(0., cfg['t_max'], T_BIG)
    lik = GLik(ys[None], jnp.ones((1, T_BIG), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_with(lik, op, ip, t_grid, sigma, lr, clip):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=4, perform_m_step=True, learn_output_params=False,
        learning_rate=jnp.full((N_EM,), lr),
        clip_norm=clip,
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls = np.array([float(np.mean(h['length_scales'])) for h in history])
    var = np.array([float(h['output_scale'])**2 for h in history])
    return dict(ls=ls, var=var, wall=wall)


def main():
    print(f"[diag] SparseGP LR+clip sweep on Duffing T={T_BIG} (oracle C)",
          flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    # 4 LRs × 4 clips
    LRS = [3e-3, 5e-3, 1e-2, 2e-2]
    CLIPS = [None, 1.0, 0.1, 0.01]
    results = {}
    for lr in LRS:
        for clip in CLIPS:
            tag = f"lr={lr:.0e}, clip={clip}"
            print(f"\n[diag] {tag}...", flush=True)
            r = fit_with(lik, op, ip, t_grid, CFG['sigma'], lr, clip)
            ls_f = float(r['ls'][-1])
            var_f = float(r['var'][-1])
            has_nan = (np.any(np.isnan(r['ls']))
                        or np.any(np.isnan(r['var'])))
            first_nan = -1
            if has_nan:
                for t in range(len(r['ls'])):
                    if np.isnan(r['ls'][t]) or np.isnan(r['var'][t]):
                        first_nan = t
                        break
            print(f"  final ℓ={ls_f:.4g}, σ²={var_f:.4g}, "
                  f"NaN={'YES@'+str(first_nan) if has_nan else 'no'}, "
                  f"wall={r['wall']:.0f}s",
                  flush=True)
            results[tag] = (r, has_nan, ls_f, var_f, first_nan)

    print("\n[diag] Summary (looking for STABLE + closest to GT MLE ℓ≈0.93):",
          flush=True)
    print(f"  {'config':25s}  {'ℓ_f':>10s}  {'σ²_f':>10s}  {'NaN':>10s}",
          flush=True)
    for tag, (r, has_nan, ls_f, var_f, first_nan) in results.items():
        print(f"  {tag:25s}  {ls_f:>10.4g}  {var_f:>10.4g}  "
              f"{'YES@'+str(first_nan) if has_nan else 'no':>10s}",
              flush=True)


if __name__ == '__main__':
    main()
