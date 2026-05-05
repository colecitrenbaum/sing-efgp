"""
diag_x64_fix.py

Verify that enabling JAX float64 fixes the NaN cases that
backward-Cholesky-in-float32 was producing.

Test matrix (3 known NaN cases + 1 baseline that worked):
  Duffing T=2000 M=25 ls_init=0.7  — was NaN at float32
  Duffing T=2000 M=25 ls_init=3.0  — was NaN at float32
  Anharmonic T=2000 M=25 ls_init=3.0 — was NaN at float32
  Linear T=2000 M=25 ls_init=0.7   — baseline (was OK at float32)

Run:
  ~/myenv/bin/python demos/diag_x64_fix.py
"""
from __future__ import annotations

# CRITICAL: enable x64 BEFORE any jax import that touches arrays.
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
T = 2000
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
VAR_INIT = 1.0


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

BENCHMARKS = {
    'linear': dict(name='Damped rotation (linear)',
                    drift_fn=lambda x, t: _A_rot @ x,
                    x0=jnp.array([2.0, 0.0]),
                    T_base=400, t_max_base=8.0, sigma=0.3, N_obs=8, seed=7),
    'duffing': dict(name='Duffing double-well',
                     drift_fn=lambda x, t: jnp.stack(
                         [x[1], x[0] - x[0]**3 - 0.5*x[1]]),
                     x0=jnp.array([1.2, 0.0]),
                     T_base=400, t_max_base=15.0, sigma=0.2, N_obs=8, seed=13),
    'anharmonic': dict(name='Anharmonic oscillator',
                        drift_fn=lambda x, t: jnp.stack(
                            [x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3]),
                        x0=jnp.array([1.5, 0.0]),
                        T_base=400, t_max_base=10.0, sigma=0.3, N_obs=8, seed=21),
}


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg, T):
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
    return xs, lik, op, ip, t_grid


def fit_one(short, ls_init, num_per_dim=5):
    cfg = BENCHMARKS[short]
    xs, lik, op, ip, t_grid = make_data(cfg, T)
    sigma = cfg['sigma']
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    try:
        mp, _, _, dp, *_ = fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
            n_iters_m=N_M_INNER, perform_m_step=True,
            learn_output_params=False,
            learning_rate=jnp.full((N_EM,), MSTEP_LR),
            print_interval=999, drift_params_history=history)
        wall = time.perf_counter() - t0
        ls_final = float(jnp.mean(dp['length_scales']))
        var_final = float(dp['output_scale']) ** 2
        nan_iter = None
        for i, h in enumerate(history):
            if not (math.isfinite(float(jnp.mean(h['length_scales'])))
                    and math.isfinite(float(h['output_scale']))):
                nan_iter = i
                break
        return dict(ls=ls_final, var=var_final, nan_iter=nan_iter,
                     wall=wall, history=history)
    except Exception as e:
        wall = time.perf_counter() - t0
        return dict(error=str(e), wall=wall)


def main():
    print(f"diag_x64_fix: testing JAX x64 (enabled={jax.config.read('jax_enable_x64')})")
    print(f"Default float dtype check: {jnp.zeros(1).dtype}\n")

    cases = [
        ('linear',     0.7, 5),  # baseline that worked at float32
        ('duffing',    0.7, 5),  # NaN'd at float32
        ('duffing',    3.0, 5),  # NaN'd at float32
        ('anharmonic', 3.0, 5),  # NaN'd at float32
        ('linear',     0.7, 12), # M=144 mechanism A — also NaN at float32
    ]
    for short, ls_init, n_per_dim in cases:
        M = n_per_dim ** 2
        print(f"=== {short} | ls_init={ls_init} | M={M} ({n_per_dim}x{n_per_dim}) ===",
              flush=True)
        r = fit_one(short, ls_init, num_per_dim=n_per_dim)
        if 'error' in r:
            print(f"  EXC: {r['error'][:80]}  wall={r['wall']:.1f}s",
                  flush=True)
        else:
            ls = r['ls'] if math.isfinite(r['ls']) else float('nan')
            var = r['var'] if math.isfinite(r['var']) else float('nan')
            ni = ('NaN at iter ' + str(r['nan_iter']) if r['nan_iter']
                  is not None else 'no NaN')
            print(f"  final ℓ={ls:.4f}, σ²={var:.4f}, "
                  f"wall={r['wall']:.1f}s  [{ni}]", flush=True)
        print()


if __name__ == '__main__':
    main()
