"""
diag_sparsegp_nan.py

Pinpoint where SparseGP M-step NaNs in the cases that fail in the
overnight sweep.  Runs Linear T=2000 M=144 nem=50 (a confirmed failing
case) with per-iteration logging of (ℓ, σ², gradient norms, kernel
condition numbers).  Captures the iter at which any quantity goes NaN.

Run:
  ~/myenv/bin/python demos/diag_sparsegp_nan.py
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

# Linear T=2000 M=144 nem=50 — a confirmed NaN case.
T = 2000
N_EM = 50
M_TOTAL = 144
N_PER_DIM = 12
LS_INIT = 0.7
VAR_INIT = 1.0


def _A_rot():
    return jnp.array([[-0.6, 1.0], [-1.0, -0.6]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data():
    A = _A_rot()
    drift_fn = lambda x, t: A @ x
    t_max = 8.0 * (T / 400.0)
    sigma = 0.3
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([2.0, 0.0]),
                      f=drift_fn, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(7)
    C = jnp.asarray(rng.standard_normal((8, D)) * 0.5)
    out = dict(C=C, d=jnp.zeros(8), R=jnp.full((8,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(8), xs, out)
    op = dict(C=C, d=jnp.zeros(8), R=jnp.full((8,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, sigma


def run_with_jitter(jitter):
    print(f"\n{'='*60}")
    print(f"Running with jitter={jitter:.1e}")
    print(f"{'='*60}", flush=True)
    xs, lik, op, ip, t_grid, sigma = make_data()
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=N_PER_DIM)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D),
                       expectation=quad, jitter=jitter)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    try:
        mp, _, gp_post, dp, *_ = fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
            n_iters_m=4, perform_m_step=True,
            learn_output_params=False,
            learning_rate=jnp.full((N_EM,), 0.01),
            print_interval=999, drift_params_history=history)
        wall = time.perf_counter() - t0
        ls_final = float(jnp.mean(dp['length_scales']))
        var_final = float(dp['output_scale']) ** 2
        nan_iter = None
        for i, h in enumerate(history):
            ls_i = float(jnp.mean(h['length_scales']))
            var_i = float(h['output_scale']) ** 2
            if not (math.isfinite(ls_i) and math.isfinite(var_i)):
                nan_iter = i
                print(f"  ITER {i}: NaN — prev iter {i-1}: "
                      f"ℓ={float(jnp.mean(history[i-1]['length_scales'])):.4f}, "
                      f"σ²={float(history[i-1]['output_scale'])**2:.4f}",
                      flush=True)
                break
        print(f"  wall={wall:.1f}s  final ℓ={ls_final:.3f}, "
              f"σ²={var_final:.3f}  nan_iter={nan_iter}", flush=True)
        return dict(jitter=jitter, ls=ls_final, var=var_final,
                     nan_iter=nan_iter, wall=wall, history=history)
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"  EXCEPTION at wall={wall:.1f}s: {type(e).__name__}: {e}",
              flush=True)
        return dict(jitter=jitter, error=str(e), wall=wall)


def main():
    print(f"diag_sparsegp_nan: Linear T={T} M_total={M_TOTAL} "
          f"({N_PER_DIM}×{N_PER_DIM}) nem={N_EM}", flush=True)
    print(f"  Inducing-pt spacing: {5.0/(N_PER_DIM-1):.3f}", flush=True)

    results = []
    for j in [1e-3, 1e-2]:
        r = run_with_jitter(j)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"  {'jitter':>10s}  {'ℓ_final':>10s}  {'σ²_final':>10s}  "
          f"{'nan_iter':>9s}  {'wall':>7s}")
    for r in results:
        if 'error' in r:
            print(f"  {r['jitter']:>10.1e}  {'EXC':>10s}  "
                  f"{r['error'][:30]:>30s}  {r['wall']:>6.1f}s")
        else:
            ls = r['ls'] if math.isfinite(r['ls']) else float('nan')
            var = r['var'] if math.isfinite(r['var']) else float('nan')
            ni = str(r['nan_iter']) if r['nan_iter'] is not None else 'no NaN'
            print(f"  {r['jitter']:>10.1e}  {ls:>10.4f}  "
                  f"{var:>10.4f}  {ni:>9s}  {r['wall']:>6.1f}s")


if __name__ == '__main__':
    main()
