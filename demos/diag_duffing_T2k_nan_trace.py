"""
diag_duffing_T2k_nan_trace.py

Find what specifically NaNs in Duffing T=2000 lr=1e-2 with the full Cholesky
fix. Run a single fit, capture per-iter (ℓ_per_dim, σ², ELBO, ell, KL,
prior). Identify what value goes NaN first.

Run:
  ~/myenv/bin/python demos/diag_duffing_T2k_nan_trace.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
import io
import os as _os
import contextlib

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
    name='Duffing',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    t_max=15.0 * (T_BIG / 400.0), sigma=0.2, N_obs=8, seed=13,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def main():
    sigma_fn = lambda x, t: CFG['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(CFG['seed']), x0=CFG['x0'],
                      f=CFG['drift_fn'], t_max=CFG['t_max'],
                      n_timesteps=T_BIG, sigma=sigma_fn)
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
    t_grid = jnp.linspace(0., CFG['t_max'], T_BIG)
    lik = GLik(ys[None], jnp.ones((1, T_BIG), dtype=bool))

    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []

    print(f"[diag] Duffing T=2000 lr=1e-2 NaN trace, jitter={sparse.jitter}",
          flush=True)
    print(f"[diag] verbose=True to capture ELBO per iter", flush=True)

    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        try:
            fit_variational_em(
                key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
                drift_params=drift_params0, init_params=ip, output_params=op,
                sigma=CFG['sigma'], rho_sched=rho_sched, n_iters=N_EM,
                n_iters_e=10, n_iters_m=4, perform_m_step=True,
                learn_output_params=False,
                learning_rate=jnp.full((N_EM,), float(_os.environ.get('LR', '1e-2'))),
                print_interval=1, drift_params_history=history)
        except Exception as e:
            print(f"EXCEPTION: {e}")

    log_text = log_buf.getvalue()
    print(log_text[-3000:], flush=True)

    # Per-iter ℓ, σ²
    print(f"\n[diag] Per-iter trajectory:", flush=True)
    for t, h in enumerate(history):
        ls = np.asarray(h['length_scales'])
        os_ = float(np.asarray(h['output_scale']))
        nan_h = np.any(np.isnan(ls)) or np.isnan(os_)
        print(f"  iter {t:3d}: ℓ={ls}, σ²={os_**2:.4g}, "
              f"NaN={nan_h}",
              flush=True)
        if nan_h and t > 0:
            print(f"  --> previous iter (sane): ℓ={np.asarray(history[t-1]['length_scales'])}, "
                  f"σ²={float(np.asarray(history[t-1]['output_scale']))**2:.4g}",
                  flush=True)
            break


if __name__ == '__main__':
    main()
