"""
diag_inv_kzz_nan.py

Trace why SparseGP NaNs at T=2000 lr=2e-2. Hypothesis under test: the
inv(Kzz)-based gradient blows up.

Method: run 1 EM iter at lr=2e-2, T=2000. After each M-step Adam step,
compute and report:
  - drift_params (length_scales, output_scale)
  - cond(Kzz)
  - || grad_loss_wrt_log_ls ||, || grad_loss_wrt_log_var ||
  - max abs val of inv(Kzz)
  - the loss itself

Identify the iter at which something NaNs and print the immediately-prior state.

Run:
  ~/myenv/bin/python demos/diag_inv_kzz_nan.py
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
from jax import vmap

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params
import optax


D = 2
T_BIG = 2000
LS_INIT = 0.7
VAR_INIT = 1.0


CFG = dict(
    name='Anharmonic',
    drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3]),
    x0=jnp.array([1.5, 0.0]),
    t_max=10.0 * (T_BIG / 400.0), sigma=0.3, N_obs=8, seed=21,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def main():
    print(f"[diag] inv(Kzz) NaN trace, T={T_BIG}", flush=True)

    # Build data
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

    # SparseGP
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)

    # Initial drift params
    dp = dict(length_scales=jnp.full((D,), float(LS_INIT)),
              output_scale=jnp.asarray(math.sqrt(VAR_INIT)))

    # Initialize natural params + smoother (running 1 E-step quickly)
    print(f"[diag] Initializing smoother...", flush=True)
    from sing.sing import _build_initial_natural_params, _e_step
    # Just use the zero-initialization hack from fit_variational_em logic
    nat_p = jax.tree_util.tree_map(
        lambda x: x[None], dict(
            J11=jnp.zeros((T_BIG, D, D)),
            J22=jnp.zeros((T_BIG, D, D)),
            J12=jnp.zeros((T_BIG, D, D)),
            h1=jnp.zeros((T_BIG, D)),
            h2=jnp.zeros((T_BIG, D)),
        )) if False else None
    # Skip detailed init — just run with default rho_sched briefly
    from sing.sing import fit_variational_em
    history = []

    def run_lr(lr, n_iters=15):
        history.clear()
        try:
            mp, nat_p, gp_post, dp_final, *_ = fit_variational_em(
                key=jr.PRNGKey(33), fn=sparse, likelihood=lik,
                t_grid=t_grid, drift_params=dp, init_params=ip,
                output_params=op, sigma=CFG['sigma'],
                rho_sched=jnp.linspace(0.05, 0.7, n_iters),
                n_iters=n_iters, n_iters_e=10, n_iters_m=4,
                perform_m_step=True, learn_output_params=False,
                learning_rate=jnp.full((n_iters,), lr),
                clip_norm=None,
                print_interval=999,
                drift_params_history=history)
            return dp_final, gp_post, mp, nat_p
        except Exception as e:
            return None, None, None, str(e)

    # Run at lr=2e-2 to get to NaN, capture states from history
    print(f"[diag] Running at lr=2e-2 to find NaN onset...", flush=True)
    dp_final, gp_post, mp, nat_p = run_lr(2e-2, n_iters=15)
    if isinstance(nat_p, str):
        print(f"  exception: {nat_p[:200]}", flush=True)
    else:
        print(f"  lr=2e-2 final: ℓ={dp_final['length_scales']}, "
              f"σ²={float(dp_final['output_scale'])**2:.4g}", flush=True)
    # Inspect history
    for t, h in enumerate(history):
        ls = np.asarray(h['length_scales'])
        os_ = float(np.asarray(h['output_scale']))
        nan_ls = np.any(np.isnan(ls))
        nan_os = np.isnan(os_)
        print(f"  iter {t:2d}: ℓ={ls}, output_scale={os_:.4g}, "
              f"NaN={nan_ls or nan_os}", flush=True)

    # Now: at the iter right BEFORE NaN, manually compute Kzz, cond,
    # and gradient terms.
    print(f"\n[diag] Inspecting historical states for Kzz conditioning...",
          flush=True)
    for t, h in enumerate(history):
        if np.any(np.isnan(np.asarray(h['length_scales']))):
            break
        dp_t = dict(length_scales=h['length_scales'],
                    output_scale=h['output_scale'])
        Kzz = vmap(vmap(
            lambda z1, z2: sparse.kernel.K(z1, z2, kernel_params=dp_t),
            (None, 0)), (0, None))(zs, zs) + sparse.jitter * jnp.eye(len(zs))
        cond = float(jnp.linalg.cond(Kzz))
        eig_min = float(jnp.linalg.eigvalsh(Kzz).min())
        eig_max = float(jnp.linalg.eigvalsh(Kzz).max())
        Kzz_inv = float(jnp.abs(jnp.linalg.inv(Kzz)).max())
        ls_str = ', '.join(f'{x:.4g}' for x in np.asarray(h['length_scales']))
        print(f"  iter {t:2d}: ℓ=[{ls_str}], σ²={float(h['output_scale'])**2:.4g}, "
              f"  cond(Kzz)={cond:.2e}, eig_min={eig_min:.2e}, "
              f"  eig_max={eig_max:.2e}, max|inv(Kzz)|={Kzz_inv:.2e}",
              flush=True)


if __name__ == '__main__':
    main()
