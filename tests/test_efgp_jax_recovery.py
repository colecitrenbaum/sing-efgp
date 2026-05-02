"""
Hyperparameter recovery test for the JAX EM path.

Synthesises an SDE whose drift is a fresh sample from the SE-kernel GP
prior at known ``ls_true``, simulates a trajectory + noisy obs, then runs
``fit_efgp_sing_jax`` from an *intentionally wrong* initial lengthscale
and checks that the M-step pulls it toward the truth.

This is a sanity check that the kernel M-step has the right sign and
the cold-start collapse fix (``kernel_warmup_iters``) does its job.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

# Load jax_finufft FIRST via efgp_em (which imports efgp_jax_primitives).
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import jax
import jax.numpy as jnp
import jax.random as jr
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


def _gp_drift_factory(ls_true: float, var_true: float, D: int, key,
                       extent: float = 4.0, eps_grid: float = 1e-2):
    """Sample a fresh d-dim drift from the SE-kernel GP prior.

    Returns a JAX function ``drift(x, t) -> R^D`` (D independent GP draws,
    one per output dim).
    """
    # Build the spectral grid for a representative cloud spanning [-extent, extent]
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template,
                                eps=eps_grid)

    # Sample epsilon ~ CN(0, I) of shape (D, M) — one draw per output dim
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)                                     # (D, M)
    fk_draw = grid.ws[None, :] * eps               # (D, M) Fourier coefs

    def drift_x(x):
        """x: (D,) -> R^D."""
        x_b = x[None, :]                            # (1, D)
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    def drift(x, t):
        return drift_x(x)

    return drift, fk_draw, grid


@pytest.mark.slow
def test_lengthscale_recovery_jax():
    D = 2
    T = 80
    sigma = 0.4
    ls_true = 0.6
    var_true = 1.0
    n_em = 8

    # Sample the GP drift
    drift_fn, fk_draw, grid_true = _gp_drift_factory(
        ls_true, var_true, D, jr.PRNGKey(123))
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([0.0, 0.0]),
                       f=drift_fn, t_max=2.0, n_timesteps=T,
                       sigma=sigma_fn)
    # Bound observations to a reasonable range (rough sanity)
    xs = jnp.clip(xs, -3.0, 3.0)

    # Gaussian observations
    N = 6
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                     R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(8), xs, out_true)

    class GLik(Likelihood):
        def ell(self, y, m, v, op):
            R = op['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - m) ** 2 + v) / R)

    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                  dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    ls_init = 1.5  # deliberately too large
    var_init = 1.0
    _, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., 2.0, T),
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=var_init, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10,
        rho_sched=jnp.linspace(0.3, 0.9, n_em),
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        kernel_warmup_iters=2,
        verbose=True,
    )

    final_ls = hist.lengthscale[-1]
    print(f"\n  ls_true={ls_true}, ls_init={ls_init}, final_ls={final_ls}")

    # Assertion: M-step moves ls in the RIGHT DIRECTION (toward smaller),
    # not the wrong direction (it doesn't blow up to infinity).
    assert final_ls < ls_init, (
        f"M-step did not pull ls toward the truth: "
        f"started at {ls_init}, ended at {final_ls}, true {ls_true}")
    assert final_ls > 0.1, (
        f"M-step collapsed to ls={final_ls} — cold-start guard failed")
