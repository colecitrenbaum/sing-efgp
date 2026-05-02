"""
Cross-check: EFGPDrift vs. SING's stock SparseGP on a regime where the
inducing-point method is known to give the right answer.

Setup:
  * 1D latent SDE  (d=2 would slow this test down significantly)
  * Stable linear drift  f(x) = -alpha x  with alpha = 1.5
  * Gaussian observations (so SING's stock SparseGP path applies cleanly)
  * Both methods run for the SAME number of E/M iterations and start from
    matched initial conditions

Assertion: the two methods agree on latent RMSE within a factor of 1.5x —
EFGP must not be dramatically worse than SparseGP in a regime where
SparseGP works.

This test is deliberately *coarse*; the demo notebook contains the full
quantitative side-by-side comparison.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sing importable
_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))
# efgpnd path
_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
if str(_GP_QUAD) not in sys.path:
    sys.path.insert(0, str(_GP_QUAD))

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from sing.efgp_drift import EFGPDrift
from sing.efgp_em import fit_efgp_sing
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs

from kernels import SquaredExponential as EFGPSE
from kernels import GPParams


class _GaussLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


@pytest.mark.slow
def test_efgp_vs_sparsegp_linear_drift():
    """EFGP-SING and SING-SparseGP should give comparable latent RMSE on a
    well-behaved linear-drift problem."""
    D = 2
    T = 60
    sigma = 0.4
    n_em = 6
    n_estep = 6
    n_mstep = 0           # don't update kernel; isolate the inference quality

    # --------------- synthetic data ---------------
    A_true = jnp.array([[-1.5, 0.0], [0.0, -1.5]])
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.2, -0.8]),
                      f=drift_true, t_max=2.0, n_timesteps=T,
                      sigma=sigma_fn)

    N = 6
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N), R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(8), xs, out_true)

    # Shared init for emissions
    ys_centered = ys - ys.mean(axis=0)
    _, _, Vt = jnp.linalg.svd(ys_centered, full_matrices=False)
    output_params_init = dict(C=Vt[:D].T, d=ys.mean(axis=0),
                              R=jnp.full((N,), 0.1))

    init_params = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1)
    )

    t_grid = jnp.linspace(0., 2.0, T)

    # --------------- EFGP path ---------------
    efgp_kernel = EFGPSE(dimension=D, init_lengthscale=0.7, init_variance=1.0)
    efgp_gp = GPParams(kernel=efgp_kernel, init_sig2=1.0)
    drift_efgp = EFGPDrift(kernel=efgp_kernel, gp_params=efgp_gp,
                           latent_dim=D, sigma_drift_sq=sigma ** 2,
                           eps_grid=1e-3, S_marginal=4, J_hutchinson=4)
    lik = _GaussLik(ys[None], jnp.ones((1, T), dtype=bool))
    mp_efgp, _, _, _, hist_efgp = fit_efgp_sing(
        drift=drift_efgp, likelihood=lik, t_grid=t_grid,
        output_params=output_params_init, init_params=init_params,
        sigma=sigma, n_em_iters=n_em, n_estep_iters=n_estep,
        n_mstep_iters=n_mstep,
        rho_sched=jnp.linspace(0.05, 0.5, n_em),
        learn_emissions=True, learn_kernel=False, update_R=False,
        true_xs=np.asarray(xs), verbose=False,
    )
    rmse_efgp = float(np.sqrt(np.mean((np.asarray(mp_efgp['m'][0]) -
                                       np.asarray(xs)) ** 2)))

    # --------------- SparseGP path ---------------
    quadrature = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)            # 64 inducing pts
    rbf = RBF(latent_dim=D)
    sparse_drift = SparseGP(zs=zs, kernel=rbf, expectation=quadrature)
    sparse_drift_params = dict(
        length_scales=jnp.full((D,), 0.7),
        output_scale=jnp.asarray(1.0),
    )
    mp_sp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(9),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=init_params, output_params=output_params_init,
        rho_sched=jnp.linspace(0.05, 0.5, n_em),
        sigma=sigma,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=0,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((n_em,), 0.05),
        print_interval=999,
    )
    rmse_sp = float(np.sqrt(np.mean((np.asarray(mp_sp['m'][0]) -
                                     np.asarray(xs)) ** 2)))

    print(f"\n[efgp_vs_sparsegp] EFGP rmse={rmse_efgp:.3f},  "
          f"SparseGP rmse={rmse_sp:.3f}")

    # Both should be sensible (better than predicting zeros).
    rmse_zero = float(np.sqrt(np.mean(np.asarray(xs) ** 2)))
    assert rmse_efgp < rmse_zero, (
        f"EFGP didn't beat zero baseline: {rmse_efgp:.3f} >= {rmse_zero:.3f}")
    assert rmse_sp < rmse_zero, (
        f"SparseGP didn't beat zero baseline: {rmse_sp:.3f} >= {rmse_zero:.3f}")

    # EFGP within 50% of SparseGP (both ways: EFGP shouldn't be much worse,
    # and we don't expect it to be dramatically better either at this small T).
    ratio = rmse_efgp / rmse_sp
    assert 0.5 < ratio < 1.5, (
        f"EFGP/SparseGP latent-RMSE ratio {ratio:.2f} out of [0.5, 1.5]; "
        f"EFGP={rmse_efgp:.3f}, SparseGP={rmse_sp:.3f}")
