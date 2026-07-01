"""Tests for the gmix-aux + full-Eff helpers in
``sing/efgp_gmix_qx_moments.py``.

The legacy ``gmix_E_moments_at`` / ``LiveGmixEFGPDrift`` per-transition
path was removed (replaced by ``drift_moments_gmix_jax`` +
``FrozenEFGPDrift`` shim — see ``test_efgp_qx_v_hutch.py`` for the
batched-gather correctness tests). What's still tested here:

  1. ``gmix_E_full_Eff`` matches a brute-force MC reference for
     ``E[bar_f^T bar_f]`` (used for ELBO reporting).
  2. ``precompute_aux`` produces the spectral autocorrelation that
     ``gmix_E_full_Eff`` consumes.
  3. End-to-end smoke: ``fit_efgp_sing_jax(qx_moments_method='gmix_batched')``
     and ``='linearised_shim'`` both produce finite marginals on a tiny
     linear-OU recovery problem.
"""
import jax
import jax.numpy as jnp
import numpy as np

import sing.efgp_jax_primitives as jp
from sing.efgp_gmix_qx_moments import gmix_E_full_Eff, precompute_aux


def _build_problem(d=2, n_per_dim=7, seed=0):
    """Tiny synthetic EFGP setup with a Hermitian-symmetric ``mu_r``."""
    X = jnp.array([[-1.0] * d, [1.0] * d])
    grid = jp.spectral_grid_se(0.5, 1.0, X, eps=1e-2)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(d,) + tuple(grid.mtot_per_dim))
    rev = tuple(slice(None, None, -1) for _ in range(d))
    sym = 0.5 * (raw + raw[(slice(None),) + rev])
    mu_r = jnp.asarray(sym.reshape(d, grid.M))
    return grid, mu_r


def _mc_reference(m, S, mu_r, grid, n_samples=80_000, key_seed=42):
    """Brute-force ground truth for E[bar_f^T bar_f]."""
    L = jnp.linalg.cholesky(S + 1e-10 * jnp.eye(S.shape[-1]))
    eps = jax.random.normal(jax.random.PRNGKey(key_seed),
                             (n_samples, S.shape[-1]), dtype=jnp.float64)
    xs = m[None, :] + eps @ L.T
    ws_real = grid.ws.real
    phases = jnp.exp(2j * jnp.pi * xs @ grid.xis_flat.T)
    bf = jnp.einsum('rm,sm,m->sr', mu_r, phases, ws_real).real
    return (bf ** 2).sum(-1).mean()


def test_gmix_E_full_Eff_matches_mc():
    """``gmix_E_full_Eff`` matches brute-force MC for E[bar_f^T bar_f]."""
    grid, mu_r = _build_problem()
    aux = precompute_aux(mu_r, grid)
    m = jnp.array([0.3, -0.2])
    S = jnp.array([[0.1, 0.02], [0.02, 0.08]])
    Eff_full = gmix_E_full_Eff(m, S, mu_r, grid, aux)
    Eff_mc = _mc_reference(m, S, mu_r, grid)
    np.testing.assert_allclose(Eff_full, Eff_mc, atol=2e-2, rtol=0)


def _tiny_fit_setup():
    """Reuse the multitrial test's fixture pattern (K=1 small linear OU)."""
    import jax.random as jr
    from sing.likelihoods import Likelihood
    from sing.simulate_data import simulate_sde, simulate_gaussian_obs

    D, T, sigma = 2, 50, 0.3

    def drift_fn(x, t):
        A = jnp.array([[-1.0, 0.0], [0.0, -1.5]])
        return A @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)

    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.0, -0.5]),
                       f=drift_fn, t_max=2.0, n_timesteps=T,
                       sigma=sigma_fn)
    N = 4
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
    op = dict(C=jnp.asarray(C_true), d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                  dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    return dict(D=D, T=T, sigma=sigma, lik=lik, op=op, ip=ip, xs=xs)


def test_fit_efgp_sing_gmix_batched_runs():
    """End-to-end smoke: gmix_batched (DEFAULT) produces finite marginals
    on a tiny linear OU recovery problem."""
    from sing.efgp_em import fit_efgp_sing_jax
    s = _tiny_fit_setup()
    mp, _, _, _, _ = fit_efgp_sing_jax(
        likelihood=s['lik'], t_grid=jnp.linspace(0., 2.0, s['T']),
        output_params=s['op'], init_params=s['ip'], latent_dim=s['D'],
        lengthscale=0.6, variance=1.0, sigma=s['sigma'],
        sigma_drift_sq=s['sigma'] ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=4, n_estep_iters=6,
        rho_sched=jnp.linspace(0.3, 0.9, 4),
        learn_emissions=False, learn_kernel=False, kernel_warmup_iters=2,
        verbose=False, qx_moments_method='gmix_batched')
    assert jnp.all(jnp.isfinite(mp['m']))
    assert jnp.all(jnp.isfinite(mp['S']))


def test_fit_efgp_sing_legacy_shim_still_works():
    """End-to-end smoke: legacy linearised_shim path still runs."""
    from sing.efgp_em import fit_efgp_sing_jax
    s = _tiny_fit_setup()
    mp, _, _, _, _ = fit_efgp_sing_jax(
        likelihood=s['lik'], t_grid=jnp.linspace(0., 2.0, s['T']),
        output_params=s['op'], init_params=s['ip'], latent_dim=s['D'],
        lengthscale=0.6, variance=1.0, sigma=s['sigma'],
        sigma_drift_sq=s['sigma'] ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=4, n_estep_iters=6,
        rho_sched=jnp.linspace(0.3, 0.9, 4),
        learn_emissions=False, learn_kernel=False, kernel_warmup_iters=2,
        verbose=False, qx_moments_method='linearised_shim')
    assert jnp.all(jnp.isfinite(mp['m']))
