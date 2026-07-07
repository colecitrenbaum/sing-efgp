"""Tests for the analytic (Taylor-envelope NUFFT) q(f) update.

``compute_mu_r_analytic_jax`` evaluates the same Σ_i E_{q(x_i)}[...] integrand
as the MC (``compute_mu_r_jax``) and gmix (``compute_mu_r_gmix_jax``) paths,
but via type-1 NUFFTs with the per-source Gaussian envelope Taylor-expanded in
ΔS_i = S_i − S̄.  It is:

  * exact for homogeneous S (any order) — so it should match the MC reference
    at least as well as a single-seed MC does, and better than gmix (which
    carries an n_sigma stencil-tail-truncation bias);
  * increasingly accurate for heterogeneous S as ``order`` grows.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax_finufft  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd


D = 2
sigma_drift_sq = 0.1


def _grid_and_sources(T, key, *, het=0.0):
    """Smooth q(x) with optionally heterogeneous per-source S."""
    t_grid = jnp.linspace(0., 1., T)
    del_t = t_grid[1:] - t_grid[:-1]
    tt = t_grid[:, None]
    ms = jnp.stack([
        jnp.sin(2 * tt[:, 0]) + 0.3 * jnp.cos(4 * tt[:, 0]),
        jnp.cos(3 * tt[:, 0]) + 0.2 * jnp.sin(5 * tt[:, 0]),
    ], axis=1)
    # base scale ramps from (1-het) to (1+het) across time when het>0
    ramp = 1.0 + het * jnp.linspace(-1.0, 1.0, T)
    base = 0.04 * ramp
    Ss = jnp.stack([
        jnp.array([[b, 0.5 * b], [0.5 * b, 0.7 * b]]) for b in np.array(base)
    ])
    SSs = 0.95 * Ss[:-1]
    X_template = (jnp.linspace(-2., 2., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(0.6, 1.0, X_template, eps=1e-2)
    trial_mask = jnp.ones((1, T), dtype=bool)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms[None], Ss[None], SSs[None], del_t, trial_mask)
    return grid, (m_src, S_src, d_src, C_src, w_src)


def _mc_reference(args, grid, n_seeds=8, S_marg=32):
    m_src, S_src, d_src, C_src, w_src = args
    acc = None
    for s in range(n_seeds):
        mu, _, _ = jpd.compute_mu_r_jax(
            m_src, S_src, d_src, C_src, w_src, grid, jr.PRNGKey(100 + s),
            sigma_drift_sq=sigma_drift_sq, S_marginal=S_marg, D_lat=D, D_out=D)
        acc = mu if acc is None else acc + mu
    return acc / n_seeds


def _relerr(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_analytic_matches_mc_homogeneous():
    """On (near-)homogeneous S, analytic order 1 should be at least as close
    to the MC reference as a single-seed production MC (S_marginal=2)."""
    grid, args = _grid_and_sources(200, jr.PRNGKey(0), het=0.0)
    m_src, S_src, d_src, C_src, w_src = args
    mu_ref = _mc_reference(args, grid)
    mu_mc1, _, _ = jpd.compute_mu_r_jax(
        m_src, S_src, d_src, C_src, w_src, grid, jr.PRNGKey(0),
        sigma_drift_sq=sigma_drift_sq, S_marginal=2, D_lat=D, D_out=D)
    mu_an, _, top = jpd.compute_mu_r_analytic_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D, order=1)
    err_an = _relerr(mu_an, mu_ref)
    err_mc1 = _relerr(mu_mc1, mu_ref)
    print(f"\n  analytic-o1 vs MC_avg = {err_an:.4f}   single-MC vs MC_avg = {err_mc1:.4f}")
    assert top is not None
    assert mu_an.shape == (D, grid.M)
    assert err_an < err_mc1, (
        f"analytic ({err_an:.4f}) not at least as close as single-seed MC "
        f"({err_mc1:.4f}) to the MC reference on homogeneous S")


def test_analytic_order2_improves_on_heterogeneous():
    """With heterogeneous S, order 2 should be no worse (and generally better)
    than order 1, and both should beat pure-homogeneous order 0."""
    grid, args = _grid_and_sources(200, jr.PRNGKey(1), het=0.6)
    m_src, S_src, d_src, C_src, w_src = args
    mu_ref = _mc_reference(args, grid)
    errs = {}
    for o in (0, 1, 2):
        mu, _, _ = jpd.compute_mu_r_analytic_jax(
            m_src, S_src, d_src, C_src, w_src, grid,
            sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D, order=o)
        errs[o] = _relerr(mu, mu_ref)
    print(f"\n  het=0.6  order errors: {errs}")
    assert errs[1] <= errs[0] + 1e-3, "order1 should not be worse than order0"
    assert errs[2] <= errs[1] + 1e-3, "order2 should not be worse than order1"


def test_analytic_qf_and_moments_shapes():
    """The multi-trial wrapper returns correctly-shaped drift moments + top."""
    T, K = 60, 2
    grid, _ = _grid_and_sources(T, jr.PRNGKey(2))
    key = jr.PRNGKey(3)
    ms = 0.3 * jr.normal(key, (K, T, D))
    Ss = jnp.broadcast_to(0.04 * jnp.eye(D), (K, T, D, D))
    SSs = jnp.broadcast_to(0.038 * jnp.eye(D), (K, T - 1, D, D))
    del_t = jnp.full((T - 1,), 1.0 / T)
    tm = jnp.ones((K, T), dtype=bool)
    mu_r, Ef, Eff, Edfdx, top = jpd.qf_and_moments_analytic_jax(
        ms, Ss, SSs, del_t, tm, grid,
        sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D, order=1,
        return_top=True)
    assert mu_r.shape == (D, grid.M)
    assert Ef.shape == (K, T, D)
    assert Eff.shape == (K, T)
    assert Edfdx.shape == (K, T, D, D)
    assert top is not None


if __name__ == "__main__":
    test_analytic_matches_mc_homogeneous()
    test_analytic_order2_improves_on_heterogeneous()
    test_analytic_qf_and_moments_shapes()
    print("\nPassed.")
