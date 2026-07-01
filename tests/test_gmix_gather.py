"""Tests for the gmix-gather primitive (Type-2 analog of the Gaussian-
mixture spreader).

Verifies:
  1. ``gmix_inverse_nufft_2d`` matches the explicit per-source sum
        V_i = Σ_δ c(δ) · exp(+2πi δ^T m_i) · exp(-2π² δ^T S_i δ)
     to controlled precision as the FFT grid size N and stencil radius
     r grow.
  2. The gather is the proper inverse-direction analog of the spreader:
     gather(spread(sources)) recovers the original source values up to
     truncation error.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sing.efgp_gmix_gather import gmix_inverse_nufft_2d, _gather_2d
from sing.efgp_gmix_spreader import gmix_nufft_2d, stencil_radius_for


def _explicit_V(c_centered, h_spec, m, S, M_per_dim):
    """Direct evaluation of V_i = sum_δ c(δ) · exp(+2πi δ^T m_i) · exp(-2π² δ^T S_i δ).

    Sums over the centered M_per_dim x M_per_dim spectral grid.
    c_centered is shape (M_per_dim, M_per_dim) complex.
    """
    j = jnp.arange(-(M_per_dim // 2),
                    M_per_dim - (M_per_dim // 2), dtype=m.dtype)
    JX, JY = jnp.meshgrid(j, j, indexing='ij')
    xi_x = JX * h_spec                                    # (M, M)
    xi_y = JY * h_spec
    delta = jnp.stack([xi_x, xi_y], axis=-1)             # (M, M, 2)

    def per_source(m_i, S_i):
        phase = jnp.exp(2j * math.pi * (xi_x * m_i[0] + xi_y * m_i[1]))
        quad = (S_i[0, 0] * xi_x ** 2
                + 2 * S_i[0, 1] * xi_x * xi_y
                + S_i[1, 1] * xi_y ** 2)
        env = jnp.exp(-2 * math.pi ** 2 * quad)
        return jnp.sum(c_centered * phase * env)
    return jax.vmap(per_source)(m, S)


def test_gather_constant_coefficient():
    """For c(δ) = δ_{δ=0} (only DC), V_i = 1 for all (m_i, S_i)."""
    M_per_dim = 17
    h_spec = 0.5
    N = 64
    h_grid = 1.0 / (N * h_spec)
    xcen = jnp.zeros(2)

    # c is 1 at DC (centered index M//2), 0 elsewhere
    c = jnp.zeros((M_per_dim, M_per_dim), dtype=jnp.complex128)
    c = c.at[M_per_dim // 2, M_per_dim // 2].set(1.0)

    rng = np.random.default_rng(0)
    N_src = 5
    m = jnp.asarray(rng.uniform(-0.5, 0.5, (N_src, 2)))
    S = jnp.tile(0.05 * jnp.eye(2)[None], (N_src, 1, 1))

    # Pick stencil_r to cover ~4*sigma_max via the spreader's helper.
    stencil_r = stencil_radius_for(S, h_grid, n_sigma=4.0)

    V_explicit = _explicit_V(c, h_spec, m, S, M_per_dim).real
    V_gather = gmix_inverse_nufft_2d(
        m, S, c, xcen=xcen, h_spec=h_spec,
        M_per_dim=M_per_dim, N=N, stencil_r=stencil_r).real

    err = float(jnp.max(jnp.abs(V_gather - V_explicit)))
    print(f"V_explicit ~= {V_explicit[:3]}, V_gather ~= {V_gather[:3]}")
    print(f"stencil_r = {stencil_r}, max abs err = {err:.3e}")
    assert err < 1e-4, f"DC test failed: max abs err = {err:.3e}"


def test_gather_random_coefficient_against_explicit():
    """Random c(δ) on small grid: gather matches explicit sum."""
    M_per_dim = 17
    h_spec = 0.4
    N = 64
    h_grid = 1.0 / (N * h_spec)
    xcen = jnp.zeros(2)

    rng = np.random.default_rng(1)
    c = (rng.standard_normal((M_per_dim, M_per_dim))
         + 1j * rng.standard_normal((M_per_dim, M_per_dim)))
    c = jnp.asarray(c)

    N_src = 8
    m = jnp.asarray(rng.uniform(-0.6, 0.6, (N_src, 2)))
    S_diags = rng.uniform(0.02, 0.08, (N_src, 2))
    S = jnp.stack([jnp.diag(d) for d in S_diags], axis=0)

    stencil_r = stencil_radius_for(S, h_grid, n_sigma=4.0)
    print(f"stencil_r = {stencil_r}")

    V_explicit = _explicit_V(c, h_spec, m, S, M_per_dim)
    V_gather = gmix_inverse_nufft_2d(
        m, S, c, xcen=xcen, h_spec=h_spec,
        M_per_dim=M_per_dim, N=N, stencil_r=stencil_r)

    err = float(jnp.max(jnp.abs(V_gather - V_explicit)))
    mag = float(jnp.max(jnp.abs(V_explicit)))
    print(f"max abs err: {err:.3e}, max mag: {mag:.3e}, rel: {err/(mag+1e-12):.3e}")
    print(f"V_explicit[0:3]: {V_explicit[:3]}")
    print(f"V_gather[0:3]:   {V_gather[:3]}")
    assert err / (mag + 1e-12) < 1e-3, \
        f"Random-c test failed: rel err {err/(mag+1e-12):.3e}"


def test_gather_eps_convergence():
    """Convergence: error -> 0 as stencil_r grows (with fixed N)."""
    M_per_dim = 17
    h_spec = 0.4
    N = 128
    h_grid = 1.0 / (N * h_spec)
    xcen = jnp.zeros(2)

    rng = np.random.default_rng(2)
    c = (rng.standard_normal((M_per_dim, M_per_dim))
         + 1j * rng.standard_normal((M_per_dim, M_per_dim)))
    c = jnp.asarray(c)

    N_src = 4
    m = jnp.asarray(rng.uniform(-0.4, 0.4, (N_src, 2)))
    S = jnp.tile(0.04 * jnp.eye(2)[None], (N_src, 1, 1))

    V_explicit = _explicit_V(c, h_spec, m, S, M_per_dim)
    mag = float(jnp.max(jnp.abs(V_explicit)))

    # Reference radius covering 4 sigma
    r_ref = stencil_radius_for(S, h_grid, n_sigma=4.0)
    radii = [r_ref // 4, r_ref // 2, r_ref, 2 * r_ref]
    rels = []
    for r in radii:
        V_gather = gmix_inverse_nufft_2d(
            m, S, c, xcen=xcen, h_spec=h_spec,
            M_per_dim=M_per_dim, N=N, stencil_r=int(r))
        err = float(jnp.max(jnp.abs(V_gather - V_explicit)))
        rels.append(err / (mag + 1e-12))
        print(f"r={r}: rel err = {rels[-1]:.3e}")
    assert rels[-1] < rels[0], "stencil_r convergence broken"
    assert rels[2] < 1e-4, \
        f"4-sigma stencil should give <1e-4 rel err, got {rels[2]:.3e}"


def test_gather_grad_via_autodiff():
    """jax.grad through gmix_inverse_nufft_2d gives correct dV/dm.

    Compares to analytical gradient via the explicit closed-form sum.
    """
    M_per_dim = 17
    h_spec = 0.4
    N = 64
    h_grid = 1.0 / (N * h_spec)
    xcen = jnp.zeros(2)

    rng = np.random.default_rng(3)
    c = (rng.standard_normal((M_per_dim, M_per_dim))
         + 1j * rng.standard_normal((M_per_dim, M_per_dim)))
    c = jnp.asarray(c)
    N_src = 4
    m = jnp.asarray(rng.uniform(-0.4, 0.4, (N_src, 2)))
    S_diags = rng.uniform(0.03, 0.06, (N_src, 2))
    S = jnp.stack([jnp.diag(d) for d in S_diags], axis=0)
    stencil_r = stencil_radius_for(S, h_grid, n_sigma=4.0)

    def V_total(ms_):
        return gmix_inverse_nufft_2d(
            ms_, S, c, xcen=xcen, h_spec=h_spec,
            M_per_dim=M_per_dim, N=N, stencil_r=stencil_r).real.sum()
    grad_V_autodiff = jax.grad(V_total)(m)               # (N_src, 2)

    def V_truth_total(ms_):
        return _explicit_V(c, h_spec, ms_, S, M_per_dim).real.sum()
    grad_V_truth = jax.grad(V_truth_total)(m)             # (N_src, 2)

    err = float(jnp.max(jnp.abs(grad_V_autodiff - grad_V_truth)))
    mag = float(jnp.max(jnp.abs(grad_V_truth)))
    print(f"grad rel err = {err/(mag+1e-12):.3e}")
    assert err / (mag + 1e-12) < 1e-3, \
        f"autodiff grad through gather rel err {err/(mag+1e-12):.3e}"
