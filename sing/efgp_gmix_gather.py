"""Gaussian-mixture gather (Type-2 analog of efgp_gmix_spreader).

Implements the closed-form per-source evaluation

    V_i = sum_delta c(delta) * exp(2pi i delta^T m_i)
                              * exp(-2 pi^2 delta^T S_i delta)

where c(delta) lives on a uniform spectral-difference grid and (m_i, S_i)
are per-source mean/covariance (heterogeneous-S).

Strategy (mirrors the spreader's "spread + FFT, no deconvolution"):

    1. Inverse-FFT c(delta) once to a uniform spatial grid c̃(x).
    2. For each source i, gather c̃ on a (2r+1)^D stencil around m_i
       weighted by N(x; m_i, S_i) · h_grid^D, giving a Riemann-sum
       approximation to the continuous integral.

The gather replaces the per-source NUFFT-2 evaluation that would
otherwise require a per-source coefficient grid (envelope folded
in). Cost per call: 1 FFT of size N^D + O(N_src · (2r+1)^D) gather.

Tunable precision via the stencil radius r and the FFT grid size N
(both shared with the existing spreader's precision controls).

The forward gather is `gmix_inverse_nufft_2d`; the gather-only kernel is
`_gather_2d` (the Type-2 analog of `efgp_gmix_spreader._spread_2d`).
"""
from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array


def _gather_2d(
    m: Array,                      # (N_src, 2)
    S: Array,                      # (N_src, 2, 2)
    g_grid: Array,                 # (N, N) complex on uniform spatial grid
    *,
    xcen: Array, h_grid: float, N: int, r: int,
) -> Array:
    """Per-source Gaussian gather of `g_grid` weighted by N(x; m_i, S_i).

    Returns ``out: (N_src,)`` complex with
        out_i = sum_x g_grid(x) · N(x; m_i, S_i) · h_grid^D
    where the sum is over a (2r+1)^D stencil centred at the grid cell
    nearest m_i.

    Adjoint of ``efgp_gmix_spreader._spread_2d``: reading a grid via a
    per-source Gaussian stencil instead of writing a per-source Gaussian
    onto a grid.

    Grid layout: cell index (a, b) is at physical position
        xcen + (a - N//2, b - N//2) * h_grid
    """
    cdtype = g_grid.dtype if jnp.iscomplexobj(g_grid) else jnp.complex64
    rdtype = m.dtype

    # Map source m_i to fractional grid index.
    g = (m - xcen[None, :]) / h_grid + (N // 2)             # (N_src, 2)
    g_int = jnp.round(g).astype(jnp.int32)
    g_frac = g - g_int.astype(rdtype)

    # 2D stencil offsets in floating units, integer cell offsets.
    sten_r = jnp.arange(-r, r + 1, dtype=rdtype)             # (2r+1,)
    sx, sy = jnp.meshgrid(sten_r, sten_r, indexing='ij')     # (2r+1, 2r+1)
    sten_i = jnp.arange(-r, r + 1, dtype=jnp.int32)

    h_grid_pow_D = h_grid ** 2

    def per_source(m_i, S_i, frac_i, gint_i):
        # Physical offset of stencil cell (sx, sy) relative to m_i.
        rel_x = (sx - frac_i[0]) * h_grid                    # (2r+1, 2r+1)
        rel_y = (sy - frac_i[1]) * h_grid
        # Inverse 2x2 covariance.
        det = S_i[0, 0] * S_i[1, 1] - S_i[0, 1] * S_i[1, 0]
        S_inv_00 =  S_i[1, 1] / det
        S_inv_01 = -S_i[0, 1] / det
        S_inv_11 =  S_i[0, 0] / det
        Q = (S_inv_00 * rel_x ** 2
             + 2 * S_inv_01 * rel_x * rel_y
             + S_inv_11 * rel_y ** 2)
        # N(x; m_i, S_i) = (1/(2pi sqrt(det))) exp(-0.5 Q)
        norm = 1.0 / (2.0 * math.pi * jnp.sqrt(det))
        weights = (norm * jnp.exp(-0.5 * Q)).astype(cdtype)  # (2r+1, 2r+1)
        # Read g_grid at the corresponding (wrapped) integer cells.
        idx_x = (gint_i[0] + sten_i)[:, None] % N            # (2r+1, 1)
        idx_y = (gint_i[1] + sten_i)[None, :] % N            # (1, 2r+1)
        idx_x_b = jnp.broadcast_to(idx_x, weights.shape)
        idx_y_b = jnp.broadcast_to(idx_y, weights.shape)
        g_vals = g_grid[idx_x_b, idx_y_b]                    # (2r+1, 2r+1)
        # Riemann-sum h_grid^D factor.
        return jnp.sum(g_vals * weights) * h_grid_pow_D

    return jax.vmap(per_source)(m, S, g_frac, g_int)         # (N_src,)


def gmix_inverse_nufft_2d(
    m: Array, S: Array,
    fk_grid: Array,                # (M_per_dim, M_per_dim) complex
    *,
    xcen: Array, h_spec: float, M_per_dim: int,
    N: int, stencil_r: int,
) -> Array:
    """Closed-form per-source Σ_δ fk(δ) · e^{2πi m_i^⊤δ} · e^{-2π² δ^⊤ S_i δ}
    via inverse-FFT + Gaussian gather (no deconvolution).

    Mirrors the structure of ``gmix_nufft_2d`` (forward Type-1 + spreader),
    but in reverse: 1 inverse-FFT to spatial grid + per-source Gaussian
    gather. No per-source FFT — the Gaussian envelope is *applied* on the
    spatial side via the gather kernel, not on the spectral side.

    Args:
      m, S:   per-source means/covariances; shapes ``(N_src, 2)``, ``(N_src, 2, 2)``.
      fk_grid: spectral coefficient block at frequencies
                ``ξ = (j_x, j_y) * h_spec`` for ``j ∈ [-M//2, M//2)``,
                centered (i.e.\ DC at index ``M//2``).
      xcen:   data centre (matches the spreader's xcen convention).
      h_spec: spectral grid spacing.
      M_per_dim: side length of the input fk_grid.
      N:      FFT zero-pad grid size (controls spatial resolution h_grid = 1/(N h_spec)).
      stencil_r: gather stencil half-width (controls truncation precision; pick via
                 ``stencil_radius_for(S, h_grid)`` from the spreader module).

    Returns
    -------
    out : (N_src,) complex
    """
    h_grid = 1.0 / (N * h_spec)
    cdtype = fk_grid.dtype if jnp.iscomplexobj(fk_grid) else jnp.complex64

    # Zero-pad the centered fk_grid to (N, N) on a centered N-grid; the
    # +i Fourier *inverse* (numpy ifft, +i convention) is what gives us
    # c̃(x) = sum_δ c(δ) exp(+2πi x^⊤δ) on the spatial grid.
    half = M_per_dim // 2
    c = N // 2
    pad_grid = jnp.zeros((N, N), dtype=cdtype)
    pad_grid = pad_grid.at[c - half:c - half + M_per_dim,
                              c - half:c - half + M_per_dim].set(fk_grid)
    # Phase undo for xcen-centered output (mirror of gmix_nufft_2d's
    # +xcen phase, but applied as -xcen here since this is the inverse
    # direction).
    rdtype = m.dtype
    j_axis = jnp.arange(-(N // 2), N - (N // 2), dtype=rdtype)
    JX, JY = jnp.meshgrid(j_axis, j_axis, indexing='ij')
    xi_x = JX * h_spec
    xi_y = JY * h_spec
    phase = jnp.exp(-2j * math.pi * (xi_x * xcen[0] + xi_y * xcen[1]))
    pad_grid = pad_grid * phase
    # Discrete +i IFFT, with the N^2 factor needed to make
    # ``c̃(x) ≈ sum_δ c(δ) e^{+2πi x^⊤δ}`` (numpy ifft has a built-in
    # 1/N^D normalisation we cancel out).
    c_tilde = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(pad_grid))
                                  ) * (N ** 2)

    return _gather_2d(m, S, c_tilde, xcen=xcen, h_grid=h_grid,
                        N=N, r=stencil_r)
