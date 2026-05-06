"""Custom Gaussian-mixture spreader for closed-form q(x) expectations.

Implements variant (A) from ``efgp_estep_charfun.tex`` §5: per-source
anisotropic Gaussian spreader + FFT, no deconvolution step.  Computes

    F(xi) = sum_i w_i exp(2pi i m_i^T xi) exp(-2 pi^2 xi^T S_i xi)

at output frequencies on a regular grid, in O(T (2r+1)^D + N^D log N).

The key recognition: F is the Fourier transform of the Gaussian mixture
g(x) = sum_i w_i N(x; m_i, S_i).  We tabulate g on a fine spatial grid
by spreading per-source onto a (2r+1)^D stencil, then take the FFT.
"""
from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array


def stencil_radius_for(S: Array, h_grid: float, *, n_sigma: float = 4.0,
                        ) -> int:
    """Pick stencil half-width covering n_sigma * sqrt(lambda_max(S_max))."""
    # 2x2 SPD eigenvalue: trace/2 + sqrt((trace/2)^2 - det)
    tr2 = (S[..., 0, 0] + S[..., 1, 1]) / 2.0
    det = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
    disc = jnp.sqrt(jnp.maximum(tr2 ** 2 - det, 0.0))
    lam_max = tr2 + disc
    sigma_max = float(jnp.sqrt(jnp.max(lam_max)))
    return int(math.ceil(n_sigma * sigma_max / h_grid))


def _spread_2d(
    m: Array, S: Array, w: Array, *,
    xcen: Array, h_grid: float, N: int, r: int,
) -> Array:
    """Tabulate g(x) = sum_i w_i N(x; m_i, S_i) on an (N, N) regular grid.

    Grid layout: cell index (a, b) is at physical position
        xcen + (a - N//2, b - N//2) * h_grid
    """
    cdtype = w.dtype if jnp.iscomplexobj(w) else jnp.complex64
    rdtype = m.dtype

    g = (m - xcen[None, :]) / h_grid + (N // 2)             # (T, 2)
    g_int = jnp.round(g).astype(jnp.int32)
    g_frac = g - g_int.astype(rdtype)

    sten_r = jnp.arange(-r, r + 1, dtype=rdtype)             # (2r+1,)
    sx, sy = jnp.meshgrid(sten_r, sten_r, indexing='ij')     # (2r+1, 2r+1)
    sten_i = jnp.arange(-r, r + 1, dtype=jnp.int32)

    def per_source(m_i, S_i, w_i, frac_i, gint_i):
        rel_x = (sx - frac_i[0]) * h_grid
        rel_y = (sy - frac_i[1]) * h_grid
        det = S_i[0, 0] * S_i[1, 1] - S_i[0, 1] * S_i[1, 0]
        S_inv_00 =  S_i[1, 1] / det
        S_inv_01 = -S_i[0, 1] / det
        S_inv_11 =  S_i[0, 0] / det
        Q = (S_inv_00 * rel_x ** 2
             + 2 * S_inv_01 * rel_x * rel_y
             + S_inv_11 * rel_y ** 2)
        norm = 1.0 / (2.0 * math.pi * jnp.sqrt(det))
        contrib = (w_i * norm * jnp.exp(-0.5 * Q)).astype(cdtype)
        return contrib, gint_i

    contribs, centers = jax.vmap(per_source)(m, S, w, g_frac, g_int)

    tar_x = (centers[:, 0:1, None] + sten_i[None, :, None]) % N      # (T, 2r+1, 1)
    tar_y = (centers[:, 1:2, None] + sten_i[None, None, :]) % N      # (T, 1, 2r+1)
    tar_x_b = jnp.broadcast_to(tar_x, contribs.shape).reshape(-1)
    tar_y_b = jnp.broadcast_to(tar_y, contribs.shape).reshape(-1)

    grid = jnp.zeros((N, N), dtype=cdtype)
    grid = grid.at[tar_x_b, tar_y_b].add(contribs.reshape(-1))
    return grid


def gmix_nufft_2d(
    m: Array, S: Array, w: Array, *,
    xcen: Array, h_spec: float, N: int, stencil_r: int,
) -> Array:
    """Closed-form Σ_i w_i e^{2πi m_i^T ξ} e^{-2π² ξ^T S_i ξ} at all FFT-grid
    frequencies, evaluated as spread + single FFT (no deconvolution).

    Returns
    -------
    F : (N, N) complex
        Indexed by centered frequency: F[N//2 + j_x, N//2 + j_y] is the
        value at xi = (j_x, j_y) * h_spec.
    """
    h_grid = 1.0 / (N * h_spec)
    g_grid = _spread_2d(m, S, w, xcen=xcen, h_grid=h_grid, N=N, r=stencil_r)

    # FFT with +i convention via ifft2.  ifftshift moves the centered
    # grid's origin (cell N//2) to numpy index 0; fftshift puts the
    # output's zero frequency at index N//2.  After this, no extra
    # phase factor is needed for centering — ifftshift already absorbs
    # the (-1)^j shift artefact for integer mode indices.
    G = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(g_grid)))

    # Phase from data being centered at xcen (vs origin):
    #   F(ξ) = e^{2πi ξ^T xcen} * <FT of centered-at-zero density at ξ>
    rdtype = m.dtype
    j_axis = jnp.arange(-(N // 2), N - (N // 2), dtype=rdtype)
    JX, JY = jnp.meshgrid(j_axis, j_axis, indexing='ij')
    xi_x = JX * h_spec
    xi_y = JY * h_spec
    phase = jnp.exp(2j * math.pi * (xi_x * xcen[0] + xi_y * xcen[1]))

    # Riemann-sum factor: h_grid^D * N^D = (N h_grid)^D = (1/h_spec)^D
    return G * phase * (N * h_grid) ** 2


def crop_centered(F: Array, M_per_dim: int) -> Array:
    """Crop the centered FFT cube to a centered (M, M) sub-block."""
    N = F.shape[0]
    half = M_per_dim // 2
    c = N // 2
    return F[c - half:c + half + 1, c - half:c + half + 1]


def pick_grid_size(*, h_spec: float, m_extent: float, sigma_max: float,
                    n_tail_sigma: float = 4.0,
                    n_resolve: float = 4.0) -> int:
    """Pick an FFT grid size N such that:

      1. Fine resolution h_grid = 1/(N h_spec) <= sigma_max / n_resolve
         (so the sharpest source is resolved by ~n_resolve cells per σ),
      2. Total spatial extent N h_grid = 1/h_spec covers
         m_extent + n_tail_sigma * sigma_max,

    and N is the next power of 2 satisfying both.
    """
    cover = m_extent + n_tail_sigma * sigma_max
    # 1/h_spec must >= cover; if h_spec was chosen by EFGP it already does.
    # Binding constraint here is the resolution.
    N_min_resolve = math.ceil(n_resolve / (sigma_max * h_spec))
    N_min_cover = math.ceil(cover * h_spec)  # purely a sanity check
    N = max(N_min_resolve, N_min_cover, 32)
    # Round up to next power of 2
    return 1 << (int(N) - 1).bit_length()
