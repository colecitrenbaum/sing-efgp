"""GPU-parallel alternatives to :func:`sing.efgp_jax_drift.compute_mu_r_analytic_jax`.

The analytic q(f) update evaluates, for a set of frequencies ``freqs`` and
source strengths ``b``,

    F(freq) = Σ_i b_i · e^{-2πi freq·(m_i - xcen)} · e^{-2π² freqᵀ S_i freq}

both on the spectral grid (the RHS ``h_r``) and on the BTTB difference grid
(the Toeplitz generator).  The production analytic path
(:func:`compute_mu_r_analytic_jax`) computes this with type-1 NUFFTs
(``jax_finufft`` → cuFINUFFT) plus a Taylor expansion of the per-source
envelope in ``ΔS_i = S_i − S̄``.  That is FLOP-optimal (O(N + M log M)) but
each NUFFT crosses into the FINUFFT library with plan/launch overhead and is
latency-bound on GPU — the reason the gmix path (pure-JAX scatter + FFT, fully
fused by XLA) parallelizes better there.

The transforms in the analytic path all share the SAME nodes ``m_src`` and the
SAME output grid; only the per-source strength vector changes.  That is exactly
the structure a dense matmul (NUDFT) or a shared-footprint gridding exploits.

This module provides drop-in replacements for ``compute_mu_r_analytic_jax``
(same signature, same returned ``(mu_r, None, top)`` in the relative/xcen-aware
frame) that stay inside XLA:

* :func:`compute_mu_r_densegemm_jax` — **dense NUDFT via GEMM** (Strategy 2).
  Builds ``E[freq, i] = e^{-2πi freq·(m_i−xcen) − 2π² freqᵀ S_i freq}`` and
  contracts against the strengths with a single (batched) complex matmul.
  O(F·N) FLOPs — deliberately *not* FLOP-optimal — but one cuBLAS GEMM, the
  most throughput-efficient GPU primitive, and **exact in per-source S**
  (no Taylor truncation), so it also removes the analytic path's large-T
  ΔS accuracy regression.  ``freq_chunk`` bounds peak memory via ``lax.scan``.

* :func:`compute_mu_r_gridnufft_jax` — **pure-JAX gridding NUFFT** (Strategy 1).
  A hand-rolled type-1 NUFFT (Gaussian gridding + ``jnp.fft`` + deconvolution)
  with the Taylor-envelope handling of ΔS, so it stays FLOP-frugal but never
  leaves XLA and shares the spreading footprint across the whole strength
  batch.  2-D only.

All three are 2-D only (matching the production analytic path).
"""
from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

import sing.efgp_jax_primitives as jp
from sing.efgp_jax_drift import _env_taylor_pairs


# ---------------------------------------------------------------------------
# Strategy 2: dense NUDFT via GEMM (exact per-source S)
# ---------------------------------------------------------------------------
def _nudft_dense(freqs: Array,      # (F, 2) real
                 m_rel: Array,      # (N, 2) real  (already xcen-subtracted)
                 S_src: Array,      # (N, 2, 2) real
                 B: Array,          # (nB, N) complex  strength batch
                 cdtype,
                 freq_chunk: int | None):
    """Return ``ewft`` of shape ``(nB, F)`` with

        ewft[b, k] = Σ_i B[b, i] e^{-2πi freqs[k]·m_rel_i − 2π² freqs[k]ᵀ S_i freqs[k]}.

    ``freq_chunk`` (if not None) processes the frequency axis in blocks of that
    size via ``lax.scan`` to bound peak memory at ``freq_chunk · N`` instead of
    ``F · N``; otherwise a single dense GEMM.
    """
    fx = freqs[:, 0]
    fy = freqs[:, 1]
    S00 = S_src[:, 0, 0]
    S01 = S_src[:, 0, 1]
    S11 = S_src[:, 1, 1]
    two_pi2 = 2.0 * (math.pi ** 2)

    def block(fx_b, fy_b):
        # phase[k, i] = freq_k · m_rel_i ; E_osc = exp(-2πi phase)
        phase = fx_b[:, None] * m_rel[None, :, 0] + fy_b[:, None] * m_rel[None, :, 1]
        # quad[k, i] = freq_kᵀ S_i freq_k
        quad = (fx_b[:, None] ** 2 * S00[None, :]
                + 2.0 * (fx_b[:, None] * fy_b[:, None]) * S01[None, :]
                + fy_b[:, None] ** 2 * S11[None, :])
        E = jnp.exp((-2j * math.pi) * phase.astype(cdtype)
                    - (two_pi2 * quad).astype(cdtype))          # (Fb, N)
        return B @ E.T                                          # (nB, Fb)

    if freq_chunk is None or freq_chunk >= freqs.shape[0]:
        return block(fx, fy)

    F = freqs.shape[0]
    n_chunk = (F + freq_chunk - 1) // freq_chunk
    pad = n_chunk * freq_chunk - F
    fx_p = jnp.concatenate([fx, jnp.zeros(pad, fx.dtype)]).reshape(n_chunk, freq_chunk)
    fy_p = jnp.concatenate([fy, jnp.zeros(pad, fy.dtype)]).reshape(n_chunk, freq_chunk)

    def scan_body(carry, xy):
        fxb, fyb = xy
        return carry, block(fxb, fyb)                          # (nB, freq_chunk)

    _, out = jax.lax.scan(scan_body, None, (fx_p, fy_p))       # (n_chunk, nB, freq_chunk)
    out = jnp.transpose(out, (1, 0, 2)).reshape(B.shape[0], n_chunk * freq_chunk)
    return out[:, :F]


def compute_mu_r_densegemm_jax(
    m_src: Array,                     # (N_src, D_lat)
    S_src: Array,                     # (N_src, D_lat, D_lat)
    d_src: Array,                     # (N_src, D_lat)
    C_src: Array,                     # (N_src, D_lat, D_lat)
    weights: Array,                   # (N_src,)
    grid: jp.JaxGridState,
    *,
    sigma_drift_sq: float,
    D_lat: int,
    D_out: int,
    freq_chunk: int | None = None,
    cg_tol: float = 1e-5,
    max_cg_iter: int = 2000,
    **_ignored,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Dense-GEMM NUDFT analytic q(f) update (Strategy 2, exact per-source S).

    Drop-in for :func:`sing.efgp_jax_drift.compute_mu_r_analytic_jax`:
    identical inputs/returns and relative (xcen-aware) frame.  Accepts and
    ignores ``order``/``nufft_eps`` (kept for signature parity) — this path is
    exact in S so there is no Taylor order.
    """
    if D_lat != 2:
        raise NotImplementedError("compute_mu_r_densegemm_jax: 2-D only.")
    cdtype = grid.ws.dtype
    xcen = grid.xcen
    h = grid.h_per_dim
    mtot = grid.mtot_per_dim
    ws_real_c = grid.ws.real.astype(cdtype)
    m_rel = m_src - xcen[None, :]

    # ---- BTTB generator on the difference grid (Ld per dim) ----
    Ld = tuple(2 * (mm - 1) + 1 for mm in mtot)
    hm = (Ld[0] - 1) // 2
    dax = jnp.arange(-hm, hm + 1, dtype=m_src.dtype) * h[0]
    DX, DY = jnp.meshgrid(dax, dax, indexing='ij')
    dfreqs = jnp.stack([DX.ravel(), DY.ravel()], axis=-1)       # (Ld², 2)

    w_T = (weights / sigma_drift_sq).astype(cdtype)[None, :]    # (1, N)
    v_kernel = _nudft_dense(dfreqs, m_rel, S_src, w_T, cdtype, freq_chunk)[0]
    v_kernel = v_kernel.reshape(Ld)
    top = jp.make_toeplitz(v_kernel, force_pow2=True)
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)

    # Jacobi preconditioner (identical to the MC / gmix / analytic paths)
    center_idx = tuple((s - 1) // 2 for s in v_kernel.shape)
    T_diag = v_kernel[center_idx].real.astype(jnp.float32)
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real.astype(jnp.float32)
    M_inv_diag = (1.0 / (1.0 + ws_sq * T_diag)).astype(cdtype)
    M_inv_apply = lambda v: M_inv_diag * v

    # ---- RHS on the spectral grid ----
    xi = grid.xis_flat
    bases = [d_src[:, r] / sigma_drift_sq for r in range(D_out)]
    for r in range(D_out):
        for j in range(D_lat):
            bases.append(C_src[:, j, r] / sigma_drift_sq)
    B = jnp.stack(bases).astype(cdtype)                         # (nB, N)
    ewft = _nudft_dense(xi, m_rel, S_src, B, cdtype, freq_chunk)  # (nB, M)

    def _idx_C(r, j):
        return D_out + r * D_lat + j

    def per_r(r):
        h1 = ws_real_c * ewft[r]
        h2 = jnp.zeros(grid.M, dtype=cdtype)
        for j in range(D_lat):
            xi_j = xi[:, j].astype(cdtype)
            h2 = h2 + (-2j * math.pi * xi_j) * (ws_real_c * ewft[_idx_C(r, j)])
        h_r = h1 + h2
        rhs_norm = jnp.linalg.norm(h_r).real
        return jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter,
                                   M_inv_apply=M_inv_apply),
            operand=None,
        )

    mu_r = jnp.stack([per_r(r) for r in range(D_out)], axis=0)
    return mu_r, None, top


# ---------------------------------------------------------------------------
# Strategy 2b: SEPARABLE NUDFT via GEMM (tensor-product frequency grid)
# ---------------------------------------------------------------------------
def _nudft_separable(fx1d: Array,       # (nx,) real 1-D x-frequencies
                     fy1d: Array,       # (ny,) real 1-D y-frequencies
                     m_rel: Array,      # (N, 2) xcen-subtracted nodes
                     S_src: Array,      # (N, 2, 2)
                     B: Array,          # (nB, N) complex strengths
                     cdtype,
                     xorder: int) -> Array:
    """Separable dense NUDFT exploiting the tensor-product frequency grid.

    The target on the full grid ξ = (fx1d[a], fy1d[c]) is
        F[b, a, c] = Σ_i B[b,i] e^{-2πi(fx_a m_ix + fy_c m_iy)}
                              · e^{-2π²(S00_i fx_a² + 2 S01_i fx_a fy_c + S11_i fy_c²)}.
    The oscillatory factor and the DIAGONAL envelope terms factor into 1-D
    matrices ``Ax`` (nx, N) and ``Ay`` (ny, N); the OFF-diagonal cross term
    ``e^{-4π² S01_i fx_a fy_c}`` is the only part that couples the two axes,
    handled by a short Taylor expansion of order ``xorder`` (each order adds
    one separable rank-term).  Peak memory is O((nx+ny)·N + nB·nx·ny) rather
    than the dense path's O(nx·ny·N) — the scaling win — at the cost of
    (xorder+1) batched GEMMs.  Returns ``(nB, nx·ny)`` (row-major a-outer,
    matching ``meshgrid(indexing='ij').ravel()``).
    """
    two_pi2 = 2.0 * (math.pi ** 2)
    S00 = S_src[:, 0, 0]; S01 = S_src[:, 0, 1]; S11 = S_src[:, 1, 1]
    nx = fx1d.shape[0]; ny = fy1d.shape[0]; nB = B.shape[0]

    Ax = jnp.exp((-2j * math.pi) * (fx1d[:, None] * m_rel[None, :, 0]).astype(cdtype)
                 - (two_pi2 * fx1d[:, None] ** 2 * S00[None, :]).astype(cdtype))  # (nx,N)
    Ay = jnp.exp((-2j * math.pi) * (fy1d[:, None] * m_rel[None, :, 1]).astype(cdtype)
                 - (two_pi2 * fy1d[:, None] ** 2 * S11[None, :]).astype(cdtype))  # (ny,N)

    F = jnp.zeros((nB, nx, ny), dtype=cdtype)
    fact = 1.0
    for p in range(xorder + 1):
        if p > 0:
            fact *= p
        coeff = ((-4.0 * math.pi ** 2) ** p) / fact
        Wp = (B * (S01[None, :] ** p)).astype(cdtype)              # (nB, N)
        AxW = Ax[None, :, :] * Wp[:, None, :]                      # (nB, nx, N)
        Gp = jnp.einsum('bai,ci->bac', AxW, Ay)                    # (nB, nx, ny)
        fxp = (fx1d ** p).astype(cdtype)
        fyp = (fy1d ** p).astype(cdtype)
        F = F + coeff * (fxp[None, :, None] * fyp[None, None, :]) * Gp
    return F.reshape(nB, nx * ny)


def compute_mu_r_sepgemm_jax(
    m_src: Array,
    S_src: Array,
    d_src: Array,
    C_src: Array,
    weights: Array,
    grid: jp.JaxGridState,
    *,
    sigma_drift_sq: float,
    D_lat: int,
    D_out: int,
    xorder: int = 2,
    cg_tol: float = 1e-5,
    max_cg_iter: int = 2000,
    **_ignored,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Separable-GEMM NUDFT analytic q(f) update (Strategy 2b).

    Same drop-in signature/returns as :func:`compute_mu_r_densegemm_jax`, but
    O((nx+ny)·N) memory instead of O(nx·ny·N).  Exact in the diagonal of S;
    the off-diagonal S is captured by an order-``xorder`` cross-term Taylor.
    """
    if D_lat != 2:
        raise NotImplementedError("compute_mu_r_sepgemm_jax: 2-D only.")
    cdtype = grid.ws.dtype
    xcen = grid.xcen
    h = grid.h_per_dim
    mtot = grid.mtot_per_dim
    ws_real_c = grid.ws.real.astype(cdtype)
    m_rel = m_src - xcen[None, :]

    # ---- BTTB generator (separable over the difference grid) ----
    Ld = tuple(2 * (mm - 1) + 1 for mm in mtot)
    hm = (Ld[0] - 1) // 2
    dfx = jnp.arange(-hm, hm + 1, dtype=m_src.dtype) * h[0]        # (Ld,)
    w_T = (weights / sigma_drift_sq).astype(cdtype)[None, :]
    v_kernel = _nudft_separable(dfx, dfx, m_rel, S_src, w_T, cdtype, xorder)[0]
    v_kernel = v_kernel.reshape(Ld)
    top = jp.make_toeplitz(v_kernel, force_pow2=True)
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)

    center_idx = tuple((s - 1) // 2 for s in v_kernel.shape)
    T_diag = v_kernel[center_idx].real.astype(jnp.float32)
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real.astype(jnp.float32)
    M_inv_diag = (1.0 / (1.0 + ws_sq * T_diag)).astype(cdtype)
    M_inv_apply = lambda v: M_inv_diag * v

    # ---- RHS (separable over the spectral grid) ----
    K = (int(mtot[0]) - 1) // 2
    sfx = jnp.arange(-K, K + 1, dtype=m_src.dtype) * h[0]          # (mtot,) 1-D ξ
    xi = grid.xis_flat
    bases = [d_src[:, r] / sigma_drift_sq for r in range(D_out)]
    for r in range(D_out):
        for j in range(D_lat):
            bases.append(C_src[:, j, r] / sigma_drift_sq)
    B = jnp.stack(bases).astype(cdtype)
    ewft = _nudft_separable(sfx, sfx, m_rel, S_src, B, cdtype, xorder)  # (nB, M)

    def _idx_C(r, j):
        return D_out + r * D_lat + j

    def per_r(r):
        h1 = ws_real_c * ewft[r]
        h2 = jnp.zeros(grid.M, dtype=cdtype)
        for j in range(D_lat):
            xi_j = xi[:, j].astype(cdtype)
            h2 = h2 + (-2j * math.pi * xi_j) * (ws_real_c * ewft[_idx_C(r, j)])
        h_r = h1 + h2
        rhs_norm = jnp.linalg.norm(h_r).real
        return jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter,
                                   M_inv_apply=M_inv_apply),
            operand=None,
        )

    mu_r = jnp.stack([per_r(r) for r in range(D_out)], axis=0)
    return mu_r, None, top


# ---------------------------------------------------------------------------
# Strategy 1: pure-JAX gridding NUFFT (shared footprint) + Taylor envelope
# ---------------------------------------------------------------------------
def _grid_nudft_batched(m_rel: Array,      # (N, 2)  xcen-subtracted nodes
                        strengths: Array,  # (P, N)  complex per-transform weights
                        out_shape: Tuple[int, int],
                        h: Array,          # (2,) spectral spacing (h_k)
                        cdtype,
                        *,
                        n_grid: int,
                        gw: int,
                        beta: float) -> Array:
    """Batched pure-JAX type-1 NUDFT sharing a single spreading footprint.

    Computes, for each transform p and each output frequency k on the natural
    (iflag=-1) grid of shape ``out_shape``,

        f_p[k] = Σ_i strengths[p, i] e^{-2πi ξ_k · m_rel_i}.

    Implemented by Gaussian-gridding the nodes onto an oversampled spatial
    grid of size ``n_grid`` (gridding half-width ``gw``, shape parameter
    ``beta``), one batched ``jnp.fft.ifftn``, deconvolution by the analytic
    gridding-kernel FT, and a centered crop.  The per-node grid footprint and
    kernel weights are geometry-only (independent of ``strengths``), so they
    are computed ONCE and reused across all ``P`` transforms — the batch only
    changes the scatter values.  Everything stays inside XLA.

    ``out_shape`` is odd per dim (2K+1); the fundamental spatial period is
    ``1/h`` so the spatial grid spacing is ``dx = 1/(h·n_grid)``.
    """
    Kx = (out_shape[0] - 1) // 2
    Ky = (out_shape[1] - 1) // 2
    P, N = strengths.shape
    rdtype = m_rel.dtype

    # Spatial grid: period L = 1/h per dim, spacing dx = L/n_grid.
    Lx = 1.0 / h[0]
    Ly = 1.0 / h[1]
    dx = Lx / n_grid
    dy = Ly / n_grid

    # Node position in oversampled-cell units, centered at n_grid//2.
    gx = m_rel[:, 0] / dx + (n_grid // 2)
    gy = m_rel[:, 1] / dy + (n_grid // 2)
    gxi = jnp.round(gx).astype(jnp.int32)
    gyi = jnp.round(gy).astype(jnp.int32)
    fxr = gx - gxi.astype(rdtype)
    fyr = gy - gyi.astype(rdtype)

    off = jnp.arange(-gw, gw + 1, dtype=rdtype)              # (2gw+1,)
    offi = jnp.arange(-gw, gw + 1, dtype=jnp.int32)
    # Gaussian gridding kernel exp(-beta * (r*dx)^2) style, but in cell units:
    # kernel(off) = exp(-beta * ((off - frac) )^2)  (dimensionless cell offset)
    kx = jnp.exp(-beta * (off[None, :] - fxr[:, None]) ** 2)   # (N, 2gw+1)
    ky = jnp.exp(-beta * (off[None, :] - fyr[:, None]) ** 2)   # (N, 2gw+1)
    # Outer product kernel footprint per node: (N, 2gw+1, 2gw+1)
    kern = (kx[:, :, None] * ky[:, None, :]).astype(cdtype)

    # Target cells (wrapped) per node.
    tx = (gxi[:, None] + offi[None, :]) % n_grid               # (N, 2gw+1)
    ty = (gyi[:, None] + offi[None, :]) % n_grid               # (N, 2gw+1)
    TX = jnp.broadcast_to(tx[:, :, None], (N, 2 * gw + 1, 2 * gw + 1)).reshape(-1)
    TY = jnp.broadcast_to(ty[:, None, :], (N, 2 * gw + 1, 2 * gw + 1)).reshape(-1)

    def spread_one(s):                                         # s: (N,)
        vals = (s[:, None, None] * kern).reshape(-1)
        g = jnp.zeros((n_grid, n_grid), dtype=cdtype)
        return g.at[TX, TY].add(vals)

    grids = jax.vmap(spread_one)(strengths)                    # (P, n_grid, n_grid)

    # FFT (iflag=-1 convention → forward FFT here; we normalize so result is a
    # plain sum with no 1/n).  Use fftn with fftshift on input (node origin at
    # cell n_grid//2) and on output (zero freq → center).
    G = jnp.fft.fftshift(
        jnp.fft.fftn(jnp.fft.ifftshift(grids, axes=(1, 2)), axes=(1, 2)),
        axes=(1, 2))

    # Deconvolve by the gridding-kernel FT.  Kernel k(u)=exp(-beta u²) in cell
    # units → its DTFT sampled at integer mode j is ∝ exp(-(π j)²/(beta n²))·… ;
    # we use the standard Gaussian-gridding correction exp(+ (π j / n)² / beta).
    # Gaussian-gridding correction: kernel exp(-beta·(cell offset)²) has
    # continuous FT amplitude ∝ (β/π)^{D/2} and spectral deconvolution
    # exp(π²j²/(n²β)) per dim.  The (β/π) amplitude (with the dx^D Riemann
    # factor already absorbed) is a global scale but does NOT cancel in
    # mu_r = (I + D T D)⁻¹ h, so it must be included.
    j_axis = jnp.arange(-(n_grid // 2), n_grid - (n_grid // 2))
    corr1d = jnp.exp((math.pi * j_axis / n_grid) ** 2 / beta)
    CX, CY = jnp.meshgrid(corr1d, corr1d, indexing='ij')
    amp = (beta / math.pi)                                     # (β/π)^{D/2}, D=2
    corr = (amp * CX * CY).astype(cdtype)
    G = G * corr[None]

    # Crop centered (2K+1) block.
    c = n_grid // 2
    G = G[:, c - Kx:c + Kx + 1, c - Ky:c + Ky + 1]
    return G.reshape(P, -1)                                    # (P, M)


def compute_mu_r_gridnufft_jax(
    m_src: Array,
    S_src: Array,
    d_src: Array,
    C_src: Array,
    weights: Array,
    grid: jp.JaxGridState,
    *,
    sigma_drift_sq: float,
    D_lat: int,
    D_out: int,
    order: int = 1,
    n_grid_mult: int = 2,
    gw: int = 6,
    beta: float | None = None,
    cg_tol: float = 1e-5,
    max_cg_iter: int = 2000,
    **_ignored,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Pure-JAX gridding-NUFFT analytic q(f) update (Strategy 1).

    Same math as :func:`compute_mu_r_analytic_jax` (Taylor envelope in ΔS) but
    the type-1 transforms are a hand-rolled Gaussian-gridding NUFFT that stays
    in XLA.  Drop-in signature/returns.  2-D only.
    """
    if D_lat != 2:
        raise NotImplementedError("compute_mu_r_gridnufft_jax: 2-D only.")
    cdtype = grid.ws.dtype
    xcen = grid.xcen
    h = grid.h_per_dim
    mtot = grid.mtot_per_dim
    ws_real_c = grid.ws.real.astype(cdtype)
    m_rel = m_src - xcen[None, :]
    Sbar = S_src.mean(axis=0)
    dS = S_src - Sbar[None]

    n_spec = int(mtot[0])
    Ld = tuple(2 * (mm - 1) + 1 for mm in mtot)
    n_grid_spec = n_grid_mult * (1 << (n_spec - 1).bit_length())
    n_grid_diff = n_grid_mult * (1 << (Ld[0] - 1).bit_length())
    if beta is None:
        # Standard Gaussian-gridding shape: beta ≈ π·(2-1/mult)/... ; a robust
        # value giving ~1e-6 accuracy at gw=6, mult=2.
        beta = math.pi * (1.0 - 0.5 / n_grid_mult)

    # ---- BTTB generator ----
    hm = (Ld[0] - 1) // 2
    dax = jnp.arange(-hm, hm + 1, dtype=m_src.dtype) * h[0]
    DX, DY = jnp.meshgrid(dax, dax, indexing='ij')
    dfreqs = jnp.stack([DX.ravel(), DY.ravel()], axis=-1)
    env_diff = jnp.exp(-2 * (math.pi ** 2)
                       * (Sbar[0, 0] * DX ** 2 + 2 * Sbar[0, 1] * DX * DY
                          + Sbar[1, 1] * DY ** 2)).astype(cdtype)
    w_T = weights / sigma_drift_sq
    pairs_d = _env_taylor_pairs(dfreqs, dS, order)
    coef_d = jnp.stack([cf for cf, _ in pairs_d]).astype(cdtype)         # (P, Ld²)
    str_d = jnp.stack([(w_T * wf) for _, wf in pairs_d]).astype(cdtype)  # (P, N)
    Fd_all = _grid_nudft_batched(m_rel, str_d, Ld, h, cdtype,
                                 n_grid=n_grid_diff, gw=gw, beta=beta)   # (P, Ld²)
    Fd = (coef_d * Fd_all).sum(axis=0)
    v_kernel = Fd.reshape(Ld) * env_diff
    top = jp.make_toeplitz(v_kernel, force_pow2=True)
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)

    center_idx = tuple((s - 1) // 2 for s in v_kernel.shape)
    T_diag = v_kernel[center_idx].real.astype(jnp.float32)
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real.astype(jnp.float32)
    M_inv_diag = (1.0 / (1.0 + ws_sq * T_diag)).astype(cdtype)
    M_inv_apply = lambda v: M_inv_diag * v

    # ---- RHS ----
    xi = grid.xis_flat
    env_spec = jnp.exp(-2 * (math.pi ** 2)
                       * (Sbar[0, 0] * xi[:, 0] ** 2
                          + 2 * Sbar[0, 1] * xi[:, 0] * xi[:, 1]
                          + Sbar[1, 1] * xi[:, 1] ** 2)).astype(cdtype)
    pairs_s = _env_taylor_pairs(xi, dS, order)
    P = len(pairs_s)
    coef_s = jnp.stack([cf for cf, _ in pairs_s]).astype(cdtype)         # (P, M)
    wfac_s = jnp.stack([wf for _, wf in pairs_s])                        # (P, N)

    bases = [d_src[:, r] / sigma_drift_sq for r in range(D_out)]
    for r in range(D_out):
        for j in range(D_lat):
            bases.append(C_src[:, j, r] / sigma_drift_sq)
    Bm = jnp.stack(bases)                                                # (nB, N)
    nB = Bm.shape[0]
    prods = (Bm[:, None, :] * wfac_s[None, :, :]).reshape(nB * P, -1).astype(cdtype)
    FT = _grid_nudft_batched(m_rel, prods, mtot, h, cdtype,
                             n_grid=n_grid_spec, gw=gw, beta=beta)       # (nB·P, M)
    FT = FT.reshape(nB, P, grid.M)
    ewft = (coef_s[None] * FT).sum(axis=1) * env_spec[None]              # (nB, M)

    def _idx_C(r, j):
        return D_out + r * D_lat + j

    def per_r(r):
        h1 = ws_real_c * ewft[r]
        h2 = jnp.zeros(grid.M, dtype=cdtype)
        for j in range(D_lat):
            xi_j = xi[:, j].astype(cdtype)
            h2 = h2 + (-2j * math.pi * xi_j) * (ws_real_c * ewft[_idx_C(r, j)])
        h_r = h1 + h2
        rhs_norm = jnp.linalg.norm(h_r).real
        return jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter,
                                   M_inv_apply=M_inv_apply),
            operand=None,
        )

    mu_r = jnp.stack([per_r(r) for r in range(D_out)], axis=0)
    return mu_r, None, top
