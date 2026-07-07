"""Direct Hutchinson-NUFFT restoration of Approximation A.

Computes $\\E_{q(x_i)}[V_r(x_i)]$ as a smooth function of $(m_i, S_i)$
inside ``compute_neg_CE_single``, with no surrogate / no PSD projection.
The expectation is

  E_{q(x_i)}[V_r(x_i)]
    = sum_{kl} (A_r^{-1})_{kl} D_k D_l * exp(2pi i (xi_l - xi_k)^T m_i
                                              - 2pi^2 (xi_l - xi_k)^T S_i (xi_l - xi_k))
    = sum_delta omega_r(delta) * Phi(m_i, S_i; delta),

with omega_r(delta) := sum_{(k,l): xi_l - xi_k = delta} (A_r^{-1})_{kl} D_k D_l
the BTTB projection of $D A_r^{-1} D$ onto the spectral-difference grid.

We estimate omega_r via Hutchinson with shared probes:
  * draw z_{p,r} ~ Rademacher in spectral-grid space (length M)
  * solve u_{p,r} = A_r^{-1} z_{p,r} via CG (existing make_A_apply machinery)
  * D-weight: tilde_u = D * u, tilde_z = D * z
  * cross-correlate via FFT on padded grid: omega_{p,r} = xcorr(tilde_u, tilde_z)
  * average over P probes
This is exactly the existing gmix-aux primitive (autocorrelation -> diff-grid
NUFFT-with-Gaussian-envelope), just with omega instead of rho.

Per inner iter, per output dim r, the per-source E[V_r(x_i)] is a
Gaussian-smoothed Type-2 NUFFT on the difference grid -- same primitive
as ``gmix_E_full_Eff``. Plain ``jax.grad`` through this gives the
unbiased Bonnet/Price gradient of E[V] w.r.t. (m_i, S_i); the gradient
is noisy with low P but unbiased.

Cost: P D_out CG solves once per outer iter (small) + per-inner-iter
diff-grid NUFFT, same order as the existing E[bar_f^T bar_f] step in
the gmix path. ~5-10% wall overhead at K=10/T=500 vs the Taylor
surrogate's 50%.
"""
from __future__ import annotations

import sing.efgp_jax_primitives as jp

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from sing.sde import SDE
from sing.efgp_jax_drift import ef_with_jac_grad, eff_with_grads
from sing.efgp_gmix_qx_moments import (
    _delta_grid_for_autocorr,
    GmixQxAux,
)


def _hutch_omega_one(z_real: Array, A_apply, M_inv_apply, ws_real: Array,
                       cdtype, grid_shape: Tuple[int, ...],
                       cg_tol: float, max_cg_iter: int) -> Array:
    """One Hutchinson probe: returns ω(δ) on the FFT-padded difference
    grid (post-fftshift), as a **complex** array. Uses the codebase's
    Jacobi-preconditioned ``jp.cg_solve`` (proper Hermitian ``vdot``
    inner product).

    For source locations that aren't centrosymmetric, the BTTB kernel ν
    is complex-Hermitian (ν(-δ) = ν(δ)*), making A complex-Hermitian
    and ω complex-Hermitian as well. Dropping ``.imag`` discards the
    Im(ω) sin(2π δᵀm) term in $\\E[V]$, which can be the dominant error
    (verified numerically vs dense ground truth). Keep complex.
    """
    z_c = z_real.astype(cdtype)
    u = jp.cg_solve(A_apply, z_c, tol=cg_tol, max_iter=max_cg_iter,
                     M_inv_apply=M_inv_apply)
    # D-weight both arms of the xcorr; this gives ω(δ) = (DA^{-1}D)_{·,·+δ}
    # in expectation under E[zz^T] = I.
    tilde_u = (ws_real * u).reshape(grid_shape)
    tilde_z = (ws_real * z_c).reshape(grid_shape)
    pad_shape = tuple(2 * n for n in grid_shape)
    Fu = jnp.fft.fftn(tilde_u, s=pad_shape)
    Fz = jnp.fft.fftn(tilde_z, s=pad_shape)
    # xcorr(tilde_u, tilde_z)(δ) = sum_k tilde_u^*(k) tilde_z(k + δ)
    # = IFFT( conj(Fu) * Fz ); fftshift puts lag 0 at the centre.
    omega = jnp.fft.ifftn(jnp.conj(Fu) * Fz)
    return jnp.fft.fftshift(omega)


def precompute_omega_per_r(mu_r_unused: Array, grid: jp.JaxGridState,
                              top: jp.ToeplitzNDJax,
                              key: Array, *,
                              D_out: int, P: int = 1,
                              cg_tol: float = 1e-4,
                              max_cg_iter: int = 200
                              ) -> GmixQxAux:
    """Precompute per-output-dim omega(delta), summed over output dims,
    plus the difference-grid lag coordinates. Drop-in replacement for
    ``efgp_gmix_qx_moments.precompute_aux`` but for $A_r^{-1}$ instead
    of $\\mu_r$'s autocorrelation.

    Returns a ``GmixQxAux`` whose ``rho_summed`` is the Hutchinson
    estimate $\\sum_r \\sigma_r^{-2} \\omega_r$ (the sigma^{-2} factor is
    folded in upstream where E[V] is added to Eff). For now: no sigma
    weighting -- leave that to the caller.

    Args:
      mu_r_unused: only here for parity with ``precompute_aux``; not used.
      grid: spectral grid.
      top: BTTB Toeplitz structure for $A_r$ (from ``return_top=True``).
      key: PRNG key for the P Hutchinson probes.
      D_out: number of output dims (replicated A_r for isotropic sigma).
      P: number of probes (low values are fine; gradient is unbiased,
         just noisy; bump if NaN'ing or trajectory wobble).
      cg_tol, max_cg_iter: CG tolerances for the $A_r^{-1} z$ solves.
    """
    grid_shape = tuple(int(n) for n in grid.mtot_per_dim)
    ws_real = grid.ws.real
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    # Jacobi preconditioner (mirrors compute_mu_r_jax line 207-217)
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    center_idx = tuple((s - 1) // 2 for s in top.v_fft.shape)
    v_kernel = jnp.fft.fftshift(jnp.fft.ifftn(top.v_fft))
    T_diag = v_kernel[center_idx].real
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real
    diag_A = (1.0 + ws_sq * T_diag).astype(rdtype)
    M_inv_diag = (1.0 / diag_A)

    def M_inv_apply(v):
        # JAX CG passes through whatever dtype its initial guess uses;
        # multiply preserves v's dtype rather than upcasting.
        return v * M_inv_diag.astype(v.dtype)

    # Total probes = P * D_out; each gives an omega(delta) array on the
    # padded grid. For isotropic sigma (the only case in production),
    # all D_out output dims share the same A_r -- summing over D_out is
    # just replication of one Hutchinson estimate. We average over the
    # full P*D_out probes (i.e. effective P_eff = P*D_out probes).
    total = P * D_out
    keys = jr.split(key, total)

    def one_probe(k_):
        z = (jr.bernoulli(k_, shape=(grid.M,)).astype(rdtype) * 2.0 - 1.0)
        return _hutch_omega_one(z, A_apply, M_inv_apply, ws_real, cdtype,
                                  grid_shape, cg_tol, max_cg_iter)

    omegas = jax.vmap(one_probe)(keys)                         # (total, *pad)
    # Average across (P, D_out) probes (sum over D_out happens elsewhere).
    omega_summed = omegas.sum(0) * (D_out / total)             # = (D_out / P) sum_p
    omega_flat = omega_summed.reshape(-1).astype(grid.ws.dtype)

    delta_flat = _delta_grid_for_autocorr(grid)                # (P_grid, d)
    return GmixQxAux(rho_summed=omega_flat, delta_flat=delta_flat)


def _abs_frame_fk_grid(rho_env: Array, delta: Array, xcen: Array,
                        pad_shape, cdtype) -> Array:
    r"""Fold the absolute-frame phase $e^{+2\pi i\,\delta\cdot x_{\rm cen}}$
    into the folded $(\rho\cdot{\rm env})$ coefficient grid so that a
    Type-2 NUFFT evaluated with centre ``grid.xcen`` returns the
    ABSOLUTE-frame value $V(x)=\sum_\delta \rho(\delta)\,e^{2\pi i\,\delta\cdot x}$.

    Why this is needed: $\omega(\delta)$ from :func:`precompute_omega_per_r`
    lives in the ABSOLUTE frame -- the gmix spreader builds $A$ (hence
    ``top`` and $\omega=\mathrm{BTTB}(D A^{-1} D)$) with basis
    $e^{2\pi i\,m\cdot\xi}$; see the frame note at
    ``sing/efgp_jax_drift.py:405-417``. A plain
    ``nufft2(x, fk, grid.xcen, h)`` computes
    $\sum_\delta fk(\delta)\,e^{2\pi i\,\delta\cdot(x-x_{\rm cen})}$, i.e.
    it evaluates $V$ at $(x-x_{\rm cen})$ -- a real spatial offset on
    off-origin data (invisible when $x_{\rm cen}\approx0$, which is why the
    centred unit tests never caught it). Pre-multiplying by
    $e^{+2\pi i\,\delta\cdot x_{\rm cen}}$ cancels the $-x_{\rm cen}$ shift
    while KEEPING the NUFFT coordinates centred at ``xcen`` (numerically
    safe: the nonuniform phases stay $\sim 2\pi h(x-x_{\rm cen})$ rather
    than blowing up as $2\pi h x$). This mirrors the $-x_{\rm cen}$
    phase-undo already present in the hetS gather
    (:func:`sing.efgp_gmix_gather.gmix_inverse_nufft_2d`). No-op when
    $x_{\rm cen}\approx0$.
    """
    phase = jnp.exp(2j * jnp.pi
                    * (delta @ xcen.astype(delta.dtype))).astype(cdtype)
    return (rho_env * phase).reshape(pad_shape)


def E_V_at_all_homogeneous(m_per_source: Array, S_homo: Array,
                              omega_aux: GmixQxAux,
                              grid: jp.JaxGridState,
                              nufft_eps: float = 6e-8) -> Array:
    """Evaluate $\\sum_r \\E_{q(x_i)}[V_r(x_i)]$ at ALL eval points
    $\\{m_i\\}$ in a single Type-2 NUFFT. Treats $S_i \\equiv S_{\\rm homo}$
    (homogeneous covariance) so the envelope is factored once outside
    the per-source loop.

    Cost: $O(M\\log M + N)$ vs $O(NM)$ for the per-source einsum.

    Args:
      m_per_source: ``(N, d)`` eval points.
      S_homo:       ``(d, d)`` shared covariance for the Gaussian envelope.
      omega_aux:    Hutchinson aux with $\\rho$ on the FFT-padded difference
                    grid and the corresponding lag coordinates.
      grid:         spectral grid (provides h_per_dim, xcen for NUFFT-2).

    Returns:
      ``(N,)`` real array of $\\sum_r \\E[V_r(x_i)]$ at each $m_i$.
    """
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    rho   = omega_aux.rho_summed.astype(cdtype)              # (P,)
    delta = omega_aux.delta_flat.astype(rdtype)              # (P, d)
    # Envelope on the difference grid -- one materialisation, shared
    # across all eval points.
    quad = jnp.einsum('pd,de,pe->p', delta, S_homo.astype(rdtype), delta)
    env  = jnp.exp(-2 * jnp.pi**2 * quad).astype(cdtype)
    # Reshape onto the FFT-padded grid; jp.nufft2 expects natural-
    # order block layout matching its mtot_per_dim shape parameter.
    # Fold in the absolute-frame phase so the NUFFT (evaluated with
    # grid.xcen for numerical centring) returns V(x), not V(x - xcen).
    pad_shape = tuple(2 * int(n) for n in grid.mtot_per_dim)
    fk_grid = _abs_frame_fk_grid(rho * env, delta, grid.xcen, pad_shape, cdtype)
    # Standard Type-2 NUFFT at the eval points; the difference-grid h
    # equals the spectral grid h, and lag 0 sits at the centre of the
    # padded grid (consistent with ``_delta_grid_for_autocorr``'s
    # ``arange(2n) - n`` lag layout, which matches jax_finufft's
    # implicit ``[-N/2, N/2)`` convention).
    return jp.nufft2(m_per_source, fk_grid,
                      grid.xcen, grid.h_per_dim,
                      eps=nufft_eps).real


def E_V_at(m: Array, S: Array, omega_aux: GmixQxAux,
            grid: jp.JaxGridState) -> Array:
    """Single-source einsum version (for correctness testing only;
    do not use inside an inner-iter scan -- see
    ``E_V_at_all_homogeneous`` for the production O(M log M + N) path).
    Returns a real scalar."""
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    rho     = omega_aux.rho_summed.astype(cdtype)
    delta   = omega_aux.delta_flat.astype(rdtype)
    d_phase = jnp.exp(2j * jnp.pi * (delta @ m.astype(rdtype)).astype(cdtype))
    d_quad  = jnp.einsum('pd,de,pe->p', delta, S.astype(rdtype), delta)
    d_env   = jnp.exp(-2 * jnp.pi**2 * d_quad).astype(cdtype)
    return (rho * d_phase * d_env).real.sum()


def precompute_E_V_per_t(omega_aux: GmixQxAux,
                            ms: Array,
                            S_homo: Array,
                            grid: jp.JaxGridState,
                            nufft_eps: float = 6e-8) -> Array:
    """Batched Type-2 NUFFT to evaluate $\\E_{q(x_i)}[V(x_i)]$ at every
    source $\\{m_{k,t}\\}$ in one call. Homogeneous-$S$ baseline: a
    single Gaussian envelope $e^{-2\\pi^2 \\delta^\\top S_{\\rm homo} \\delta}$
    is folded into $\\rho$ once, so the per-source eval is a *standard*
    Type-2 NUFFT against a shared ``fk_grid``.

    Cost: $O(P)$ envelope mult + 1 batched NUFFT-2 of cost
    $O(M\\log M + N)$ where $N = K(T-1)$. **Single FFT shared across all
    sources** — this is the asymptotic improvement over the per-source
    einsum / per-source NUFFT-2 (which both pay $O(NM)$).

    Args:
      omega_aux: Hutchinson aux with $\\rho$ on the FFT-padded
                 difference grid.
      ms:        ``(K, T, D)`` source points.
      S_homo:    ``(D, D)`` shared covariance for the Gaussian envelope.
      grid:      spectral grid (provides h_per_dim, xcen, mtot_per_dim).

    Returns:
      ``V_per_kt: (K, T)`` real array of $\\sum_r \\E[V_r]$ at each source.
    """
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    rho   = omega_aux.rho_summed.astype(cdtype)
    delta = omega_aux.delta_flat.astype(rdtype)
    quad  = jnp.einsum('pd,de,pe->p', delta, S_homo.astype(rdtype), delta)
    env   = jnp.exp(-2 * jnp.pi**2 * quad).astype(cdtype)
    # Fold in the absolute-frame phase (see _abs_frame_fk_grid): keeps the
    # NUFFT numerically centred at grid.xcen while returning V(x), not
    # V(x - xcen). No-op on centred data.
    pad_shape = tuple(2 * int(n) for n in grid.mtot_per_dim)
    fk_grid = _abs_frame_fk_grid(rho * env, delta, grid.xcen, pad_shape, cdtype)

    # Batched NUFFT-2: vmap over K trials, single call per trial
    # evaluates at all T sources. Shared FFT inside each trial.
    def per_trial(ms_k):
        return jp.nufft2(ms_k.astype(rdtype), fk_grid,
                          grid.xcen, grid.h_per_dim,
                          eps=nufft_eps).real
    return jax.vmap(per_trial)(ms)                                    # (K, T)


def precompute_V_and_grad_E_V_per_t(omega_aux: GmixQxAux,
                                       ms: Array,
                                       S_homo: Array,
                                       grid: jp.JaxGridState,
                                       nufft_eps: float = 6e-8
                                       ) -> Tuple[Array, Array]:
    """Single forward + reverse-mode pass: returns
    ``(V_per_kt, grad_V_per_kt)`` of shapes ``(K, T)`` and ``(K, T, D)``.

    Implementation: ``jax.grad`` of ``sum(precompute_E_V_per_t(ms_))``
    w.r.t. ``ms_`` yields the per-source $\\partial\\E[V]/\\partial m_i$
    via reverse-mode AD through ``jp.nufft2``. JAX's nufft2 VJP runs
    one Type-1 NUFFT per spatial dim, so total cost is one forward
    Type-2 plus $D$ adjoint Type-1 NUFFTs per outer iter -- the same
    asymptotic cost as the explicit "$D$ frequency-modulated NUFFT-2"
    formulation, with cleaner code.

    Returns ``(V_kt, grad_V_kt)``: ``V_kt = (K, T)``,
    ``grad_V_kt = (K, T, D)``. Both real.
    """
    def V_sum(ms_):
        V_kt = precompute_E_V_per_t(omega_aux, ms_, S_homo, grid,
                                       nufft_eps=nufft_eps)
        return V_kt.sum(), V_kt
    grad_V_kt, V_kt = jax.grad(V_sum, has_aux=True)(ms)
    return V_kt, grad_V_kt


# ---------------------------------------------------------------------------
# Heterogeneous-S path: gmix-gather (Type-2 analog of the Gaussian
# spreader, lives in efgp_gmix_gather). Per-source S_i envelope so the
# per-source Type-2 NUFFT-with-envelope cannot be folded into a shared
# fk_grid; the gather replaces the per-source NUFFT-2 with a per-source
# Gaussian-stencil read of an inverse-FFT'd ω̃ on a uniform spatial grid.
# Cost: 1 IFFT (size N×N) + per-source O((2r+1)^D) gather. Tunable r
# (precision/cost tradeoff via n_sigma); r=ceil(n_sigma·σ_max/h_grid).
# ---------------------------------------------------------------------------
def precompute_E_V_per_t_hetS(omega_aux: GmixQxAux,
                                ms: Array,
                                Ss: Array,
                                grid: jp.JaxGridState,
                                *,
                                gather_N: int,
                                stencil_r: int,
                                ) -> Array:
    """Heterogeneous-$S$ analog of :func:`precompute_E_V_per_t`.

    Each source $(m_i, S_i)$ has its own Gaussian envelope $e^{-2\\pi^2
    \\delta^\\top S_i \\delta}$, so the envelope cannot be pre-folded
    into a shared $\\rho\\cdot\\text{env}$ coefficient grid as in the
    homogeneous-$S$ baseline. We use the gmix-gather primitive (1 IFFT
    of $\\omega(\\delta)$ to a spatial grid + per-source Gaussian
    stencil read of $\\widetilde\\omega(x)$, $\\widetilde\\omega := \\F^{-1}\\omega$).

    Args:
      omega_aux: Hutchinson aux with $\\rho$ on the $2M$-padded
                 difference grid (same as homogeneous-$S$ path).
      ms:        ``(K, T, D)`` per-source means.
      Ss:        ``(K, T, D, D)`` per-source covariances.
      grid:      spectral grid (provides ``h_per_dim``, ``xcen``,
                 ``mtot_per_dim``).
      gather_N:  FFT zero-pad size for the IFFT step. Choose
                 ``gather_N >= 2 * grid.mtot_per_dim[0]`` (next
                 power-of-2 typically). Larger ``gather_N`` → finer
                 spatial grid → larger ``stencil_r`` for fixed precision.
      stencil_r: half-width of the per-source Gaussian stencil. Use
                 :func:`sing.efgp_gmix_spreader.stencil_radius_for`
                 (with ``n_sigma=2`` for ${\\sim}5\\%$ rel error or
                 ``n_sigma=4`` for full ${\\sim}10^{-5}$ rel error).

    Returns:
      ``V_per_kt: (K, T)`` real array.
    """
    from sing.efgp_gmix_gather import gmix_inverse_nufft_2d

    M_per_dim_diff = 2 * int(grid.mtot_per_dim[0])         # difference-grid side
    h_spec = grid.h_per_dim[0]                             # JAX scalar OK
    xcen = grid.xcen
    # Absolute-frame centring (off-origin data). omega(delta) is absolute
    # (see _abs_frame_fk_grid), but gmix_inverse_nufft_2d applies xcen TWICE
    # -- once as its internal e^{-2πi ξ·xcen} phase-undo and once as the
    # spatial readout at (m - xcen) in _gather_2d -- so a raw call returns
    # V(m - 2·xcen). Pre-multiplying omega by e^{+2πi δ·(2·xcen)} cancels
    # both, giving V(m). No-op when xcen ≈ 0 (why the centred gather tests
    # never caught it). Verified to ~1e-5 vs the E_V_at einsum reference in
    # test_V_hutch_offcenter_absolute_frame.
    cdtype = grid.ws.dtype
    delta = omega_aux.delta_flat.astype(grid.xcen.dtype)
    abs_phase = jnp.exp(
        2j * jnp.pi * (delta @ (2.0 * grid.xcen))).astype(cdtype)
    fk_grid = (omega_aux.rho_summed.astype(cdtype) * abs_phase).reshape(
        M_per_dim_diff, M_per_dim_diff)

    K, T, D = ms.shape
    ms_flat = ms.reshape(-1, D)
    Ss_flat = Ss.reshape(-1, D, D)
    V_flat = gmix_inverse_nufft_2d(
        ms_flat, Ss_flat, fk_grid,
        xcen=xcen, h_spec=h_spec,
        M_per_dim=M_per_dim_diff,
        N=gather_N, stencil_r=stencil_r).real
    return V_flat.reshape(K, T)


def precompute_V_and_grad_E_V_per_t_hetS(omega_aux: GmixQxAux,
                                            ms: Array,
                                            Ss: Array,
                                            grid: jp.JaxGridState,
                                            *,
                                            gather_N: int,
                                            stencil_r: int,
                                            ) -> Tuple[Array, Array]:
    """Heterogeneous-$S$ analog: returns ``(V_kt, grad_V_kt)``.

    Gradient via ``jax.grad`` reverse-mode through the gmix-gather
    primitive (Gaussian stencil + IFFT). Cost: ${\\sim}2\\times$ forward
    (one extra adjoint pass).
    """
    def V_sum(ms_):
        V_kt = precompute_E_V_per_t_hetS(
            omega_aux, ms_, Ss, grid,
            gather_N=gather_N, stencil_r=stencil_r)
        return V_kt.sum(), V_kt
    grad_V_kt, V_kt = jax.grad(V_sum, has_aux=True)(ms)
    return V_kt, grad_V_kt


# ---------------------------------------------------------------------------
# Custom-VJP shim: batched-NUFFT precompute injected as a constant +
# precomputed gradient. Rationale: nat_grad_transition is per-transition
# vmap'd, so we cannot batch the V evaluation inside ``ff(t, m, S)`` --
# JAX/XLA does not amortise FFTs across vmap'd nufft2 calls. We
# precompute V_per_t and grad_V_per_t in one batched call OUTSIDE the
# vmap, then inject into the per-transition autodiff via custom_vjp.
# ---------------------------------------------------------------------------
@jax.custom_vjp
def _v_with_grad(m: Array, V_const: Array, grad_const: Array) -> Array:
    """Returns ``V_const`` (scalar) with VJP $\\partial L / \\partial m =
    \\text{grad\\_const} \\cdot \\partial L / \\partial V$.

    ``V_const`` and ``grad_const`` are precomputed at the same point
    $m$ via batched NUFFT-2; injecting them with this custom VJP
    exposes a per-transition Bonnet gradient
    $\\partial \\E[V] / \\partial m_i$ to ``compute_neg_CE_single``'s
    autodiff trace without paying per-transition NUFFT cost.
    """
    return V_const

def _v_fwd(m, V_const, grad_const):
    return V_const, (grad_const,)

def _v_bwd(res, g):
    (grad_const,) = res
    return (g * grad_const,                                            # ∂L/∂m
            jnp.zeros_like(jnp.asarray(0.0)),                         # ∂L/∂V_const
            jnp.zeros_like(grad_const))                                # ∂L/∂grad_const

_v_with_grad.defvjp(_v_fwd, _v_bwd)


# ---------------------------------------------------------------------------
# Combined custom-VJP for ff(m, S) = ||Ef||² + tr[J^T J S] + V, returning
# the value as Eff_const + V_const (both injected) and propagating the
# linearised B' gradient through (m, S):
#   ∂L/∂m = ∂L/∂ff · (2 J^T Ef + grad_V_const)
#   ∂L/∂S = ∂L/∂ff · (J^T J)
# Replaces TWO custom_vjp calls (eff_with_grads + _v_with_grad) per
# transition with ONE — eliminates ~half the per-transition shim
# overhead, which the differential bench shows dominates V-Hutch's wall
# at production size.
# ---------------------------------------------------------------------------
@jax.custom_vjp
def _ff_plus_v_with_grads(m, S, Eff_legacy_const, Ef_const, jac_const,
                              V_const, grad_V_const):
    """Returns ``Eff_legacy_const + V_const``; backward injects the
    full ``∂(Eff+V)/∂(m,S)`` by adding ``grad_V_const`` to the legacy
    ``2 J^T Ef`` Bonnet term."""
    del S
    return Eff_legacy_const + V_const

def _ff_v_fwd(m, S, Eff_legacy_const, Ef_const, jac_const,
                  V_const, grad_V_const):
    return Eff_legacy_const + V_const, (Ef_const, jac_const, grad_V_const)

def _ff_v_bwd(res, g):
    Ef_const, jac_const, grad_V_const = res
    grad_m = g * (2.0 * (Ef_const @ jac_const) + grad_V_const)
    grad_S = g * (jac_const.T @ jac_const)
    return (grad_m, grad_S, jnp.zeros(()),
            jnp.zeros_like(Ef_const), jnp.zeros_like(jac_const),
            jnp.zeros(()), jnp.zeros_like(grad_V_const))

_ff_plus_v_with_grads.defvjp(_ff_v_fwd, _ff_v_bwd)


@jax.tree_util.register_pytree_node_class
class FrozenEFGPDriftWithVHutch(SDE):
    """SDE shim that adds $\\sum_r\\sigma_r^{-2}\\E[V_r]$ to
    ``ff(t, m, S)`` via custom-VJP injection of a precomputed
    (V_per_t, grad_V_per_t) field.

    The (V, grad_V) field is computed once per outer iter via TWO
    batched Type-2 NUFFTs (one for V, one for ∂V/∂m -- D such NUFFTs
    for the D-dim gradient): cost $O((D+1)\\cdot K \\cdot (M\\log M + T))$
    per outer iter. The per-transition shim looks up V[idx] and exposes
    the precomputed gradient via custom_vjp. ∂V/∂S is dropped (Price
    term -- analogous to legacy B' asymmetric drop, gradient kept
    consistent with the linearised expression).

    Holds:
      Ef_per_t, Eff_per_t, Edfdx_per_t : legacy moments (for $\\bar f$).
      V_per_t      : (T,) precomputed $\\sum_r\\E[V_r]$ at $m_t$.
      grad_V_per_t : (T, D) precomputed $\\partial V/\\partial m_t$.
    """

    def __init__(self, *, latent_dim: int, t_grid: Array,
                 Ef_per_t: Array, Eff_per_t: Array, Edfdx_per_t: Array,
                 V_per_t: Array, grad_V_per_t: Array,
                 D_out: int):
        super().__init__(expectation=None, latent_dim=latent_dim)
        self._t_grid = t_grid
        self._Ef = Ef_per_t
        self._Eff = Eff_per_t
        self._Edfdx = Edfdx_per_t
        self._V = V_per_t                    # (T,)
        self._grad_V = grad_V_per_t          # (T, D)
        self._D_out = D_out

    def tree_flatten(self):
        children = (self._Ef, self._Eff, self._Edfdx,
                    self._V, self._grad_V)
        aux = (self.latent_dim, self._t_grid, self._D_out)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        latent_dim, t_grid, D_out = aux
        Ef, Eff, Edfdx, V, gV = children
        return cls(latent_dim=latent_dim, t_grid=t_grid,
                   Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx,
                   V_per_t=V, grad_V_per_t=gV, D_out=D_out)

    def _idx(self, t):
        return jnp.argmin(jnp.abs(self._t_grid - t))

    def drift(self, drift_params, x, t):
        return self._Ef[self._idx(t)]

    def f(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        return ef_with_jac_grad(m, self._Ef[idx], self._Edfdx[idx])

    def ff(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        # Single combined custom_vjp: Eff_legacy + V value, with
        # gradient ∂/∂m = (2 J^T Ef) + grad_V, ∂/∂S = J^T J. Replaces
        # two separate custom_vjp calls — cuts per-transition shim
        # overhead in half.
        return _ff_plus_v_with_grads(
            m, S, self._Eff[idx], self._Ef[idx], self._Edfdx[idx],
            self._V[idx], self._grad_V[idx])

    def dfdx(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        return self._Edfdx[self._idx(t)]
