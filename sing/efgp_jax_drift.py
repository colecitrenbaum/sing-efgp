"""
Pure-JAX q(f) update + drift-moment evaluation for SING-EFGP.

Provides the EFGP drift-block primitives used by :mod:`sing.efgp_em`'s
jit-compiled EM loop.  ``mu_r``, ``Ef``, ``Eff``, ``Edfdx`` are JAX
arrays and the whole pipeline is jit-traceable, so q(f) update + drift
moments + SING natural-grad live in a single compiled graph.

Math symbol → expression in this module
---------------------------------------

  Stein-corrected q(f) update     compute_mu_r_jax
  E[f(m)], E[J_f(m)] at marginals drift_moments_jax
  Combined "one inner-iter step"  qf_and_moments_jax
  Collapsed kernel M-step          m_step_kernel_jax
  SDE wrapper (custom-VJP shim)    FrozenEFGPDrift
"""
from __future__ import annotations

# IMPORTANT: import jax_finufft FIRST via efgp_jax_primitives (load-order)
import sing.efgp_jax_primitives as jp

import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from sing.sde import SDE


# ---------------------------------------------------------------------------
# Custom-VJP delta-method shims (pure JAX; safe to import without torch)
# ---------------------------------------------------------------------------
@jax.custom_vjp
def ef_with_jac_grad(m, ef_const, jac_const):
    """Returns ``ef_const`` (D_out,) with VJP ``∂L/∂m = jacᵀ ∂L/∂ef``."""
    return ef_const


def _ef_fwd(m, ef_const, jac_const):
    return ef_const, (ef_const, jac_const)


def _ef_bwd(res, g):
    ef_const, jac_const = res
    return (jac_const.T @ g, jnp.zeros_like(ef_const), jnp.zeros_like(jac_const))


ef_with_jac_grad.defvjp(_ef_fwd, _ef_bwd)


@jax.custom_vjp
def eff_with_grads(m, S, eff_const, ef_const, jac_const):
    """``E[||f||²]`` with local-quadratic VJP through (m, S)."""
    del S
    return eff_const


def _eff_fwd(m, S, eff_const, ef_const, jac_const):
    return eff_const, (ef_const, jac_const)


def _eff_bwd(res, g):
    ef_const, jac_const = res
    grad_m = 2.0 * g * (ef_const @ jac_const)
    grad_S = g * (jac_const.T @ jac_const)
    return (grad_m, grad_S, jnp.zeros(()),
            jnp.zeros_like(ef_const), jnp.zeros_like(jac_const))


eff_with_grads.defvjp(_eff_fwd, _eff_bwd)


@jax.tree_util.register_pytree_node_class
class FrozenEFGPDrift(SDE):
    """Pure-JAX SDE shim: returns precomputed (Ef, Eff, Edfdx) per time
    step with delta-method custom VJPs through (m, S).

    The drift moments are computed off-graph (typically once per inner
    E-step iter via :func:`compute_mu_r_jax` + :func:`drift_moments_jax`)
    and exposed only as values + first-order sensitivities to SING's
    natural-grad update.  This is the local-quadratic transition
    approximation:

        E[f(x)]                ≈  f(m)              ∂/∂m: J_f(m)        ∂/∂S: 0
        E[f^T Σ⁻¹ f]           ≈  f(m)^T Σ⁻¹ f(m)   ∂/∂m: 2 Jᵀ Σ⁻¹ f    ∂/∂S: Jᵀ Σ⁻¹ J
        E[J_f(x)]              ≈  J_f(m)            (treat constant)

    Registered as a JAX pytree so it can be passed across ``jax.vmap``
    over a leading trial axis: leaves ``(Ef, Eff, Edfdx)`` get sliced,
    aux ``(latent_dim, t_grid)`` is shared.
    """

    def __init__(self, *, latent_dim: int, t_grid: Array,
                 Ef_per_t: Array, Eff_per_t: Array, Edfdx_per_t: Array):
        super().__init__(expectation=None, latent_dim=latent_dim)
        self._t_grid = t_grid
        self._Ef = Ef_per_t
        self._Eff = Eff_per_t
        self._Edfdx = Edfdx_per_t

    def tree_flatten(self):
        children = (self._Ef, self._Eff, self._Edfdx)
        aux = (self.latent_dim, self._t_grid)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        latent_dim, t_grid = aux
        Ef, Eff, Edfdx = children
        return cls(latent_dim=latent_dim, t_grid=t_grid,
                   Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)

    def _idx(self, t):
        return jnp.argmin(jnp.abs(self._t_grid - t))

    def drift(self, drift_params, x, t):
        return self._Ef[self._idx(t)]

    def f(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        return ef_with_jac_grad(m, self._Ef[idx], self._Edfdx[idx])

    def ff(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        return eff_with_grads(m, S, self._Eff[idx],
                              self._Ef[idx], self._Edfdx[idx])

    def dfdx(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        return self._Edfdx[self._idx(t)]


def _build_pseudo_cloud(
    m_src: Array,        # (N_src, D_lat)
    S_src: Array,        # (N_src, D_lat, D_lat)
    key: Array,
    S_marginal: int,
    D_lat: int,
) -> Array:
    """Sample S_marginal pseudo-points per source: x^(s) ~ N(m_i, S_i).

    Returns ``X_flat`` of shape ``(N_src * S_marginal, D_lat)``.

    Sources here are flat across trials and transitions: callers (the
    multi-trial qf_and_moments_*_jax wrappers) flatten ``(K, T-1, D)``
    Stein quantities into ``(K*(T-1), D)`` before calling.  Padded
    transitions (where trial_mask=0) enter with their associated
    ``weights`` set to 0, so the corresponding pseudo-points contribute
    nothing to the BTTB or RHS regardless of where they sit in space.
    """
    N_src = m_src.shape[0]
    eps = jax.random.normal(key, (N_src, S_marginal, D_lat))
    L_chol = jnp.linalg.cholesky(S_src + 1e-9 * jnp.eye(D_lat))
    X_cloud = m_src[:, None, :] + jnp.einsum('iab,isb->isa', L_chol, eps)
    X_flat = X_cloud.reshape(-1, D_lat)
    return X_flat


def compute_mu_r_jax(
    m_src: Array,                     # (N_src, D_lat)
    S_src: Array,                     # (N_src, D_lat, D_lat)
    d_src: Array,                     # (N_src, D_lat)         — Stein d_{·, :}
    C_src: Array,                     # (N_src, D_lat, D_lat)  — Stein C_{·, :, :}
    weights: Array,                   # (N_src,)               — del_t * trans_mask
    grid: jp.JaxGridState,
    key: Array,
    *,
    sigma_drift_sq: float,
    S_marginal: int,
    D_lat: int,
    D_out: int,
    cg_tol: float = 1e-5,
    max_cg_iter: int = 2000,    # generous; with Jacobi precond CG usually converges in <50
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Pure-JAX Stein-corrected q(f) update.  Returns (mu_r, X_flat, top).

    Inputs are pre-flattened across trials and transitions so this
    function is K-agnostic: ``N_src = K * (T-1)`` for K trials of length
    T.  Padded transitions enter with ``weights[i] = 0`` and contribute
    nothing to the BTTB Toeplitz or RHS.

    ``mu_r`` shape: (D_out, M).  Implements the no-inputs case of the
    EFGP q(f) update with σ_r² tied across output dims (v0).
    """
    M = grid.M
    cdtype = grid.ws.dtype

    # Pseudo-input cloud (S_marginal samples per source)
    X_flat = _build_pseudo_cloud(m_src, S_src, key, S_marginal, D_lat)
    weights_per_pseudo = jnp.repeat(weights, S_marginal)         # (N_src*S,)

    # BTTB Toeplitz
    bttb_w = weights_per_pseudo / (S_marginal * sigma_drift_sq)
    v_kernel = jp.bttb_conv_vec_weighted(
        X_flat, bttb_w.astype(cdtype), grid.xcen, grid.h_per_dim,
        grid.mtot_per_dim, eps=nufft_eps)
    top = jp.make_toeplitz(v_kernel, force_pow2=True)
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    ws_real_c = grid.ws.real.astype(cdtype)

    # Jacobi preconditioner: diag(A)_k = 1 + |ws_k|^2 * T_diag where
    # T_diag = central value of v_kernel = sum of weights = O(N).  Without
    # preconditioning, CG iters scale as sqrt(cond(A)) ~ sqrt(N), which is
    # fine for N ~ 10K but borderline at 20K+ pseudo-points.  Jacobi gives
    # ~10x CG speedup at our largest T values and prevents the eval drift
    # blow-up at T=10K (debugged 2026-05-02).
    center_idx = tuple((s - 1) // 2 for s in v_kernel.shape)
    T_diag = v_kernel[center_idx].real.astype(jnp.float32)
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real.astype(jnp.float32)
    diag_A = 1.0 + ws_sq * T_diag
    M_inv_diag = (1.0 / diag_A).astype(cdtype)
    M_inv_apply = lambda v: M_inv_diag * v

    # Per-output-dim RHS: pseudo-cloud factor ``1/(S σ²)``.  The per-source
    # mask is already folded into d_src / C_src by the caller's
    # ``_flatten_stein``, so padded transitions contribute nothing.

    # Build h_r per output dim r in a vmap
    def per_r(r):
        # h_1,r = D F_X* a_r,    (a_r)_{i,s} = d_{i,r} / (S σ²)
        a_r = jnp.repeat(d_src[:, r], S_marginal) / (S_marginal * sigma_drift_sq)
        Fstar_a = jp.nufft1(X_flat, a_r.astype(cdtype), grid.xcen,
                             grid.h_per_dim,
                             out_shape=grid.mtot_per_dim,
                             eps=nufft_eps).reshape(-1)
        h1 = ws_real_c * Fstar_a

        # h_2,r = Σ_j conj(2πi ξ_j) ⊙ D F_X* c_{j,r} = Σ_j (-2πi ξ_j) ⊙ ...
        # The factor comes from J_φ^* (adjoint Jacobian) — conjugate of
        # ∂φ_k/∂x_j = 2πi ξ_{kj} φ_k, so (-2πi ξ).
        def per_j(j):
            c_jr = jnp.repeat(C_src[:, j, r], S_marginal) \
                / (S_marginal * sigma_drift_sq)
            Fstar_c = jp.nufft1(X_flat, c_jr.astype(cdtype), grid.xcen,
                                 grid.h_per_dim,
                                 out_shape=grid.mtot_per_dim,
                                 eps=nufft_eps).reshape(-1)
            xi_j = grid.xis_flat[:, j].astype(cdtype)
            return (-2j * math.pi * xi_j) * (ws_real_c * Fstar_c)

        h2 = jax.vmap(per_j)(jnp.arange(D_lat)).sum(axis=0)
        h_r = h1 + h2
        # CG solve (Jacobi preconditioned).  Skip if RHS is essentially
        # zero (initial iter).
        rhs_norm = jnp.linalg.norm(h_r).real
        mu = jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter,
                                   M_inv_apply=M_inv_apply),
            operand=None,
        )
        return mu

    mu_r = jax.vmap(per_r)(jnp.arange(D_out))                    # (D_out, M)
    return mu_r, X_flat, top


# ---------------------------------------------------------------------------
# Closed-form variant: Gaussian-mixture spreader (no MC over q(x))
# ---------------------------------------------------------------------------
def compute_mu_r_gmix_jax(
    m_src: Array,                     # (N_src, D_lat)
    S_src: Array,                     # (N_src, D_lat, D_lat)
    d_src: Array,                     # (N_src, D_lat)         — Stein d_{·, :}
    C_src: Array,                     # (N_src, D_lat, D_lat)  — Stein C_{·, :, :}
    weights: Array,                   # (N_src,)               — del_t * trans_mask
    grid: jp.JaxGridState,
    *,
    sigma_drift_sq: float,
    D_lat: int,
    D_out: int,
    fine_N: int,                      # static FFT grid size per dim
    stencil_r: int,                   # static stencil half-width
    cg_tol: float = 1e-5,
    max_cg_iter: int = 2000,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Closed-form q(f) update via per-source Gaussian-mixture spreader.

    Replaces the MC NUFFTs in :func:`compute_mu_r_jax` with a custom
    Gaussian spreader that computes the q(x_i)-expectation of e^{2πi
    x_i^T ξ} exactly (Gaussian characteristic function) and then a single
    FFT.  Keeps BTTB structure, CG, and the rest of the pipeline
    unchanged.

    Inputs are pre-flattened across trials and transitions:
    ``N_src = K * (T-1)`` for K trials of length T.  Padded transitions
    enter with ``weights[i] = 0``.

    See ``efgp_estep_charfun.tex`` for the math, and ``efgp_gmix_spreader.py``
    for the spread + FFT primitive.

    Returns
    -------
    (mu_r, X_flat_or_None, top)
        ``mu_r``: (D_out, M).  ``X_flat_or_None`` is always ``None`` for
        the gmix variant (no cloud).  ``top`` is the BTTB Toeplitz built
        from the closed-form generator.
    """
    from sing.efgp_gmix_spreader import _spread_2d
    cdtype = grid.ws.dtype
    rdtype = m_src.dtype
    h_spec = grid.h_per_dim[0]
    M_per_dim = int(grid.mtot_per_dim[0])
    K_modes = (M_per_dim - 1) // 2

    if D_lat != 2:
        raise NotImplementedError(
            "compute_mu_r_gmix_jax: only D_lat == 2 in v0.")

    # ---- Closed-form spread + FFT helper ----
    h_grid = 1.0 / (fine_N * h_spec)

    def _gmix_fft(w):
        """Spread w_i N(x; m_i, S_i) and FFT.  Returns (fine_N, fine_N) complex
        on centered FFT grid (frequency h_spec * j for j in [-N//2, N//2))."""
        g = _spread_2d(m_src, S_src, w.astype(cdtype),
                       xcen=grid.xcen, h_grid=h_grid, N=fine_N, r=stencil_r)
        G = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(g)))
        # Centering phase: e^{2πi ξ^T xcen}  (grid is centered at xcen)
        j_axis = jnp.arange(-(fine_N // 2), fine_N - (fine_N // 2),
                              dtype=rdtype)
        JX, JY = jnp.meshgrid(j_axis, j_axis, indexing='ij')
        xi_x = JX * h_spec
        xi_y = JY * h_spec
        phase = jnp.exp(2j * math.pi
                         * (xi_x * grid.xcen[0] + xi_y * grid.xcen[1]))
        # Riemann-sum factor: (N h_grid)^D = (1/h_spec)^D
        return G * phase * (fine_N * h_grid) ** 2

    def _crop_centered(F, M):
        c = fine_N // 2
        half = M // 2
        return F[c - half:c + half + 1, c - half:c + half + 1]

    # ---- BTTB generator from closed form ----
    # Output is on the BTTB difference grid: (4K+1, 4K+1)
    # Note: existing make_toeplitz expects shape (2*mtot - 1) per dim.
    # For mtot = 2K+1, that's 4K+1.  Our crop matches.
    w_T = (weights / sigma_drift_sq).astype(cdtype)
    F_T = _gmix_fft(w_T)
    M_diff_per_dim = 2 * M_per_dim - 1                          # = 4K+1
    # MC's bttb_conv_vec_weighted uses nufft1 with iflag=-1; gmix uses
    # iflag=+1.  Conjugate to match MC's BTTB-generator convention.
    v_kernel = jnp.conj(_crop_centered(F_T, M_diff_per_dim))
    top = jp.make_toeplitz(v_kernel, force_pow2=True)
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    ws_real_c = grid.ws.real.astype(cdtype)

    # Jacobi preconditioner
    center_idx = tuple((s - 1) // 2 for s in v_kernel.shape)
    T_diag = v_kernel[center_idx].real.astype(jnp.float32)
    ws_sq = (grid.ws * jnp.conj(grid.ws)).real.astype(jnp.float32)
    diag_A = 1.0 + ws_sq * T_diag
    M_inv_diag = (1.0 / diag_A).astype(cdtype)
    M_inv_apply = lambda v: M_inv_diag * v

    # ---- h_r per output dim r ----
    # Stein quantities ``d_src`` / ``C_src`` are already pre-masked by
    # the caller's ``_flatten_stein``, so padded transitions contribute
    # nothing.  The RHS does NOT carry a ``del_t`` factor (only the BTTB
    # generator does); see ``efgp_estep_charfun.tex``.
    def per_r(r):
        # h_1,r: closed-form FT of  sum_i (d_{i,r} / σ²) N(x; m_i, S_i)
        w_h1 = (d_src[:, r] / sigma_drift_sq).astype(cdtype)
        F_h1 = _gmix_fft(w_h1)
        Fh1_spec = _crop_centered(F_h1, M_per_dim).reshape(-1)   # (M,)
        # h_1,r = D ⊙ adjoint-FT of sources weighted by d_{i,r}/σ²
        # In the MC formulation we used  D ⊙ F_X^* a_r .  F_X^* uses iflag=-1
        # (negative-i convention), while our gmix_nufft_2d uses iflag=+1.
        # Reconcile by conjugating: F_X^* w  = conj( gmix_nufft_2d( conj(w) ) ).
        # For real-valued input weights this just conjugates the output.
        h1 = ws_real_c * jnp.conj(Fh1_spec)

        # h_2,r = sum_j (-2πi ξ_j) ⊙ D ⊙ adjoint-FT of sources weighted by C_{i,j,r}/σ²
        def per_j(j):
            w_h2 = (C_src[:, j, r] / sigma_drift_sq).astype(cdtype)
            F_h2 = _gmix_fft(w_h2)
            Fh2_spec = _crop_centered(F_h2, M_per_dim).reshape(-1)
            xi_j = grid.xis_flat[:, j].astype(cdtype)
            return (-2j * math.pi * xi_j) * (ws_real_c * jnp.conj(Fh2_spec))
        h2 = jax.vmap(per_j)(jnp.arange(D_lat)).sum(axis=0)
        h_r = h1 + h2

        rhs_norm = jnp.linalg.norm(h_r).real
        mu = jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter,
                                   M_inv_apply=M_inv_apply),
            operand=None,
        )
        return mu

    mu_r = jax.vmap(per_r)(jnp.arange(D_out))
    return mu_r, None, top


def _flatten_stein(
    ms: Array,        # (K, T, D)
    Ss: Array,        # (K, T, D, D)
    SSs: Array,       # (K, T-1, D, D)   cross-cov
    del_t: Array,     # (K, T-1) or (T-1,) (broadcast)
    trial_mask: Array,  # (K, T) bool
):
    """Flatten multi-trial Stein quantities into supersource arrays.

    Padded transitions (where trans_mask=0) are zeroed in the BTTB
    weight ``del_t * trans_mask`` and the Stein quantities ``d_src``,
    ``C_src``.  Source positions ``m_src`` / ``S_src`` may still hold
    arbitrary values at padded slots, but they contribute nothing
    because every weighted reduction (BTTB convolution / RHS NUFFT /
    gmix spread) carries the padded zeros through.

    Returns
    -------
    m_src  : (K*(T-1), D)        — source means
    S_src  : (K*(T-1), D, D)     — source covariances
    d_src  : (K*(T-1), D)        — Stein d, masked
    C_src  : (K*(T-1), D, D)     — Stein C, masked
    weights: (K*(T-1),)          — del_t * trans_mask  (BTTB weights)
    """
    K_, T_ = ms.shape[0], ms.shape[1]
    D_ = ms.shape[2]
    trans_mask = trial_mask[:, :-1] & trial_mask[:, 1:]            # (K, T-1)
    tm_f = trans_mask.astype(ms.dtype)                             # (K, T-1)

    # Replace padded covariance slots with the identity to avoid
    # divide-by-zero in downstream Gaussian-mixture spreaders (where
    # det(S) appears in the denominator); the matching weights are zero
    # so these slots contribute nothing.
    eye_d = jnp.eye(D_, dtype=Ss.dtype)
    Ss_safe = jnp.where(tm_f[..., None, None].astype(bool),
                         Ss[:, :-1], eye_d)                         # (K, T-1, D, D)

    m_src = ms[:, :-1].reshape(-1, D_)
    S_src = Ss_safe.reshape(-1, D_, D_)

    d_a = (ms[:, 1:] - ms[:, :-1])                                 # (K, T-1, D)
    # SING convention: SSs[t] = Cov(x_{t+1}, x_t).  Stein-correction
    # PDF uses Cov(x_t, x_{t+1}) = SSs[t].T.
    SSs_T = jnp.swapaxes(SSs, -1, -2)                              # (K, T-1, D, D)
    C_a = SSs_T - Ss[:, :-1]

    # Mask padded transitions
    d_a = d_a * tm_f[..., None]
    C_a = C_a * tm_f[..., None, None]
    d_src = d_a.reshape(-1, D_)
    C_src = C_a.reshape(-1, D_, D_)

    if del_t.ndim == 1:
        del_t_b = jnp.broadcast_to(del_t, (K_, T_ - 1))
    else:
        del_t_b = del_t                                            # (K, T-1)
    weights = (del_t_b * tm_f).reshape(-1)
    return m_src, S_src, d_src, C_src, weights


def qf_and_moments_gmix_jax(
    ms: Array,           # (K, T, D)
    Ss: Array,           # (K, T, D, D)
    SSs: Array,          # (K, T-1, D, D)
    del_t: Array,        # (K, T-1) or (T-1,) (broadcast)
    trial_mask: Array,   # (K, T) bool
    grid: jp.JaxGridState, *,
    sigma_drift_sq: float, D_lat: int, D_out: int,
    fine_N: int, stencil_r: int,
    cg_tol: float = 1e-5, max_cg_iter: int = 2000,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, Array, Array]:
    """Closed-form multi-trial variant of qf_and_moments_jax.

    K trials of length T (padded; ``trial_mask`` zeros padded slots).
    Returns ``(mu_r, Ef, Eff, Edfdx)`` with ``Ef: (K, T, D_out)``,
    ``Eff: (K, T)``, ``Edfdx: (K, T, D_out, D_lat)``.
    """
    m_src, S_src, d_src, C_src, weights = _flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, weights, grid,
        sigma_drift_sq=sigma_drift_sq, D_lat=D_lat, D_out=D_out,
        fine_N=fine_N, stencil_r=stencil_r,
        cg_tol=cg_tol, max_cg_iter=max_cg_iter,
    )
    Ef, Eff, Edfdx = drift_moments_jax(
        mu_r, grid, ms, D_lat=D_lat, D_out=D_out, nufft_eps=nufft_eps)
    return mu_r, Ef, Eff, Edfdx


def drift_moments_jax(
    mu_r: Array,                  # (D_out, M)
    grid: jp.JaxGridState,
    ms: Array,                    # (K, T, D_lat)  — multi-trial
    *,
    D_lat: int, D_out: int,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, Array]:
    """Posterior mean drift, second moment (delta-method), and Jacobian.

    Multi-trial: vmaps the per-trial NUFFT2 evaluation over the leading
    K axis of ``ms``.

    Returns (Ef, Eff, Edfdx) with shapes
    ``(K, T, D_out)``, ``(K, T)``, ``(K, T, D_out, D_lat)``.
    """
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    ws_real_c = grid.ws.real.astype(cdtype)
    ms_c = ms.astype(rdtype)

    # Mean coefficients (same across trials)
    fk_mean = ws_real_c[None, :] * mu_r                          # (D_out, M)

    # Jacobian coefficients per latent-input dim j
    def jac_fk_j(j):
        xi_j = grid.xis_flat[:, j].astype(cdtype)
        return ((2j * math.pi * xi_j)[None, :]
                * (ws_real_c[None, :] * mu_r))                    # (D_out, M)
    fk_jac = jax.vmap(jac_fk_j)(jnp.arange(D_lat))                # (D_lat, D_out, M)

    # Per-trial helper: do NUFFT2 of each fk-coefficient vector at the
    # trial's latent path m_t.
    def per_trial(ms_k):                                          # ms_k: (T, D)
        def eval_mean_r(fk):
            out = jp.nufft2(ms_k, fk.reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=nufft_eps)
            return out.real                                        # (T,)

        Ef_per_r = jax.vmap(eval_mean_r)(fk_mean)                 # (D_out, T)
        Ef_k = Ef_per_r.T                                         # (T, D_out)

        def eval_jac_j(fk_j_per_r):
            return jax.vmap(eval_mean_r)(fk_j_per_r)              # (D_out, T)

        Edfdx_dot = jax.vmap(eval_jac_j)(fk_jac)                  # (D_lat, D_out, T)
        Edfdx_k = jnp.transpose(Edfdx_dot, (2, 1, 0))              # (T, D_out, D_lat)
        Eff_k = (Ef_k * Ef_k).sum(axis=1)                         # (T,)
        return Ef_k, Eff_k, Edfdx_k

    Ef, Eff, Edfdx = jax.vmap(per_trial)(ms_c)                    # (K, ...)
    return Ef, Eff, Edfdx


def qf_and_moments_jax(
    ms: Array,           # (K, T, D)
    Ss: Array,           # (K, T, D, D)
    SSs: Array,          # (K, T-1, D, D)
    del_t: Array,        # (K, T-1) or (T-1,)
    trial_mask: Array,   # (K, T) bool
    grid: jp.JaxGridState, key: Array, *,
    sigma_drift_sq: float, S_marginal: int,
    D_lat: int, D_out: int,
    cg_tol: float = 1e-5, max_cg_iter: int = 2000,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, Array, Array]:
    """Multi-trial q(f) update + drift moment evaluation, all JAX, all jit-able.

    K trials of length T (padded; ``trial_mask`` zeros padded slots).
    Returns ``(mu_r, Ef, Eff, Edfdx)`` with ``Ef: (K, T, D_out)``,
    ``Eff: (K, T)``, ``Edfdx: (K, T, D_out, D_lat)``.
    """
    m_src, S_src, d_src, C_src, weights = _flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = compute_mu_r_jax(
        m_src, S_src, d_src, C_src, weights, grid, key,
        sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
        D_lat=D_lat, D_out=D_out,
        cg_tol=cg_tol, max_cg_iter=max_cg_iter, nufft_eps=nufft_eps,
    )
    Ef, Eff, Edfdx = drift_moments_jax(
        mu_r, grid, ms, D_lat=D_lat, D_out=D_out, nufft_eps=nufft_eps)
    return mu_r, Ef, Eff, Edfdx


# ---------------------------------------------------------------------------
# JAX kernel M-step (collapsed L^coll, MAP form with Hutchinson trace)
# ---------------------------------------------------------------------------
def _ws_real_se(log_ls: Array, log_var: Array, xis_flat: Array, h: Array,
                  D_lat: int) -> Array:
    """Differentiable spectral weights ws_k(θ) = sqrt(S(ξ_k) * h^d).

    Computed via log-space to avoid the sqrt(0) gradient singularity: at
    high-frequency modes the spectral density underflows to 0.0 in float32,
    and d/dx sqrt(x) = 1/(2 sqrt(x)) → inf/nan at x=0.  Using exp(log_ws)
    gives gradient = exp(log_ws) * d(log_ws)/dx = 0 * finite = 0.
    """
    import math as _math
    ls_sq = jnp.exp(2.0 * log_ls)
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_S = (D_lat / 2.0) * jnp.log(2 * _math.pi * ls_sq) + log_var \
            - 2 * (_math.pi ** 2) * ls_sq * xi_norm_sq
    log_ws = 0.5 * log_S + (D_lat / 2.0) * jnp.log(jnp.asarray(h, dtype=jnp.float32))
    return jnp.exp(log_ws)


def m_step_kernel_jax(
    log_ls0: float, log_var0: float,
    *,
    mu_r_fixed: Array,                 # ignored; kept for signature compat
    z_r: Array,                        # (D_out, M) — RHS summary, h_r = D ⊙ z_r
    top: jp.ToeplitzNDJax,             # cached BTTB on the current cloud + weights
    xis_flat: Array, h_per_dim: Array, # spectral grid
    D_lat: int, D_out: int,
    n_inner: int = 10, lr: float = 0.05,
    n_hutchinson: int = 4, include_trace: bool = True,        # ignored
    cg_tol_trace: float = 1e-4, max_cg_iter_trace: int = None, # ignored
    key: Array = None,                                         # ignored
) -> Tuple[Array, Array, list]:
    """Collapsed kernel M-step using EXACT dense Cholesky-based trace.

    Minimises ``-(Re(h*μ) − ½ μ* A μ) + ½ log|A| × D_out`` over
    ``(log_ℓ, log_σ²)`` with optax/Adam.  At each Adam step the full
    loss is computed via dense linear algebra on the explicit
    ``A = I + diag(ws) · T_mat · diag(ws)`` matrix, and differentiated
    by ``jax.grad`` — autodiff handles ``log|A|`` and the implicit
    ``μ = A⁻¹h`` (envelope theorem) automatically.

    A is hermitian positive-definite by construction, so we use a
    *single* Cholesky factorization per Adam step that gives BOTH
    ``log|A|`` (from ``2 Σ log L_ii``) AND ``μ = A⁻¹h`` (via two
    triangular solves).  Algebraic simplification: since ``Aμ = h``,
    we have ``⟨μ, Aμ⟩ = ⟨μ, h⟩``, so the deterministic loss collapses
    to ``-½ Σ_r ⟨h_r, μ_r⟩``.

    Cost per Adam step: O(M³/3) for Cholesky + O(D_out · M²) for the
    triangular solves.  One-time O(M² log M) build of ``T_mat`` outside
    the Adam loop.  Practical for M ≲ 2000-4000; for larger M, switch
    to a Hutchinson trace estimator with batched CG + enough probes
    (≥ 64) to control variance — see repo history for the previous
    Hutchinson code path, and ``diag_hutch_n_cgtol_sweep`` for evidence
    that ``n=64`` is needed to match exact-slogdet on this problem.

    Hutchinson kwargs (n_hutchinson, include_trace, cg_tol_trace,
    max_cg_iter_trace, key) are accepted for backwards compatibility
    but ignored.

    Returns (log_ls_new, log_var_new, loss_history).
    """
    import optax
    import jax.scipy.linalg as jla
    h_scalar = h_per_dim[0]                    # isotropic h in v0
    M = int(xis_flat.shape[0])
    cdtype = top.v_fft.dtype

    # Precompute dense T_mat once (top is fixed across the M-step) by
    # direct indexing into the BTTB conv vector — O(M²), avoiding the
    # M Toeplitz applies (O(M² log M)) the closure-form path would do.
    # Recover the conv vector v of shape (2m_i - 1) per dim from the
    # FFT-cached form via one O(M log M) inverse FFT + crop.
    eye_c = jnp.eye(M, dtype=cdtype)
    _v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    _ns_v = tuple(2 * n - 1 for n in top.ns)
    _v_conv = _v_pad[tuple(slice(0, L) for L in _ns_v)]
    _d = len(top.ns)
    _mi = jnp.indices(top.ns).reshape(_d, -1)        # (d, M)
    _offset = jnp.array([n - 1 for n in top.ns], dtype=jnp.int32)
    _diff = (_mi[:, :, None] - _mi[:, None, :]
             + _offset[:, None, None])               # (d, M, M)
    T_mat = _v_conv[tuple(_diff[k] for k in range(_d))]   # (M, M)

    def total_loss(log_ls, log_var):
        ws_real = _ws_real_se(log_ls, log_var, xis_flat, h_scalar, D_lat)
        ws_c = ws_real.astype(cdtype)
        # A = I + diag(ws) · T_mat · diag(ws) via row+col scaling (O(M²))
        A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
        # Single Cholesky factorization gives both log|A| and the solver.
        # A is hermitian PSD by construction (T_mat is BTTB from positive
        # weights → hermitian PSD; diagonal scaling preserves this).
        L = jnp.linalg.cholesky(A)
        # log|A| = 2 Σ log(L_ii) — diagonals are real positive in the
        # Cholesky convention.
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
        # μ = A⁻¹ h via forward + back triangular solve.
        h = ws_c[None, :] * z_r
        def solve_one(b):
            y = jla.solve_triangular(L, b, lower=True)
            return jla.solve_triangular(L.conj().T, y, lower=False)
        mu = jax.vmap(solve_one)(h)
        # Deterministic loss = -(Re⟨h, μ⟩ - ½ Re⟨μ, Aμ⟩) summed over r.
        # Since Aμ = h, ⟨μ, Aμ⟩ = ⟨μ, h⟩, so this collapses to
        # -½ Σ_r Re⟨h_r, μ_r⟩.
        det_loss = -0.5 * jnp.sum(
            jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
        return det_loss + 0.5 * D_out * logdet

    grad_fn = jax.jit(jax.value_and_grad(total_loss, argnums=(0, 1)))

    opt = optax.adam(lr)
    params = (jnp.asarray(log_ls0, dtype=jnp.float32),
              jnp.asarray(log_var0, dtype=jnp.float32))
    opt_state = opt.init(params)

    loss_history = []
    for step in range(n_inner):
        loss, (g_ls, g_var) = grad_fn(params[0], params[1])
        updates, opt_state = opt.update((g_ls, g_var), opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(float(loss))

    return params[0], params[1], loss_history
