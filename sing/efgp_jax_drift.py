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
    """

    def __init__(self, *, latent_dim: int, t_grid: Array,
                 Ef_per_t: Array, Eff_per_t: Array, Edfdx_per_t: Array):
        super().__init__(expectation=None, latent_dim=latent_dim)
        self._t_grid = t_grid
        self._Ef = Ef_per_t
        self._Eff = Eff_per_t
        self._Edfdx = Edfdx_per_t

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
    ms: Array,           # (T, D_lat)
    Ss: Array,           # (T, D_lat, D_lat)
    key: Array,
    S_marginal: int,
    D_lat: int,
) -> Tuple[Array, Array]:
    """Sample the pseudo-input cloud x_a^(s) ~ N(m_a, S_a) for a=0..T-2.

    Returns (X_flat, delta_unit_per_pseudo) where:
      * X_flat:  (T-1)*S_marginal x D_lat
      * delta_unit_per_pseudo: (T-1)*S_marginal x 1, replicated del_t
        weights (callers multiply by del_t / (S σ²)).
    """
    T = ms.shape[0]
    # eps shape (T-1, S, D_lat)
    eps = jax.random.normal(key, (T - 1, S_marginal, D_lat))
    L_chol = jnp.linalg.cholesky(Ss[:-1] + 1e-9 * jnp.eye(D_lat))
    X_cloud = ms[:-1, None, :] + jnp.einsum('tij,tsj->tsi', L_chol, eps)
    X_flat = X_cloud.reshape(-1, D_lat)
    return X_flat


def compute_mu_r_jax(
    ms: Array,                        # (T, D_lat)
    Ss: Array,                        # (T, D_lat, D_lat)
    SSs: Array,                       # (T-1, D_lat, D_lat)  (cross-cov)
    del_t: Array,                     # (T-1,)
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

    ``mu_r`` shape: (D_out, M).  Implements the no-inputs case of the
    EFGP q(f) update with σ_r² tied across output dims (v0).
    """
    T = ms.shape[0]
    M = grid.M
    cdtype = grid.ws.dtype

    # Stein quantities
    d_a = ms[1:] - ms[:-1]                                      # (T-1, D_lat)
    # SING convention: SSs[t] = Cov(x_{t+1}, x_t) (see sing/utils/sing_helpers.py
    # and the docstring in compute_neg_CE_single).  The Stein-correction PDF
    # uses S_{i, i+1} = Cov(x_t, x_{t+1}) = SSs[t].T.  In practice SSs is
    # near-symmetric for the smoothed posterior so the transpose is a small
    # effect, but we apply it for theoretical correctness.
    SSs_T = jnp.transpose(SSs, (0, 2, 1))                       # = Cov(x_t, x_{t+1})
    C_a = SSs_T - Ss[:-1]                                       # (T-1, D_lat, D_lat)

    # Pseudo-input cloud
    X_flat = _build_pseudo_cloud(ms, Ss, key, S_marginal, D_lat)
    delta_per_pseudo = jnp.repeat(del_t, S_marginal)             # ((T-1)*S,)

    # BTTB Toeplitz
    weights = delta_per_pseudo / (S_marginal * sigma_drift_sq)
    v_kernel = jp.bttb_conv_vec_weighted(
        X_flat, weights.astype(cdtype), grid.xcen, grid.h_per_dim,
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

    # Build h_r per output dim r in a vmap
    def per_r(r):
        # h_1,r = D F_X* a_r,    (a_r)_a,s = d_{a,r} / (S σ²)
        a_r = jnp.repeat(d_a[:, r], S_marginal) / (S_marginal * sigma_drift_sq)
        Fstar_a = jp.nufft1(X_flat, a_r.astype(cdtype), grid.xcen,
                             grid.h_per_dim,
                             out_shape=grid.mtot_per_dim,
                             eps=nufft_eps).reshape(-1)
        h1 = ws_real_c * Fstar_a

        # h_2,r = Σ_j conj(2πi ξ_j) ⊙ D F_X* c_{j,r} = Σ_j (-2πi ξ_j) ⊙ ...
        # The factor comes from J_φ^* (adjoint Jacobian) — conjugate of
        # ∂φ_k/∂x_j = 2πi ξ_{kj} φ_k, so (-2πi ξ).
        def per_j(j):
            c_jr = jnp.repeat(C_a[:, j, r], S_marginal) \
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


def drift_moments_jax(
    mu_r: Array,                  # (D_out, M)
    grid: jp.JaxGridState,
    ms: Array,                    # (T, D_lat)
    *,
    D_lat: int, D_out: int,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, Array]:
    """Posterior mean drift, second moment (delta-method), and Jacobian.

    Returns (Ef, Eff, Edfdx) with shapes (T, D_out), (T,), (T, D_out, D_lat).
    """
    T = ms.shape[0]
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    ws_real_c = grid.ws.real.astype(cdtype)
    ms_c = ms.astype(rdtype)

    # ------- mean: Φ(m_t) μ_r per r -------
    fk_mean = ws_real_c[None, :] * mu_r                          # (D_out, M)

    def eval_mean_r(fk):
        out = jp.nufft2(ms_c, fk.reshape(*grid.mtot_per_dim),
                        grid.xcen, grid.h_per_dim, eps=nufft_eps)
        return out.real                                            # (T,)

    Ef_per_r = jax.vmap(eval_mean_r)(fk_mean)                     # (D_out, T)
    Ef = Ef_per_r.T                                               # (T, D_out)

    # ------- Jacobian: Φ(m_t) [(2πi ξ_j) ⊙ μ_r] per (r, j) -------
    def eval_jac_j(j):
        xi_j = grid.xis_flat[:, j].astype(cdtype)
        fk_j = ((2j * math.pi * xi_j)[None, :]
                * (ws_real_c[None, :] * mu_r))                    # (D_out, M)

        def eval_per_r(fk):
            out = jp.nufft2(ms_c, fk.reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=nufft_eps)
            return out.real
        return jax.vmap(eval_per_r)(fk_j)                          # (D_out, T)

    Edfdx_dot = jax.vmap(eval_jac_j)(jnp.arange(D_lat))            # (D_lat, D_out, T)
    Edfdx = jnp.transpose(Edfdx_dot, (2, 1, 0))                     # (T, D_out, D_lat)

    # ------- Eff (delta-method): Σ_r f_r(m)² -------
    Eff = (Ef * Ef).sum(axis=1)                                    # (T,)

    return Ef, Eff, Edfdx


def qf_and_moments_jax(
    ms: Array, Ss: Array, SSs: Array, del_t: Array,
    grid: jp.JaxGridState, key: Array, *,
    sigma_drift_sq: float, S_marginal: int,
    D_lat: int, D_out: int,
    cg_tol: float = 1e-5, max_cg_iter: int = 2000,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, Array, Array]:
    """One full q(f) update + drift moment evaluation, all JAX, all jit-able.

    Returns (mu_r, Ef, Eff, Edfdx).
    """
    mu_r, _, _ = compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, key,
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
