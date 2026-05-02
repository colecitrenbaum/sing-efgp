"""
Pure-JAX q(f) update + drift-moment evaluation for SING-EFGP.

Mirrors :meth:`sing.efgp_drift.EFGPDrift.update_dynamics_params` and
:meth:`drift_moments_at_marginals` exactly, but in JAX so the resulting
``mu_r``, ``Ef``, ``Eff``, ``Edfdx`` are JAX arrays and the whole thing is
jit-traceable.

Once this is in place, :mod:`sing.efgp_em` can fold q(f) update + drift
moments INSIDE the existing jit'd ``_build_jit_estep`` and ``lax.scan`` the
entire inner E-step.  This is what closes the wall-time gap to SparseGP.

Math symbol → expression in this module
---------------------------------------

  Stein-corrected q(f) update     compute_mu_r_jax
  E[f(m)], E[J_f(m)] at marginals drift_moments_jax
  Combined "one inner-iter step"  qf_and_moments_jax
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
    step with delta-method custom VJPs through (m, S).  No torch deps.

    See :mod:`sing.efgp_drift` for the rationale (we cache the drift
    moments off-graph and expose only their values + first-order
    sensitivities to SING's natural-grad update).
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
    max_cg_iter: int = 200,
    nufft_eps: float = 6e-8,
) -> Tuple[Array, Array, jp.ToeplitzNDJax]:
    """Pure-JAX Stein-corrected q(f) update.  Returns (mu_r, X_flat, top).

    ``mu_r`` shape: (D_out, M).  Same math as
    :meth:`EFGPDrift.update_dynamics_params` for the no-inputs case with
    σ_r² tied across r.
    """
    T = ms.shape[0]
    M = grid.M
    cdtype = grid.ws.dtype

    # Stein quantities
    d_a = ms[1:] - ms[:-1]                                      # (T-1, D_lat)
    C_a = SSs - Ss[:-1]                                          # (T-1, D_lat, D_lat)

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

    # Build h_r per output dim r in a vmap
    def per_r(r):
        # h_1,r = D F_X* a_r,    (a_r)_a,s = d_{a,r} / (S σ²)
        a_r = jnp.repeat(d_a[:, r], S_marginal) / (S_marginal * sigma_drift_sq)
        Fstar_a = jp.nufft1(X_flat, a_r.astype(cdtype), grid.xcen,
                             grid.h_per_dim,
                             out_shape=grid.mtot_per_dim,
                             eps=nufft_eps).reshape(-1)
        h1 = ws_real_c * Fstar_a

        # h_2,r = Σ_j (2πi ξ_j) ⊙ D F_X* c_{j,r}
        def per_j(j):
            c_jr = jnp.repeat(C_a[:, j, r], S_marginal) \
                / (S_marginal * sigma_drift_sq)
            Fstar_c = jp.nufft1(X_flat, c_jr.astype(cdtype), grid.xcen,
                                 grid.h_per_dim,
                                 out_shape=grid.mtot_per_dim,
                                 eps=nufft_eps).reshape(-1)
            xi_j = grid.xis_flat[:, j].astype(cdtype)
            return (2j * math.pi * xi_j) * (ws_real_c * Fstar_c)

        h2 = jax.vmap(per_j)(jnp.arange(D_lat)).sum(axis=0)
        h_r = h1 + h2
        # CG solve.  Skip if RHS is essentially zero (initial iter).
        rhs_norm = jnp.linalg.norm(h_r).real
        mu = jax.lax.cond(
            rhs_norm < 1e-30,
            lambda _: jnp.zeros_like(h_r),
            lambda _: jp.cg_solve(A_apply, h_r, tol=cg_tol,
                                   max_iter=max_cg_iter),
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
    cg_tol: float = 1e-5, max_cg_iter: int = 200,
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
    """Differentiable spectral weights ws_k(θ) = sqrt(S(ξ_k) * h^d)."""
    import math as _math
    ls = jnp.exp(log_ls)
    var = jnp.exp(log_var)
    ls_sq = ls * ls
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_S = (D_lat / 2.0) * jnp.log(2 * _math.pi * ls_sq) + jnp.log(var) \
            - 2 * (_math.pi ** 2) * ls_sq * xi_norm_sq
    return jnp.sqrt(jnp.exp(log_S) * (h ** D_lat))


def m_step_kernel_jax(
    log_ls0: float, log_var0: float,
    *,
    mu_r_fixed: Array,                 # (D_out, M) — held fixed during M-step
    z_r: Array,                        # (D_out, M) — RHS summary, h_r = D ⊙ z_r
    top: jp.ToeplitzNDJax,             # cached BTTB on the current cloud + weights
    xis_flat: Array, h_per_dim: Array, # spectral grid
    D_lat: int, D_out: int,
    n_inner: int = 10, lr: float = 0.05,
    n_hutchinson: int = 4, include_trace: bool = True,
    cg_tol_trace: float = 1e-4, max_cg_iter_trace: int = None,
    key: Array = None,
) -> Tuple[Array, Array, list]:
    """JAX MAP-form collapsed kernel M-step.  Same math as
    :meth:`sing.efgp_drift.EFGPDrift.m_step_kernel`, but JAX/optax instead
    of torch.autograd/torch.optim.

    Returns (log_ls_new, log_var_new, loss_history).
    """
    import optax
    if key is None:
        key = jr.PRNGKey(0)
    h_scalar = h_per_dim[0]                    # isotropic h in v0
    M = int(xis_flat.shape[0])
    if max_cg_iter_trace is None:
        max_cg_iter_trace = 2 * M

    opt = optax.adam(lr)
    params = (jnp.asarray(log_ls0, dtype=jnp.float32),
              jnp.asarray(log_var0, dtype=jnp.float32))
    opt_state = opt.init(params)

    def deterministic_loss(params):
        log_ls, log_var = params
        ws = _ws_real_se(log_ls, log_var, xis_flat, h_scalar, D_lat)
        ws_c = ws.astype(top.v_fft.dtype)

        def per_r(r):
            mu = mu_r_fixed[r]
            h_r = ws_c * z_r[r]
            term1 = jnp.vdot(h_r, mu).real
            Tv = jp.toeplitz_apply(top, ws_c * mu)
            Av = ws_c * Tv + mu
            term2 = jnp.vdot(mu, Av).real * 0.5
            return -(term1 - term2)
        return jax.vmap(per_r)(jnp.arange(D_out)).sum()

    grad_det_fn = jax.jit(jax.value_and_grad(deterministic_loss))

    def hutchinson_trace_grads(params, key_h):
        """Estimate ∂_θ log|A| × D_out  via J Rademacher probes.

        The negative-log-det loss contribution is +½ log|A| × D_out per the
        sign convention used in :meth:`EFGPDrift.m_step_kernel` (we MINIMIZE
        the negative of L^coll which itself is the log-marginal-likelihood
        plus the negative of the log-det).
        """
        log_ls, log_var = params
        ws = _ws_real_se(log_ls, log_var, xis_flat, h_scalar, D_lat)
        ws_c = ws.astype(top.v_fft.dtype)

        # ∂(log ws²)/∂(log ℓ) = d - 4π² ℓ² ||ξ||²
        # ∂(log ws²)/∂(log σ²) = 1
        # ∂ws/∂θ = ws/2 × ∂(log ws²)/∂θ
        ls = jnp.exp(log_ls)
        ls_sq = ls * ls
        xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
        import math as _math
        dws_dlogls = ws * 0.5 * (D_lat - 4 * _math.pi * _math.pi * ls_sq * xi_norm_sq)
        dws_dlogvar = ws * 0.5

        def A_apply_d(v):
            tv = jp.toeplitz_apply(top, ws_c * v)
            return ws_c * tv + v

        def dA_apply(dD, v):
            Dv = ws_c * v
            TDv = jp.toeplitz_apply(top, Dv)
            first = dD.astype(top.v_fft.dtype) * TDv
            second = ws_c * jp.toeplitz_apply(top,
                                              dD.astype(top.v_fft.dtype) * v)
            return first + second

        def one_probe(key_p):
            v_real = (jax.random.bernoulli(key_p, shape=(M,))
                      .astype(jnp.float32) * 2.0 - 1.0)
            v = v_real.astype(top.v_fft.dtype)
            b_ls = dA_apply(dws_dlogls, v)
            u_ls = jp.cg_solve(A_apply_d, b_ls, tol=cg_tol_trace,
                               max_iter=max_cg_iter_trace)
            b_var = dA_apply(dws_dlogvar, v)
            u_var = jp.cg_solve(A_apply_d, b_var, tol=cg_tol_trace,
                                max_iter=max_cg_iter_trace)
            return jnp.array([jnp.vdot(v, u_ls).real,
                              jnp.vdot(v, u_var).real])

        keys = jr.split(key_h, n_hutchinson)
        probes = jax.vmap(one_probe)(keys)
        traces = probes.mean(axis=0)
        # Loss contribution: +½ tr[A^-1 ∂A] × D_out (per the docstring above)
        return 0.5 * traces[0] * D_out, 0.5 * traces[1] * D_out

    if include_trace:
        hutchinson_trace_grads_jit = jax.jit(hutchinson_trace_grads)

    loss_history = []
    for step in range(n_inner):
        loss, (g_ls, g_var) = grad_det_fn(params)
        if include_trace:
            key, sub = jr.split(key)
            tg_ls, tg_var = hutchinson_trace_grads_jit(params, sub)
            g_ls = g_ls + tg_ls
            g_var = g_var + tg_var
        grads = (g_ls, g_var)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(float(loss))

    return params[0], params[1], loss_history
