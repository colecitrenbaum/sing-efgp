"""
EFGP-based GP drift block for SING (latent SDE inference).

This module provides

* :class:`EFGPDrift` — a SING ``SDE`` subclass that *holds* the EFGP posterior
  ``q(w_r) = N(μ_r, A_r⁻¹)`` for every drift output dimension ``r``.  All
  numerics live in PyTorch (via :mod:`sing.efgp_backend`); JAX is used only
  for SING's natural-gradient updates on ``q(x)``.

* :class:`_FrozenEFGPDrift` — an internal SDE that wraps the *current* EFGP
  drift moments ``(Ef[t], Eff[t], Edfdx[t])`` as JAX constants so that
  SING's ``jax.grad`` through ``compute_neg_CE_single`` works without ever
  re-entering torch.  This is the standard "stop-gradient on the
  expectations" approximation; the gradient still picks up the explicit
  ``(m, S)`` terms of the cross-entropy.

The gateway between JAX and torch is intentionally narrow: ONE round trip
per E-step iteration produces three flat float64 tensors of shapes
``(T, D_out)``, ``(T,)``, ``(T, D_out, D_lat)``.  No ``pure_callback`` is
needed — we just call ``np.asarray`` on the torch outputs.

# TODO(efgp-jax): when a JAX-native EFGPBackend exists, the round-trips here
# disappear: ``EFGPDrift.f/ff/dfdx`` can be jit-traceable directly and the
# whole EM loop in :mod:`sing.efgp_em` can be wrapped in ``@jax.jit`` /
# ``lax.scan`` again, recovering SING's parallelism.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.sde import SDE
from sing.efgp_backend import EFGPBackend, GridState, ToeplitzOp, TorchEFGPBackend


# --------------------------------------------------------------------------
# Container for the cached q(f) state (torch side)
# --------------------------------------------------------------------------
@dataclass
class EFGPCache:
    """All torch state describing the current ``q(f)``.

    Stored as a single object so it can be swapped atomically when
    ``update_dynamics_params`` rebuilds ``q(f)``.
    """
    grid: GridState
    top: ToeplitzOp                 # BTTB on the most recent pseudo-input cloud
    A_apply: Any                    # Callable: torch matvec for A_r v
    mu_r: torch.Tensor              # (D_out, M) posterior mean (whitened weights)
    sigma_r_sq: torch.Tensor        # (D_out,) drift noise variance per dim (real)
    X_pseudo: torch.Tensor          # (T*S, d) the pseudo-input cloud
    weights: torch.Tensor           # (T*S,) BTTB weights actually used
    D_lat: int
    D_out: int


# --------------------------------------------------------------------------
# Frozen SDE used inside SING's natural-grad update
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Delta-method custom-VJPs: cached value + cached Jacobian-based gradient
# --------------------------------------------------------------------------
# Why these exist: SING's ``nat_grad_transition`` calls ``jax.grad`` through
# ``compute_neg_CE_single``, which calls ``fn.f``, ``fn.ff``, ``fn.dfdx``.
# A naive frozen value (returning a constant w.r.t. ``m``) gives a zero
# implicit gradient — biasing the q(x) update.  We *do* compute the
# Jacobian ``E[J_f(m)]`` (= ``Edfdx``) on the torch side; we just need to
# wire it into the JAX gradient via ``jax.custom_vjp``.  This implements the
# standard "delta-method" approximation:
#
#     E_q[f(x)] ≈ f(m)            ∂/∂m: ≈ J_f(m) = Edfdx
#     E_q[fᵀ Σ⁻¹ f] ≈ f(m)ᵀ Σ⁻¹ f(m)   ∂/∂m: ≈ 2 (Σ⁻¹ Ef) · Edfdx
#     E_q[J_f(x)] ≈ J_f(m)        treated constant (drops third moments)
#
# Note: dropping the ∂/∂S terms is also delta-method.  This is the same
# approximation used by most sparse-GP VI codes when an analytic E[J_f]
# is available.

@jax.custom_vjp
def _ef_with_jac_grad(m, ef_const, jac_const):
    """Returns ``ef_const`` (a (D_out,) value) but with VJP

           ∂L/∂m = jac_const.T @ ∂L/∂ef

    so SING's autodiff sees the Jacobian as ∂E[f]/∂m.
    """
    return ef_const


def _ef_fwd(m, ef_const, jac_const):
    return ef_const, (ef_const, jac_const)


def _ef_bwd(res, g):
    ef_const, jac_const = res
    # g has shape (D_out,); jac_const has shape (D_out, D_lat).
    grad_m = jac_const.T @ g
    return (grad_m, jnp.zeros_like(ef_const), jnp.zeros_like(jac_const))


_ef_with_jac_grad.defvjp(_ef_fwd, _ef_bwd)


@jax.custom_vjp
def _eff_with_grads(m, S, eff_const, ef_const, jac_const):
    """Returns scalar ``eff_const = E[||f||²]`` with a *local-quadratic*-
    transition-approximation VJP through both ``m`` and ``S``.

    Local quadratic expansion ``f(x) ≈ f(m) + J(x − m)`` gives

       E_q[||f(x)||²] ≈ ||f(m)||² + tr(J S Jᵀ)

    so

       ∂/∂m  ≈ 2 Jᵀ f(m)         (already encodes S=0 limit cleanly)
       ∂/∂S  ≈ Jᵀ J              (matrix grad; symmetric)

    The σ⁻² (diffusion) factor is applied separately by SING's
    ``compute_neg_CE_single`` via its outer ``sigma`` argument.

    We keep the cached scalar value (which can be more accurate than the
    quadratic surrogate when the Hutchinson variance is included), but use
    the quadratic-form gradients above — those are the principled local
    sensitivities of E[||f||²] under a Gaussian q(x).
    """
    del S
    return eff_const


def _eff_fwd(m, S, eff_const, ef_const, jac_const):
    return eff_const, (ef_const, jac_const)


def _eff_bwd(res, g):
    ef_const, jac_const = res
    grad_m = 2.0 * g * (ef_const @ jac_const)              # (D_lat,)
    grad_S = g * (jac_const.T @ jac_const)                  # (D_lat, D_lat)
    return (grad_m, grad_S,
            jnp.zeros(()),
            jnp.zeros_like(ef_const),
            jnp.zeros_like(jac_const))


_eff_with_grads.defvjp(_eff_fwd, _eff_bwd)


class _FrozenEFGPDrift(SDE):
    """SDE whose ``f/ff/dfdx`` return precomputed values **and** expose
    delta-method gradients through ``custom_vjp``.

    Built once per E-step inner iteration from ``EFGPDrift``'s cached
    moments at the *current* marginal means.  SING's natural-gradient
    update then auto-differentiates ``compute_neg_CE_single`` and gets:

      * the explicit ``(m, S)`` gradients of the cross-entropy, AND
      * the implicit ``∂E[f]/∂m`` and ``∂E[f^T Σ⁻¹ f]/∂m`` via the cached
        Jacobian (``Edfdx``) and mean (``Ef``).

    ``∂/∂S`` of the drift expectations is set to zero (delta-method).
    """

    def __init__(self, *, latent_dim: int, t_grid: jnp.ndarray,
                 Ef_per_t: jnp.ndarray, Eff_per_t: jnp.ndarray,
                 Edfdx_per_t: jnp.ndarray):
        super().__init__(expectation=None, latent_dim=latent_dim)
        self._t_grid = t_grid                      # (T,)
        self._Ef = Ef_per_t                        # (T, D_out)
        self._Eff = Eff_per_t                      # (T,)  = E[||f||²]
        self._Edfdx = Edfdx_per_t                  # (T, D_out, D_lat)

    def _idx(self, t):
        # SING calls with t = t_grid[i] exactly, so argmin is robust under vmap.
        return jnp.argmin(jnp.abs(self._t_grid - t))

    def drift(self, drift_params, x, t):
        return self._Ef[self._idx(t)]

    def f(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        ef = self._Ef[idx]
        jac = self._Edfdx[idx]
        return _ef_with_jac_grad(m, ef, jac)

    def ff(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        idx = self._idx(t)
        eff = self._Eff[idx]
        ef = self._Ef[idx]
        jac = self._Edfdx[idx]
        return _eff_with_grads(m, S, eff, ef, jac)

    def dfdx(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        # Delta-method: treat the Jacobian itself as constant w.r.t. (m, S).
        # Adding a (m,S) gradient here would require third-order moments.
        return self._Edfdx[self._idx(t)]


# --------------------------------------------------------------------------
# Main EFGP drift block
# --------------------------------------------------------------------------
class EFGPDrift(SDE):
    """SDE drift modelled by an independent GP per output coordinate, with
    EFGP (whitened-Fourier) posterior.

    Held state (torch-side):

      * ``self.kernel``  — efgpnd Kernel object (e.g. ``SquaredExponential``)
      * ``self.gp_params`` — efgpnd ``GPParams`` wrapping kernel hypers + σ²
      * ``self.cache: Optional[EFGPCache]`` — most recently fitted ``q(f)``;
        ``None`` until ``update_dynamics_params`` has been called once.

    Public API SING calls:

      * ``update_dynamics_params(...)`` — rebuilds the cached ``q(f)``.
      * ``f``, ``ff``, ``dfdx`` — single-step versions that look up the
        torch cache.  Inside the EM loop you would normally NOT use these
        directly: use :meth:`drift_moments_at_marginals` to get a batched
        ``(T, D_out)`` answer in a single round-trip, then build a
        ``_FrozenEFGPDrift`` to feed to SING's nat-grad routines.

    Constructor parameters
    ----------------------
    kernel : object
        efgpnd Kernel (must expose ``spectral_density``).
    gp_params : object
        efgpnd ``GPParams`` (provides the kernel hyper transforms).
    latent_dim : int
        ``D_lat = D_out`` for SING (drift maps R^D → R^D).
    sigma_drift_sq : float | torch.Tensor
        Per-output-dim drift noise variance σ_r² (PDF E-step §1.2).  Scalar
        means tied across r; (D_out,) tensor means per-dim.  This is NOT the
        observation noise.  In Euler form, ``Σ = sigma_drift_sq * I``.
    eps_grid : float
        EFGP spectral truncation tolerance (~1e-3 typical).
    cg_tol : float
        Relative residual tolerance for CG solves.
    nufft_eps : float
        NUFFT accuracy (default ``6e-8``).
    S_marginal : int
        Number of MC samples per transition for the q(x) cloud.
    J_hutchinson : int
        Hutchinson probes for variance estimation in ``ff`` and the M-step.
    backend : EFGPBackend
        Defaults to a fresh ``TorchEFGPBackend``.
    """

    def __init__(self, *, kernel, gp_params, latent_dim: int,
                 sigma_drift_sq: float | torch.Tensor = 1.0,
                 eps_grid: float = 1e-3, cg_tol: float = 1e-5,
                 nufft_eps: float = 6e-8,
                 S_marginal: int = 4, J_hutchinson: int = 16,
                 include_variance_in_ff: bool = False,
                 backend: Optional[EFGPBackend] = None):
        """include_variance_in_ff: if True, ``ff(m, S) = ||Ef||² + tr(Φ A⁻¹ Φ*)``
        with the trace estimated by ``J_hutchinson`` probes per time step
        (expensive — O(J × T) CG solves per E-step iter).  If False (default,
        v0), only the mean part ``||Ef||²`` is used.  The custom-VJP gradient
        through ``Eff`` is identical either way (it uses the cached Jacobian
        via the local-quadratic transition approximation), so the only effect
        is on the *value* of ``Eff``, not its sensitivity to ``(m, S)``.
        """
        super().__init__(expectation=None, latent_dim=latent_dim)
        self.kernel = kernel
        self.gp_params = gp_params
        self.D_lat = latent_dim
        self.D_out = latent_dim
        self.eps_grid = eps_grid
        self.cg_tol = cg_tol
        self.nufft_eps = nufft_eps
        self.S_marginal = int(S_marginal)
        self.J_hutchinson = int(J_hutchinson)
        self.include_variance_in_ff = bool(include_variance_in_ff)
        self.backend = backend or TorchEFGPBackend(nufft_eps=nufft_eps)

        sd = torch.as_tensor(sigma_drift_sq, dtype=self.backend.rdtype,
                             device=self.backend.device)
        if sd.ndim == 0:
            sd = sd.expand(self.D_out).clone()
        self.sigma_drift_sq = sd                    # (D_out,)

        self.cache: Optional[EFGPCache] = None
        # Grid cache: ``setup_grid`` does a kernel-truncation bisection that
        # accounts for ~40% of update_dynamics_params at small M.  We reuse
        # the grid across E-step iters whenever the kernel hypers haven't
        # changed and the X-cloud extent is similar.
        self._grid_cache_key = None
        self._grid_cache = None

    # ----------------------------------------------------------------
    # Required SDE abstract method
    # ----------------------------------------------------------------
    def drift(self, drift_params, x, t):
        """Posterior-mean drift evaluated at a single ``x``.

        Mostly used for rollouts/diagnostics; the EM loop uses
        ``drift_moments_at_marginals`` directly.
        """
        if self.cache is None:
            raise RuntimeError("EFGPDrift.cache is None — call "
                               "update_dynamics_params first.")
        x_jnp = jnp.atleast_1d(jnp.asarray(x))
        x_np = np.asarray(x_jnp)[None, :]           # (1, D_lat)
        x_torch = torch.tensor(x_np, dtype=self.backend.rdtype,
                               device=self.backend.device)
        f_mean = self._eval_mean_torch(x_torch).cpu().numpy()
        return jnp.asarray(f_mean[0])

    # ----------------------------------------------------------------
    # PUBLIC: per-call f/ff/dfdx — DELIBERATELY return JAX-pure constants.
    #
    # Why constants? SING's ``initialize_params`` (and other utilities like
    # ``linearize_prior``) call ``fn.f``/``fn.dfdx`` under ``vmap``+``jit``.
    # Doing a torch round-trip there would break tracing.  For the EM-loop
    # path we never use these; we use :meth:`drift_moments_at_marginals` to
    # build a ``_FrozenEFGPDrift`` whose ``f/ff/dfdx`` are JAX arrays with
    # custom-VJP gradients.
    #
    # The "constant" defaults we return (zero drift, zero Jacobian, scalar
    # ``trace(I)`` for ``ff``) make ``linearize_prior`` produce a
    # diffusion-only initialisation — equivalent to "no prior knowledge of
    # the drift", which SING refines in the first few E-step iterations.
    # ----------------------------------------------------------------
    def f(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        return jnp.zeros(self.D_out, dtype=jnp.asarray(m).dtype)

    def ff(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        return jnp.asarray(0.0, dtype=jnp.asarray(m).dtype)

    def dfdx(self, drift_params, key, t, m, S, gp_post=None, *args, **kwargs):
        return jnp.zeros((self.D_out, self.D_lat), dtype=jnp.asarray(m).dtype)

    # ----------------------------------------------------------------
    # PUBLIC: batched drift moments at all marginal means in one round-trip.
    # This is what the EM loop actually calls.
    # ----------------------------------------------------------------
    def drift_moments_at_marginals(self, ms_jax: jnp.ndarray,
                                   Ss_jax: jnp.ndarray
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate (Ef, Eff, Edfdx) at every (m_t, S_t).

        Parameters
        ----------
        ms_jax : (T, D_lat) jax array
        Ss_jax : (T, D_lat, D_lat) jax array

        Returns
        -------
        Ef     : (T, D_out)            jax array
        Eff    : (T,)                  jax array  (this is E[f^T Σ^{-1} f])
        Edfdx  : (T, D_out, D_lat)     jax array

        # TODO(efgp-jax): replace numpy round-trip with native JAX computation.
        """
        if self.cache is None:
            raise RuntimeError("update_dynamics_params has not been called.")
        ms_np = np.asarray(ms_jax)
        Ss_np = np.asarray(Ss_jax)

        ms_t = torch.tensor(ms_np, dtype=self.backend.rdtype,
                            device=self.backend.device)
        Ss_t = torch.tensor(Ss_np, dtype=self.backend.rdtype,
                            device=self.backend.device)

        Ef = self._eval_mean_torch(ms_t)                            # (T, D_out)
        Edfdx = self._eval_jacobian_torch(ms_t)                     # (T, D_out, D_lat)
        Eff = self._eval_second_moment_torch(ms_t, Ss_t)            # (T,)

        return (jnp.asarray(Ef.cpu().numpy()),
                jnp.asarray(Eff.cpu().numpy()),
                jnp.asarray(Edfdx.cpu().numpy()))

    # ----------------------------------------------------------------
    # PUBLIC: build the q(f) update (replaces SparseGP.update_dynamics_params)
    # ----------------------------------------------------------------
    def update_dynamics_params(self, key, t_grid, marginal_params, trial_mask,
                               drift_params, inputs, input_effect, sigma):
        """One full sweep of the EFGP collapsed q(f) update.

        Implements the algorithm in ``EFGP SING E step details.pdf §10`` /
        the M-step writeup §9.2 with Stein-corrected marginal sampling.

        Inputs are JAX arrays; output is a thin JAX-pytree-friendly dict
        (``GPPostEFGP``) so SING's existing book-keeping passes through.
        Heavy state lives in ``self.cache``.
        """
        del drift_params, sigma  # we own σ_drift via self.sigma_drift_sq

        # Convert JAX → numpy → torch.  marginal_params arrives with a
        # leading batch (trials) dim from `vmap`; we squeeze it for v0.
        ms = np.asarray(marginal_params['m'])                 # (B, T, D)
        Ss = np.asarray(marginal_params['S'])                 # (B, T, D, D)
        SSs = np.asarray(marginal_params['SS'])               # (B, T-1, D, D)
        if ms.ndim == 3:
            assert ms.shape[0] == 1, ("v0 supports a single trial; multi-"
                                       "trial would just stack along axis 0.")
            ms, Ss, SSs = ms[0], Ss[0], SSs[0]
        t_grid_np = np.asarray(t_grid)                        # (T,)
        del_t = (t_grid_np[1:] - t_grid_np[:-1])              # (T-1,)
        T = ms.shape[0]
        D = self.D_lat

        # 1) Stein quantities
        d_a = ms[1:] - ms[:-1]                                # (T-1, D)
        if input_effect is not None and inputs is not None:
            B = np.asarray(input_effect)                      # (D, n_inputs)
            v = np.asarray(inputs)                            # (T, n_inputs)
            d_a = d_a - del_t[:, None] * (v[:-1] @ B.T)
        C_a = SSs - Ss[:-1]                                   # (T-1, D, D)

        # 2) Sample marginal cloud x_a^(s) ~ N(m_a, S_a)
        #    Use only the first T-1 marginals (those participating in transitions).
        rng = np.random.default_rng(int(np.asarray(key).flatten()[0]) & 0xFFFFFFFF)
        S = self.S_marginal
        eps = rng.standard_normal((T - 1, S, D))              # (T-1, S, D)
        L_chol = np.linalg.cholesky(Ss[:-1] + 1e-9 * np.eye(D))  # (T-1, D, D)
        X_cloud = ms[:-1, None, :] + np.einsum('tdk,tsk->tsd',
                                               L_chol, eps)      # (T-1, S, D)
        N_eff = (T - 1) * S
        X_flat = X_cloud.reshape(N_eff, D)                    # (N_eff, D)
        # Per-pseudo-point Δ_a (replicated across S samples)
        delta_flat = np.repeat(del_t, S)                       # (N_eff,)

        # 3) Build the spectral grid once (depends only on kernel hypers + cloud)
        X_torch = torch.tensor(X_flat, dtype=self.backend.rdtype,
                               device=self.backend.device)
        # Cheap cache key: (kernel hypers, cloud per-dim extent rounded).
        # Across E-step iters within an EM iter, kernel hypers are unchanged
        # and the X-cloud extent barely moves, so this cache hits ~ every
        # iter except the very first / immediately after the M-step.
        cur_extent = tuple(np.round(X_flat.max(axis=0) - X_flat.min(axis=0),
                                    decimals=1).tolist())
        key_grid = (round(float(self.kernel.lengthscale), 6),
                    round(float(self.kernel.variance), 6),
                    cur_extent, self.eps_grid)
        if self._grid_cache_key == key_grid and self._grid_cache is not None:
            grid = self._grid_cache
        else:
            grid = self.backend.setup_grid(self.kernel, X_torch, self.eps_grid)
            self._grid_cache_key = key_grid
            self._grid_cache = grid
        M = grid.M

        sigma_r = self.sigma_drift_sq.cpu().numpy()           # (D_out,)
        D_out = self.D_out

        mu_r_all = torch.zeros((D_out, M), dtype=self.backend.cdtype,
                               device=self.backend.device)

        # For the v0 we assume σ²_r is shared across r — same Toeplitz reused.
        # If/when we want per-r σ², this loop builds D_out separate operators.
        # TODO: per-r σ² when the user actually needs it.
        if not np.allclose(sigma_r, sigma_r[0]):
            raise NotImplementedError(
                "v0 EFGPDrift expects tied σ_drift across output dims "
                f"(got {sigma_r}). Add a per-r Toeplitz loop here."
            )
        sigsq = float(sigma_r[0])

        # weights W_r entries: Δ_a / (S σ_r²)
        weights_np = delta_flat / (S * sigsq)                  # (N_eff,)
        weights = torch.tensor(weights_np, dtype=self.backend.rdtype,
                               device=self.backend.device)
        top = self.backend.make_toeplitz_weighted(grid, X_torch, weights)
        A_apply = self.backend.make_A_apply(grid, top)
        ws_real = grid.ws.real.to(dtype=self.backend.rdtype)

        # Build h_r for each output dim r in one loop (cheap).
        for r in range(D_out):
            # h_{1,r} = D F_X* a_r,  (a_r)_{a,s} = d_{a,r} / (S σ_r²)
            a_r = (np.repeat(d_a[:, r], S) / (S * sigsq))     # (N_eff,)
            a_r_t = torch.tensor(a_r, dtype=self.backend.cdtype,
                                 device=self.backend.device)
            h1 = ws_real * self.backend.nufft_type1(top, a_r_t).reshape(-1)

            # h_{2,r} = sum_j (2πi ξ_j) ⊙ D F_X* c_{j,r}
            #          (c_{j,r})_{a,s} = C_{a,j,r} / (S σ_r²)
            h2 = torch.zeros(M, dtype=self.backend.cdtype,
                             device=self.backend.device)
            for j in range(self.D_lat):
                c_jr = (np.repeat(C_a[:, j, r], S) / (S * sigsq))
                c_jr_t = torch.tensor(c_jr, dtype=self.backend.cdtype,
                                      device=self.backend.device)
                Fstar_c = self.backend.nufft_type1(top, c_jr_t).reshape(-1)
                xi_j = grid.xis_flat[:, j].to(self.backend.cdtype)
                h2 = h2 + (2j * math.pi * xi_j) * (ws_real * Fstar_c)

            h_r = (h1 + h2).to(self.backend.cdtype)
            # If the RHS is essentially zero (typical at iter-0 when the
            # variational posterior has m=0, S=I), CG would degenerate.
            # The maximizer of ½ h*A⁻¹h is then μ = A⁻¹·0 = 0.
            if float(torch.linalg.norm(h_r).real) < 1e-30:
                mu_r_all[r] = torch.zeros_like(h_r)
            else:
                mu_r_all[r] = self.backend.cg_solve(A_apply, h_r,
                                                   tol=self.cg_tol,
                                                   max_iter=4 * M)

        # Cache everything for the next E-step / M-step
        self.cache = EFGPCache(
            grid=grid, top=top, A_apply=A_apply,
            mu_r=mu_r_all, sigma_r_sq=self.sigma_drift_sq,
            X_pseudo=X_torch, weights=weights,
            D_lat=self.D_lat, D_out=self.D_out,
        )

        # SING expects a JAX-pytree-friendly object back; opaque payload.
        # GPPost is TypedDict but SING only reads from it; we pass a dict
        # whose actual content lives in self.cache.
        # TODO(efgp-jax): when q(f) is JAX-native this returns real arrays.
        return {'q_u_mu': jnp.zeros((D_out, 1)),
                'q_u_sigma': jnp.eye(1)[None].repeat(D_out, axis=0)}

    # ----------------------------------------------------------------
    # PRIVATE: torch-side moment evaluators (single round-trip)
    # ----------------------------------------------------------------
    def _eval_mean_torch(self, X_eval: torch.Tensor) -> torch.Tensor:
        """E[f_r(X_eval)] = Φ_eval μ_r for all r in one batched type-2 NUFFT.

        Returns shape (N_eval, D_out) real.
        """
        c = self.cache
        ws_real = c.grid.ws.real.to(dtype=self.backend.rdtype)
        # fk has shape (D_out, M)
        fk = (ws_real.to(self.backend.cdtype) * c.mu_r)
        OUT = tuple(c.grid.mtot_per_dim)
        # Build a fresh NUFFT for X_eval (it depends on the eval cloud only)
        from efgpnd import NUFFT
        nufft_eval = NUFFT(X_eval, c.grid.xcen, c.grid.h_per_dim,
                           eps=self.nufft_eps,
                           cdtype=self.backend.cdtype,
                           device=self.backend.device)
        # nufft.type2 with batched fk of shape (D_out, M) returns (D_out, N_eval)
        out = nufft_eval.type2(fk, out_shape=OUT)             # (D_out, N_eval)
        return out.real.transpose(0, 1).contiguous()           # (N_eval, D_out)

    def _eval_jacobian_torch(self, X_eval: torch.Tensor) -> torch.Tensor:
        """E[J_f(X_eval)]_{r,j} = Φ_eval [(2πi ξ_j) ⊙ μ_r]  for all r, j.

        Returns shape (N_eval, D_out, D_lat) real.
        """
        c = self.cache
        ws_real = c.grid.ws.real.to(dtype=self.backend.rdtype)
        ws_c = ws_real.to(self.backend.cdtype)
        OUT = tuple(c.grid.mtot_per_dim)

        from efgpnd import NUFFT
        nufft_eval = NUFFT(X_eval, c.grid.xcen, c.grid.h_per_dim,
                           eps=self.nufft_eps,
                           cdtype=self.backend.cdtype,
                           device=self.backend.device)
        N_eval = X_eval.shape[0]
        out = torch.zeros((N_eval, self.D_out, self.D_lat),
                          dtype=self.backend.rdtype,
                          device=self.backend.device)
        for j in range(self.D_lat):
            xi_j = c.grid.xis_flat[:, j].to(self.backend.cdtype)
            fk_j = (2j * math.pi * xi_j)[None, :] * (ws_c * c.mu_r)  # (D_out, M)
            jac_j = nufft_eval.type2(fk_j, out_shape=OUT)            # (D_out, N_eval)
            out[:, :, j] = jac_j.real.transpose(0, 1)
        return out

    def _eval_second_moment_torch(self, ms: torch.Tensor,
                                   Ss: torch.Tensor) -> torch.Tensor:
        """E[||f(m_t)||²]  ≈  Σ_r (Φμ_r)² + Σ_r diag(Φ A_r⁻¹ Φ*)_t.

        SING expects ``ff = E[||f||²]`` (no σ scaling — the diffusion factor
        is applied in ``compute_neg_CE_single`` via the outer ``sigma`` arg).
        ``Ss`` is unused: under the delta-method we evaluate at ``m`` only.

        Returns (T,) real.
        """
        del Ss
        c = self.cache
        f_mean = self._eval_mean_torch(ms)                     # (T, D_out)
        mean_part = f_mean.pow(2).sum(dim=1)                    # (T,)

        if not self.include_variance_in_ff:
            return mean_part

        # Variance part:  with A_r tied across r (v0), this is D_out × diag.
        # NB: O(J × T) CG solves; very expensive for big T.  Off by default.
        var_diag = self.backend.hutchinson_diag(
            c.grid, c.top, ms, c.A_apply,
            J=self.J_hutchinson, cg_tol=self.cg_tol, max_cg_iter=4 * c.grid.M,
        )                                                       # (T,)
        return mean_part + self.D_out * var_diag

    # ----------------------------------------------------------------
    # PUBLIC: collapsed kernel M-step (now WITH Hutchinson trace term)
    # ----------------------------------------------------------------
    def m_step_kernel(self, *, n_inner: int = 10, lr: float = 0.05,
                      n_hutchinson: int = 4, include_trace: bool = True,
                      verbose: bool = False) -> Dict[str, float]:
        """Update kernel hyperparameters via the collapsed-EFGP objective.

        Implements the MAP-like envelope-theorem gradient (M-step PDF §3.3
        applied at the current E-step optimum, dropping the Hutchinson trace
        term ``-½ tr[Q_r ∂_θ A_θ,r]``).  At the current ``q(f)`` optimum,

            ∂_θ L^coll(θ) ≈ Σ_r [ Re((∂_θ h_θ,r)* μ_r) − ½ μ_r* (∂_θ A_θ,r) μ_r ]

        with μ_r held fixed.  This is exact for the gradient flow of the GP
        prior contribution to the ELBO under MAP / fixed-q approximation.
        The dropped trace term encodes posterior-uncertainty pull-back; for
        the v0 demos this approximation suffices to recover the true
        lengthscale.  See the M-step PDF §6 for the full Hutchinson form.

        Currently supported kernel: the efgpnd ``SquaredExponential``.

        # TODO(efgp-jax): when JAX-native, do this with jax.grad through a
        # JAX-traceable EFGPBackend instead of autograd-through-torch.

        Parameters
        ----------
        n_inner : int
            Number of Adam steps.
        lr : float
            Adam learning rate (in log-hyper space).
        verbose : bool
            If True, prints loss at each step.

        Returns
        -------
        info : dict
            ``{'lengthscale': new ℓ, 'variance': new σ_var², 'loss_history': [...]}``.
        """
        if self.cache is None:
            raise RuntimeError("update_dynamics_params has not been called.")
        c = self.cache
        # Read current (positive-domain) hypers from GPParams.
        # GPParams uses softplus by default; we re-parameterize in log-space
        # for the M-step (more aggressive lr is safe in log-space).
        cur_ls = float(self.kernel.lengthscale)
        cur_var = float(self.kernel.variance)
        log_ls = torch.tensor(math.log(cur_ls), dtype=self.backend.rdtype,
                              requires_grad=True)
        log_var = torch.tensor(math.log(cur_var), dtype=self.backend.rdtype,
                               requires_grad=True)

        opt = torch.optim.Adam([log_ls, log_var], lr=lr)

        # Cache RHS NUFFT summaries z_r so we don't redo type-1 NUFFTs across
        # Adam steps.  z_r is the part of h_r that is θ-independent, since
        # h_θ,r = D_θ ⊙ z_r  with z_r = F_X* (W_r y_r + Stein_correction).
        # We re-derive z_r from the cached μ_r and the current A_apply
        # (h_r = A_r μ_r at the E-step optimum), but a cleaner alternative
        # would be to cache (a_r, c_{j,r}) directly inside update_dynamics_params.
        # For correctness here: at the E-step optimum, A_θ0,r μ_r = h_θ0,r,
        # and h_θ0,r = D_θ0 ⊙ z_r → z_r = (1 / D_θ0) ⊙ h_θ0,r = (1 / D_θ0) ⊙ A_θ0,r μ_r.
        ws_real_0 = c.grid.ws.real.to(dtype=self.backend.rdtype)
        with torch.no_grad():
            mu_r_fixed = c.mu_r.detach().clone()                # (D_out, M)
            # Compute h_θ0,r from cache and back out z_r.
            # batched A apply: c.A_apply expects single vector; loop r.
            z_r = torch.zeros_like(mu_r_fixed)
            ws_safe = ws_real_0.clamp_min(1e-300).to(self.backend.cdtype)
            for r in range(self.D_out):
                h_r0 = c.A_apply(mu_r_fixed[r])                  # (M,)
                z_r[r] = h_r0 / ws_safe                          # = F_X* (...)

        # The Toeplitz operator c.top is fixed (the cloud and weights don't
        # change during a kernel-only M-step). Only ws changes.
        toeplitz = c.top.toeplitz
        xis = c.grid.xis_flat                                   # (M, d) real
        h_prod = torch.prod(c.grid.h_per_dim).to(dtype=self.backend.rdtype)
        d_lat = self.D_lat
        M = c.grid.M
        TWO_PI_SQ = (2 * math.pi) ** 2

        loss_history = []
        for step in range(n_inner):
            opt.zero_grad()

            # Differentiable spectral weights ws(θ) — only the SE kernel for now.
            ls = torch.exp(log_ls)
            var = torch.exp(log_var)
            ls_sq = ls * ls
            xi_norm_sq = (xis * xis).sum(dim=1)                  # (M,) real
            log_S = (d_lat / 2.0) * torch.log(2 * math.pi * ls_sq) \
                    + torch.log(var) \
                    - TWO_PI_SQ * ls_sq * xi_norm_sq             # (M,) real
            ws_real = torch.sqrt(torch.exp(log_S) * h_prod)      # (M,) real

            # h_θ,r = D_θ ⊙ z_r   (D_θ enters as multiplication by ws_real)
            ws_c = ws_real.to(self.backend.cdtype)

            # Build A_θ,r matvec inline. We want autograd through ws_real.
            def A_apply_diff(v):
                tv = toeplitz(ws_c * v)
                return ws_c * tv + v

            # ---- Deterministic part of L^coll ---------------------------
            # L^coll = Σ_r [ Re(h*μ) − ½ μ* A μ − ½ log|A| ]
            # (we MAXIMIZE; loss to MINIMIZE is the negative)
            loss = torch.zeros((), dtype=self.backend.rdtype)
            for r in range(self.D_out):
                mu_r = mu_r_fixed[r]
                h_r = ws_c * z_r[r]
                term1 = torch.vdot(h_r, mu_r).real
                Av = A_apply_diff(mu_r)
                term2 = torch.vdot(mu_r, Av).real * 0.5
                loss = loss - (term1 - term2)
            loss.backward()

            # ---- Stochastic Hutchinson trace term -----------------------
            # ∂_θ (-½ log|A|) = -½ tr[A⁻¹ ∂_θ A]
            # Estimated via J Rademacher probes:
            #   tr[A⁻¹ ∂A] ≈ (1/J) Σ_v v* (A⁻¹ (∂A) v)
            # For each θ_p we compute the trace and add -½ × it × D_out
            # (factored: same A across r in v0).
            if include_trace:
                with torch.no_grad():
                    # Detach ws_real for the trace estimate (we add the gradient
                    # in log-θ space manually).
                    ws_real_d = ws_real.detach()
                    ws_c_d = ws_real_d.to(self.backend.cdtype)
                    A_apply_d = lambda v: ws_c_d * toeplitz(ws_c_d * v) + v

                    # ∂(log D)/∂(log ℓ) = ½ (d − 4π² ℓ² ||ξ||²)
                    # ∂D/∂(log ℓ) = D × ½ (d − 4π² ℓ² ||ξ||²)
                    # ∂(log D)/∂(log σ²) = ½  →  ∂D/∂(log σ²) = D × ½
                    ls_sq_val = float(ls_sq.detach())
                    dD_dlogls = ws_real_d * 0.5 * (
                        d_lat - 4 * math.pi * math.pi * ls_sq_val * xi_norm_sq
                    )
                    dD_dlogvar = ws_real_d * 0.5

                    # (∂A) v = (∂D) T (D v) + D T ((∂D) v)   for each θ_p
                    def dA_apply(dD, v):
                        Dv = ws_c_d * v
                        TDv = toeplitz(Dv)
                        first = dD.to(self.backend.cdtype) * TDv
                        second = ws_c_d * toeplitz(dD.to(self.backend.cdtype) * v)
                        return first + second

                    trace_ls = 0.0
                    trace_var = 0.0
                    for _ in range(n_hutchinson):
                        v_real = (torch.randint(
                            0, 2, (M,), device=self.backend.device,
                            dtype=self.backend.rdtype).mul_(2).sub_(1))
                        v = v_real.to(self.backend.cdtype)
                        # log ℓ
                        b_ls = dA_apply(dD_dlogls, v)
                        u_ls = self.backend.cg_solve(A_apply_d, b_ls,
                                                     tol=self.cg_tol * 10,
                                                     max_iter=2 * M)
                        trace_ls += float(torch.vdot(v, u_ls).real)
                        # log σ²
                        b_var = dA_apply(dD_dlogvar, v)
                        u_var = self.backend.cg_solve(A_apply_d, b_var,
                                                      tol=self.cg_tol * 10,
                                                      max_iter=2 * M)
                        trace_var += float(torch.vdot(v, u_var).real)
                    trace_ls /= n_hutchinson
                    trace_var /= n_hutchinson

                    # contribution to (negative) loss gradient:
                    # ∂(-(-½ log|A|))/∂θ × D_out = +½ × tr[A⁻¹ ∂A] × D_out
                    log_ls.grad = log_ls.grad + 0.5 * trace_ls * self.D_out
                    log_var.grad = log_var.grad + 0.5 * trace_var * self.D_out

            opt.step()
            loss_history.append(float(loss.detach().cpu()))
            if verbose:
                print(f"  m-step {step + 1}/{n_inner}: loss={loss.item():.4e}, "
                      f"ℓ={float(torch.exp(log_ls.detach())):.4f}, "
                      f"σ²={float(torch.exp(log_var.detach())):.4f}")

        # Push new hypers back into the kernel (so subsequent E-step uses them)
        new_ls = float(torch.exp(log_ls.detach()))
        new_var = float(torch.exp(log_var.detach()))
        self.kernel.set_hyper('lengthscale', new_ls)
        self.kernel.set_hyper('variance', new_var)

        return {
            'lengthscale': new_ls,
            'variance': new_var,
            'loss_history': loss_history,
        }
