"""
Backend protocol for EFGP drift block.

Two implementations are envisioned:

* :class:`TorchEFGPBackend` (v0, in this file) wraps the PyTorch ``efgpnd``
  package from ``gp-quadrature``.  All operations run on torch tensors; the
  surrounding :mod:`sing.efgp_drift` module owns conversion to / from JAX.

* ``JaxEFGPBackend`` (TODO) — once a stable jax-native EFGP exists, drop in a
  second class implementing this same protocol and the rest of the pipeline
  works unchanged.  Look for ``# TODO(efgp-jax)`` markers throughout the
  EFGP-SING modules for the points where switching backend lets us re-jit
  the EM loop.

Math symbol → backend method:

    F_X* W_r F_X    (BTTB generator T_r)         → make_toeplitz_weighted
    A_r v = v + D_θ T_r D_θ v                    → make_A_apply
    A_r μ_r = h_r                                → cg_solve
    F_X* w  (type-1 NUFFT)                       → nufft_type1
    Φ_eval μ_r  (type-2 NUFFT, posterior mean)   → nufft_type2
    log|A_r|                                     → logdet_slq
    diag(Φ_eval A_r⁻¹ Φ_eval*)                   → hutchinson_diag
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Protocol, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------
# Import efgpnd from the gp-quadrature checkout. We do NOT modify efgpnd.py
# (per project constraint); we only call its public-ish primitives.
# --------------------------------------------------------------------------
_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
if str(_GP_QUAD) not in sys.path:
    sys.path.insert(0, str(_GP_QUAD))

from efgpnd import (  # noqa: E402
    NUFFT,
    ToeplitzND,
    create_A_mean,
    create_Gv,
    create_jacobi_precond,
    logdet_slq as _efgpnd_logdet_slq,
    _resolve_grid,
    _cmplx,
)
from cg import ConjugateGradients  # noqa: E402


# --------------------------------------------------------------------------
# Lightweight container objects (backend-agnostic in shape, backend-specific
# in array type).
# --------------------------------------------------------------------------
class GridState:
    """Output of ``setup_grid``: spectral / quadrature grid for one kernel."""

    def __init__(self, *, xis_flat, ws, h_per_dim, mtot_per_dim, xcen, kernel,
                 cdtype, rdtype, device):
        self.xis_flat = xis_flat            # (M, d) torch real tensor
        self.ws = ws                        # (M,)   torch complex tensor
        self.h_per_dim = h_per_dim          # (d,)   torch real tensor
        self.mtot_per_dim = mtot_per_dim    # list[int], length d
        self.xcen = xcen                    # (d,)   torch real tensor
        self.kernel = kernel                # backend's kernel object (for spectra)
        self.cdtype = cdtype
        self.rdtype = rdtype
        self.device = device
        self.M = int(xis_flat.shape[0])
        self.d = int(xis_flat.shape[1])


class ToeplitzOp:
    """Wraps a backend BTTB operator + the input cloud it was built from."""

    def __init__(self, *, toeplitz, X, weights, grid, nufft_op):
        self.toeplitz = toeplitz   # ToeplitzND
        self.X = X                 # (N_eff, d)
        self.weights = weights     # (N_eff,) real or complex
        self.grid = grid           # GridState
        self.nufft_op = nufft_op   # NUFFT for the same X and grid (reusable)
        self.N_eff = int(X.shape[0])


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------
class EFGPBackend(Protocol):
    """Abstract interface every EFGP backend implements."""

    # 1. Spectral grid construction (depends only on kernel + cloud extent + eps)
    def setup_grid(self, kernel, X, eps: float) -> GridState: ...

    # 2. BTTB generator T = F_X* diag(weights) F_X for a given pseudo-input cloud
    def make_toeplitz_weighted(self, grid: GridState, X, weights) -> ToeplitzOp: ...

    # 3. A_r matvec: v -> v + D_θ T D_θ v
    def make_A_apply(self, grid: GridState, top: ToeplitzOp) -> Callable: ...

    # 4. CG solve A_r μ = b
    def cg_solve(self, A_apply: Callable, b, *, tol: float, max_iter: int,
                 use_precond: bool) -> "torch.Tensor": ...

    # 5. NUFFTs (always with respect to a cached NUFFT op on a known X)
    def nufft_type1(self, top: ToeplitzOp, vals): ...
    def nufft_type2(self, grid: GridState, X_eval, fk_grid): ...

    # 6. log|A| via Stochastic Lanczos Quadrature (used in M-step)
    def logdet_A(self, grid: GridState, top: ToeplitzOp,
                 *, probes: int, steps: int) -> float: ...

    # 7. Hutchinson estimator for diag(Φ_eval A_r⁻¹ Φ_eval*)
    def hutchinson_diag(self, grid: GridState, top: ToeplitzOp, X_eval,
                        A_apply: Callable, *, J: int, cg_tol: float,
                        max_cg_iter: int) -> "torch.Tensor": ...


# --------------------------------------------------------------------------
# TorchEFGPBackend — wraps efgpnd
# --------------------------------------------------------------------------
class TorchEFGPBackend:
    """EFGP backend using ``efgpnd`` (PyTorch).

    All inputs are torch tensors; outputs are torch tensors.  Conversion to /
    from JAX is handled by :class:`sing.efgp_drift.EFGPDrift`.

    Note on weights: the BTTB generator T_r = F_X* W_r F_X already absorbs
    the per-output-dim scaling 1/σ_r².  Pass that as part of ``weights``;
    do NOT additionally pass σ²_r to ``make_A_apply``.
    """

    def __init__(self, *, nufft_eps: float = 6e-8, dtype: torch.dtype = torch.float64,
                 device: str = "cpu"):
        self.nufft_eps = nufft_eps
        self.rdtype = dtype
        self.cdtype = _cmplx(dtype)
        self.device = device

    # ------- 1. Spectral grid -------------------------------------------------
    def setup_grid(self, kernel, X: torch.Tensor, eps: float) -> GridState:
        """Build the equispaced Fourier grid from kernel + X extent.

        Parameters
        ----------
        kernel : object with ``spectral_density(xi)`` method (efgpnd Kernel).
        X      : (N, d) torch tensor — used only to set per-dim extent.
        eps    : spectral truncation tolerance.
        """
        X = X.to(device=self.device, dtype=self.rdtype)
        xis_1d_list, h_t, mtot_per_dim, xis_flat = _resolve_grid(
            kernel, X, eps, frozen_grid=None, max_mtot_1d=None,
            rdtype=self.rdtype, device=self.device,
        )

        # Spectral weights: ws_k = sqrt( S(ξ_k) * Π_dim h_k )
        spec = kernel.spectral_density(xis_flat).to(dtype=self.rdtype)
        h_prod = torch.prod(h_t)
        ws = torch.sqrt(spec * h_prod).to(dtype=self.cdtype)

        # Domain center (use cloud midpoint to keep NUFFT phases small)
        xcen = 0.5 * (X.max(dim=0).values + X.min(dim=0).values)

        return GridState(
            xis_flat=xis_flat, ws=ws,
            h_per_dim=h_t, mtot_per_dim=list(mtot_per_dim),
            xcen=xcen, kernel=kernel,
            cdtype=self.cdtype, rdtype=self.rdtype, device=self.device,
        )

    # ------- 2. BTTB generator T = F_X* W F_X --------------------------------
    def make_toeplitz_weighted(self, grid: GridState, X: torch.Tensor,
                               weights: torch.Tensor) -> ToeplitzOp:
        """Build the BTTB Toeplitz operator with explicit per-point weights.

        T_kl = Σ_n W_n exp(2πi (ξ_l - ξ_k)·x_n), evaluated on the shifted grid
        of frequency-differences (size 2*mtot - 1 per dim, packed as 4m+1 with
        m = (mtot-1)/2 to match efgpnd's convention).
        """
        X = X.to(device=self.device, dtype=self.rdtype)
        W = weights.to(device=self.device, dtype=self.cdtype)

        # m_per_dim = (mtot - 1) // 2 (mtot is odd by efgpnd construction)
        m_per_dim = [(m - 1) // 2 for m in grid.mtot_per_dim]
        OUT = tuple(4 * mi + 1 for mi in m_per_dim)

        # IMPORTANT: the Toeplitz conv-vector and the evaluation NUFFT must
        # share the same phase center (xcen). Otherwise the dense form of A_r
        # picks up a per-mode phase factor that breaks the BTTB property.
        nufft_T = NUFFT(X, grid.xcen, grid.h_per_dim, eps=self.nufft_eps,
                        cdtype=self.cdtype, device=self.device)
        v_kernel = nufft_T.type1(W, out_shape=OUT)

        toeplitz = ToeplitzND(v_kernel, force_pow2=False, precompute_fft=True)

        # Reuse the same NUFFT object for evaluation phases.
        nufft_eval = nufft_T

        return ToeplitzOp(toeplitz=toeplitz, X=X, weights=weights,
                          grid=grid, nufft_op=nufft_eval)

    # ------- 3. A_r matvec ----------------------------------------------------
    def make_A_apply(self, grid: GridState, top: ToeplitzOp) -> Callable:
        """Return the operator v -> v + D_θ T D_θ v.

        Note σ² = 1 is intentional: the per-output-dim 1/σ_r² already lives
        inside ``top.weights`` (pseudo-precision scaling).
        """
        ws_real = grid.ws.real.to(dtype=self.rdtype, device=self.device)
        return create_A_mean(ws_real, top.toeplitz, sigmasq_scalar=1.0,
                             cdtype=self.cdtype)

    # ------- 4. CG ------------------------------------------------------------
    def cg_solve(self, A_apply: Callable, b: torch.Tensor, *,
                 tol: float = 1e-5, max_iter: int = 200,
                 precond: Callable = None) -> torch.Tensor:
        # NB. b is complex in our setup (NUFFT outputs are complex)
        b = b.to(device=self.device, dtype=self.cdtype)
        x0 = torch.zeros_like(b)
        cg = ConjugateGradients(A_apply, b, x0, tol=tol, max_iter=max_iter,
                                early_stopping=True, M_inv_apply=precond)
        return cg.solve()

    def make_jacobi_precond(self, grid: GridState):
        """Build the cheap diagonal preconditioner |ws|² + 1 (since σ²=1)."""
        ws_real = grid.ws.real.to(dtype=self.rdtype, device=self.device)
        return create_jacobi_precond(ws_real, sigmasq_scalar=1.0)

    # ------- 5. NUFFTs --------------------------------------------------------
    def nufft_type1(self, top: ToeplitzOp, vals: torch.Tensor) -> torch.Tensor:
        """Type-1 NUFFT on the *training* cloud cached in ``top``.

        Returns the (mtot1 × ... × mtot_d)-shape grid coefficients; the caller
        is responsible for reshaping to (M,) when passing to A_apply.
        """
        OUT = tuple(top.grid.mtot_per_dim)
        return top.nufft_op.type1(vals.to(dtype=self.cdtype), out_shape=OUT)

    def nufft_type2(self, grid: GridState, X_eval: torch.Tensor,
                    fk_grid: torch.Tensor) -> torch.Tensor:
        """Type-2 NUFFT to a fresh cloud X_eval (typically m_t means).

        ``fk_grid`` is shape (M,) flat or (mtot1, ..., mtot_d) block.
        """
        X_eval = X_eval.to(device=self.device, dtype=self.rdtype)
        nufft_eval = NUFFT(X_eval, grid.xcen, grid.h_per_dim,
                           eps=self.nufft_eps, cdtype=self.cdtype,
                           device=self.device)
        OUT = tuple(grid.mtot_per_dim)
        if fk_grid.ndim == 1:
            return nufft_eval.type2(fk_grid.to(dtype=self.cdtype), out_shape=OUT)
        else:
            return nufft_eval.type2(fk_grid.to(dtype=self.cdtype), out_shape=OUT)

    # ------- 6. log|A| via SLQ ------------------------------------------------
    def logdet_A(self, grid: GridState, top: ToeplitzOp, *,
                 probes: int = 50, steps: int = 30) -> float:
        ws_real = grid.ws.real.to(dtype=self.rdtype, device=self.device)
        # efgpnd's logdet_slq computes log|I + σ⁻² D T D|; we want log|I + D T D|
        # so pass sigma2 = 1.  The trailing n*log(σ²) = 0 with σ²=1.
        return _efgpnd_logdet_slq(
            ws_real, sigma2=1.0, toeplitz=top.toeplitz,
            probes=probes, steps=steps,
            dtype=self.rdtype, device=self.device, n=top.N_eff,
        )

    # ------- 7. Hutchinson diag(Φ A⁻¹ Φ*) ------------------------------------
    def hutchinson_diag(self, grid: GridState, top: ToeplitzOp,
                        X_eval: torch.Tensor, A_apply: Callable, *,
                        J: int = 16, cg_tol: float = 1e-4,
                        max_cg_iter: int = 100) -> torch.Tensor:
        """Estimate diag(Φ_X_eval A⁻¹ Φ_X_eval*) by Rademacher probes.

        Each probe:
          z (Rademacher on X_eval) → Φ* z  (type-1 NUFFT on X_eval)
          → solve A u = Φ*z  via CG
          → Φ u             (type-2 NUFFT)
          accumulate z ⊙ Φu

        Returns a (N_eval,) real tensor.
        """
        X_eval = X_eval.to(device=self.device, dtype=self.rdtype)
        N_eval = X_eval.shape[0]
        OUT = tuple(grid.mtot_per_dim)
        nufft_eval = NUFFT(X_eval, grid.xcen, grid.h_per_dim,
                           eps=self.nufft_eps, cdtype=self.cdtype,
                           device=self.device)
        ws_real = grid.ws.real.to(dtype=self.rdtype, device=self.device)

        accum = torch.zeros(N_eval, dtype=self.rdtype, device=self.device)
        for _ in range(J):
            z = (torch.randint(0, 2, (N_eval,), device=self.device,
                               dtype=self.rdtype).mul_(2).sub_(1))
            z_c = z.to(self.cdtype)
            phi_star_z = nufft_eval.type1(z_c, out_shape=OUT).reshape(-1)
            # Solve A u = D phi_star_z (the diag(ws) factor turns Φ_eval into
            # the same coordinate system as A_r). Concretely we want
            # ws ⊙ A⁻¹ ⊙ ws acting on phi_star_z.
            rhs = ws_real * phi_star_z.real + 1j * ws_real * phi_star_z.imag
            rhs = rhs.to(self.cdtype)
            u = self.cg_solve(A_apply, rhs, tol=cg_tol,
                              max_iter=max_cg_iter)
            u_weighted = ws_real * u.real + 1j * ws_real * u.imag
            phi_u = nufft_eval.type2(u_weighted.to(self.cdtype),
                                     out_shape=OUT)
            accum += (z * phi_u.real)
        return accum / J


# --------------------------------------------------------------------------
# Convenience: a single global default backend (override per-test if needed)
# --------------------------------------------------------------------------
default_backend: EFGPBackend = TorchEFGPBackend()
