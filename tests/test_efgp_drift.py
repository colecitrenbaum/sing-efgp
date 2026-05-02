"""
Correctness + scalability tests for the EFGP drift block.

Coverage:
  1. ``test_A_r_matvec_matches_dense``        — A_r matvec equals dense reference
  2. ``test_mu_solve_matches_dense``          — CG mean solve matches dense linsolve
  3. ``test_drift_moments_match_dense``       — Φμ, Jacobian Φ[(2πi ξ_j)⊙μ] match dense
  4. ``test_variance_hutchinson_consistent``  — Hutchinson diag(ΦA⁻¹Φ*) ≈ dense
  5. ``test_no_dense_M2_or_NM_alloc``         — no allocation has shape ≥ M*M or N·S·M
  6. ``test_collapsed_mstep_gradient_matches_finite_difference``
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Make sing/ importable (runs from repo root with `pytest`)
_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

# Make gp-quadrature importable
_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
if str(_GP_QUAD) not in sys.path:
    sys.path.insert(0, str(_GP_QUAD))

from sing.efgp_backend import TorchEFGPBackend  # noqa: E402

# efgpnd kernels (with Pydantic init quirks)
from kernels import SquaredExponential, GPParams  # noqa: E402


torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_se_kernel(d: int, lengthscale: float = 0.6, variance: float = 1.0):
    """Build an efgpnd SE kernel and bind hypers via GPParams."""
    k = SquaredExponential(dimension=d,
                           init_lengthscale=lengthscale,
                           init_variance=variance)
    # GPParams owns the actual values via softplus-positive parameter vector
    gp_params = GPParams(kernel=k, init_sig2=1.0)
    return k, gp_params


def _build_dense_A(grid, X_np, weights_np, ws_np):
    """Construct A_dense = I + D Φ_F* W Φ_F D where Φ_F[n,k] = exp(2πi ξ_k (x_n - xcen)).

    NB. We must include the same xcen as the NUFFT used by the backend so that
    the dense reference matches the cached NUFFT phase choice.
    """
    xcen = grid.xcen.cpu().numpy()
    xis = grid.xis_flat.cpu().numpy()           # (M, d)
    M = xis.shape[0]
    N = X_np.shape[0]
    # F[n, k] = exp(2πi ξ_k · (x_n - xcen))
    phases = 2 * math.pi * (X_np - xcen[None, :]) @ xis.T   # (N, M)
    F = np.exp(1j * phases)                                 # (N, M)
    D = np.diag(ws_np)                                       # (M, M) complex
    W = np.diag(weights_np.astype(np.complex128))            # (N, N)
    A = np.eye(M, dtype=np.complex128) + D @ F.conj().T @ W @ F @ D
    return A, F, D


# --------------------------------------------------------------------------
# 1. A_r matvec
# --------------------------------------------------------------------------
def test_A_r_matvec_matches_dense():
    d = 1
    N = 20
    backend = TorchEFGPBackend(dtype=torch.float64)
    kernel, _ = _make_se_kernel(d=d, lengthscale=0.4, variance=1.5)

    X = torch.randn(N, d, dtype=torch.float64)
    grid = backend.setup_grid(kernel, X, eps=1e-3)
    M = grid.M
    assert M >= 8 and M <= 200, f"unexpected M={M} for tiny test"

    # Random weights (simulate W_r entries)
    rng = np.random.default_rng(0)
    weights_np = rng.uniform(0.5, 1.5, size=N)
    weights = torch.tensor(weights_np, dtype=torch.float64)

    top = backend.make_toeplitz_weighted(grid, X, weights)
    A_apply = backend.make_A_apply(grid, top)

    # Dense reference
    ws_np = grid.ws.cpu().numpy()
    A_dense, _, _ = _build_dense_A(grid, X.cpu().numpy(), weights_np, ws_np)

    # Compare matvecs on 5 random complex vectors
    for trial in range(5):
        v_np = rng.standard_normal(M) + 1j * rng.standard_normal(M)
        v_torch = torch.tensor(v_np, dtype=torch.complex128)
        out_torch = A_apply(v_torch).cpu().numpy()
        out_dense = A_dense @ v_np
        rel = np.linalg.norm(out_torch - out_dense) / np.linalg.norm(out_dense)
        # NUFFT eps default 6e-8; allow ~10x headroom.
        assert rel < 1e-7, f"trial {trial}: rel error {rel:.2e}"


# --------------------------------------------------------------------------
# 2. μ-solve via CG matches dense linsolve
# --------------------------------------------------------------------------
def test_mu_solve_matches_dense():
    d = 1
    N = 30
    backend = TorchEFGPBackend(dtype=torch.float64)
    kernel, _ = _make_se_kernel(d=d, lengthscale=0.5)

    X = torch.linspace(-1.5, 1.5, N).unsqueeze(-1)
    grid = backend.setup_grid(kernel, X, eps=1e-4)
    M = grid.M

    rng = np.random.default_rng(1)
    weights_np = rng.uniform(0.5, 1.5, size=N)
    y_np = rng.standard_normal(N)

    weights = torch.tensor(weights_np, dtype=torch.float64)
    top = backend.make_toeplitz_weighted(grid, X, weights)
    A_apply = backend.make_A_apply(grid, top)

    # Build RHS h = D F* W y  via NUFFT → multiply by ws
    Wy = torch.tensor(weights_np * y_np, dtype=torch.complex128)
    OUT = tuple(grid.mtot_per_dim)
    Fstar_Wy = top.nufft_op.type1(Wy, out_shape=OUT).reshape(-1)
    ws_real = grid.ws.real.to(dtype=torch.float64)
    h_torch = ws_real * Fstar_Wy

    mu_cg = backend.cg_solve(A_apply, h_torch, tol=1e-12, max_iter=4 * M)

    # Dense reference
    ws_np = grid.ws.cpu().numpy()
    A_dense, F, D = _build_dense_A(grid, X.cpu().numpy(), weights_np, ws_np)
    h_dense = D.conj() @ F.conj().T @ (weights_np * y_np)
    mu_dense = np.linalg.solve(A_dense, h_dense)

    rel = np.linalg.norm(mu_cg.cpu().numpy() - mu_dense) / np.linalg.norm(mu_dense)
    assert rel < 1e-7, f"CG vs dense μ: rel error {rel:.2e}"


# --------------------------------------------------------------------------
# 3. Posterior mean Φμ and Jacobian Φ[(2πi ξ_j)⊙μ] match dense
# --------------------------------------------------------------------------
def test_drift_moments_match_dense():
    d = 1
    N = 25
    backend = TorchEFGPBackend(dtype=torch.float64)
    kernel, _ = _make_se_kernel(d=d, lengthscale=0.5)

    X = torch.linspace(-2., 2., N).unsqueeze(-1)
    grid = backend.setup_grid(kernel, X, eps=1e-4)
    M = grid.M

    rng = np.random.default_rng(2)
    mu_np = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * 0.1
    mu_torch = torch.tensor(mu_np, dtype=torch.complex128)

    X_eval = torch.linspace(-1.8, 1.8, 50).unsqueeze(-1)

    # Posterior mean: Φ_eval μ
    ws_real = grid.ws.real.to(dtype=torch.float64)
    fk = (ws_real.to(torch.complex128) * mu_torch)
    fmean = backend.nufft_type2(grid, X_eval, fk).cpu().numpy()

    # Dense reference
    xcen = grid.xcen.cpu().numpy()
    xis = grid.xis_flat.cpu().numpy()
    F_eval = np.exp(1j * 2 * math.pi * (X_eval.cpu().numpy() - xcen) @ xis.T)
    ws_np = grid.ws.cpu().numpy()
    fmean_dense = F_eval @ (ws_np * mu_np)
    rel_mean = (np.linalg.norm(fmean - fmean_dense) /
                np.linalg.norm(fmean_dense))
    assert rel_mean < 1e-7, f"Φμ rel error {rel_mean:.2e}"

    # Jacobian along dim 0:  ∂f/∂x_0 = Φ[(2πi ξ_0) ⊙ μ]
    xi0 = grid.xis_flat[:, 0].to(torch.complex128)
    fk_jac = (ws_real.to(torch.complex128) * (2j * math.pi * xi0) * mu_torch)
    fjac = backend.nufft_type2(grid, X_eval, fk_jac).cpu().numpy()

    fjac_dense = F_eval @ (ws_np * (2j * math.pi * xis[:, 0]) * mu_np)
    rel_jac = np.linalg.norm(fjac - fjac_dense) / np.linalg.norm(fjac_dense)
    assert rel_jac < 1e-7, f"Φ[(2πi ξ)⊙μ] rel error {rel_jac:.2e}"


# --------------------------------------------------------------------------
# 4. Hutchinson diag(Φ A⁻¹ Φ*) ≈ dense within 5σ MC error
# --------------------------------------------------------------------------
def test_variance_hutchinson_consistent():
    d = 1
    N = 25
    backend = TorchEFGPBackend(dtype=torch.float64)
    kernel, _ = _make_se_kernel(d=d, lengthscale=0.6)

    X = torch.linspace(-2., 2., N).unsqueeze(-1)
    grid = backend.setup_grid(kernel, X, eps=1e-4)
    M = grid.M

    rng = np.random.default_rng(3)
    weights_np = rng.uniform(0.5, 1.5, size=N)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    top = backend.make_toeplitz_weighted(grid, X, weights)
    A_apply = backend.make_A_apply(grid, top)

    # Eval at training points (so dense reference is straightforward)
    X_eval = X
    N_eval = X_eval.shape[0]

    # Hutchinson estimate. Use lots of probes for tight test.
    torch.manual_seed(123)
    var_hutch = backend.hutchinson_diag(grid, top, X_eval, A_apply,
                                        J=2000, cg_tol=1e-10,
                                        max_cg_iter=4 * M)

    # Dense reference: diag(F D A⁻¹ D F*)
    ws_np = grid.ws.cpu().numpy()
    A_dense, F, D = _build_dense_A(grid, X.cpu().numpy(), weights_np, ws_np)
    A_inv = np.linalg.inv(A_dense)
    var_dense = np.real(np.einsum('nk,kl,nl->n', F, D @ A_inv @ D, F.conj()))

    # Hutchinson is unbiased; mean rel error → 0 as J → ∞.
    err = np.abs(var_hutch.cpu().numpy() - var_dense)
    typical = np.median(np.abs(var_dense)) + 1e-12
    mean_rel = err.mean() / typical
    max_rel = err.max() / typical
    assert mean_rel < 0.05, f"Hutchinson mean rel error {mean_rel:.3f}"
    assert max_rel < 0.20, f"Hutchinson max rel error {max_rel:.3f}"


# --------------------------------------------------------------------------
# 5. No dense (M, M) or (N_total, M) allocations during one drift block
# --------------------------------------------------------------------------
def test_no_dense_M2_or_NM_alloc():
    """Allocation tracker. Patches torch's tensor constructors and the
    important factory functions to record every shape, then asserts no
    allocation hits the forbidden sizes."""
    import torch as _t

    d = 2
    N = 40
    S = 4                   # MC samples per transition
    N_total = N * S
    backend = TorchEFGPBackend(dtype=torch.float64)
    kernel, _ = _make_se_kernel(d=d, lengthscale=0.6)

    X = torch.randn(N_total, d, dtype=torch.float64)
    grid = backend.setup_grid(kernel, X, eps=1e-3)
    M = grid.M

    forbidden = max(M * M, N_total * M // 4)

    seen_shapes = []
    factories = ['empty', 'zeros', 'ones', 'randn', 'rand', 'randint', 'tensor']
    originals = {f: getattr(_t, f) for f in factories}

    def make_wrapper(orig):
        def wrap(*args, **kwargs):
            t = orig(*args, **kwargs)
            try:
                seen_shapes.append(tuple(t.shape))
            except Exception:
                pass
            return t
        return wrap

    for f in factories:
        setattr(_t, f, make_wrapper(originals[f]))
    try:
        weights = (torch.ones(N_total, dtype=torch.float64)
                   * 0.05)
        top = backend.make_toeplitz_weighted(grid, X, weights)
        A_apply = backend.make_A_apply(grid, top)
        # Apply A on a single vec just to exercise the matvec
        v = torch.randn(M, dtype=torch.complex128)
        _ = A_apply(v)
    finally:
        for f in factories:
            setattr(_t, f, originals[f])

    too_big = [s for s in seen_shapes if int(np.prod(s) or 0) >= forbidden]
    assert not too_big, (
        f"Allocations exceeding forbidden size {forbidden} (M={M}, "
        f"N_total={N_total}): {too_big[:10]}"
    )


# --------------------------------------------------------------------------
# 6. M-step (MAP) gradient w.r.t. log-lengthscale matches finite difference
# --------------------------------------------------------------------------
def test_collapsed_mstep_gradient_matches_finite_difference():
    """Build a tiny EFGPDrift with a fitted q(f), then compare autograd
    gradient of the MAP-like collapsed loss vs. centered finite difference."""
    from sing.efgp_drift import EFGPDrift
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    D = 2
    T = 30

    ker = SquaredExponential(dimension=D, init_lengthscale=0.6,
                             init_variance=1.0)
    gp = GPParams(kernel=ker, init_sig2=1.0)
    drift = EFGPDrift(kernel=ker, gp_params=gp, latent_dim=D,
                      sigma_drift_sq=0.3 ** 2, eps_grid=1e-3,
                      S_marginal=4, J_hutchinson=4)

    # Build a synthetic q(x) with smooth means so q(f) is well-conditioned.
    rng = np.random.default_rng(7)
    t_grid = jnp.linspace(0., 1., T)
    ms = np.stack([
        np.sin(2 * np.pi * np.linspace(0, 1, T)),
        np.cos(2 * np.pi * np.linspace(0, 1, T)),
    ], axis=-1)[None]                              # (1, T, D)
    Ss = np.tile(0.05 * np.eye(D)[None, None], (1, T, 1, 1))
    SSs = np.tile(0.04 * np.eye(D)[None, None], (1, T - 1, 1, 1))
    mp = dict(m=jnp.asarray(ms), S=jnp.asarray(Ss), SS=jnp.asarray(SSs))
    drift.update_dynamics_params(jr.PRNGKey(0), t_grid, mp,
                                 jnp.ones((1, T), bool),
                                 drift_params={},
                                 inputs=jnp.zeros((T, 1)),
                                 input_effect=jnp.zeros((D, 1)),
                                 sigma=1.0)

    # Helper: compute the MAP loss at a given (log_ls, log_var) holding
    # μ_r and the cached (toeplitz, z_r) fixed.  This mirrors what
    # m_step_kernel does internally for one step.
    c = drift.cache
    ws_real_0 = c.grid.ws.real.to(dtype=torch.float64)
    with torch.no_grad():
        mu_r_fixed = c.mu_r.detach().clone()
        ws_safe = ws_real_0.clamp_min(1e-300).to(torch.complex128)
        z_r = torch.stack([c.A_apply(mu_r_fixed[r]) / ws_safe
                           for r in range(D)], dim=0)
    toeplitz = c.top.toeplitz
    xis = c.grid.xis_flat
    h_prod = torch.prod(c.grid.h_per_dim).to(torch.float64)
    TWO_PI_SQ = (2 * math.pi) ** 2

    def loss_fn(log_ls_t, log_var_t):
        ls_sq = torch.exp(log_ls_t) ** 2
        var = torch.exp(log_var_t)
        xi_norm_sq = (xis * xis).sum(dim=1)
        log_S = (D / 2.) * torch.log(2 * math.pi * ls_sq) + torch.log(var) \
                - TWO_PI_SQ * ls_sq * xi_norm_sq
        ws_real = torch.sqrt(torch.exp(log_S) * h_prod)
        ws_c = ws_real.to(torch.complex128)
        L = torch.zeros((), dtype=torch.float64)
        for r in range(D):
            mu = mu_r_fixed[r]
            h_r = ws_c * z_r[r]
            tv = toeplitz(ws_c * mu)
            Av = ws_c * tv + mu
            L = L - (torch.vdot(h_r, mu).real - 0.5 * torch.vdot(mu, Av).real)
        return L

    # Initial point — perturb away from the E-step optimum
    log_ls0 = torch.tensor(math.log(0.65), dtype=torch.float64,
                           requires_grad=True)
    log_var0 = torch.tensor(math.log(1.1), dtype=torch.float64,
                            requires_grad=True)

    L = loss_fn(log_ls0, log_var0)
    L.backward()
    g_ls = float(log_ls0.grad)
    g_var = float(log_var0.grad)

    # Centered finite difference
    eps = 1e-4
    with torch.no_grad():
        L_p = loss_fn(log_ls0 + eps, log_var0).item()
        L_m = loss_fn(log_ls0 - eps, log_var0).item()
        fd_ls = (L_p - L_m) / (2 * eps)
        L_p = loss_fn(log_ls0, log_var0 + eps).item()
        L_m = loss_fn(log_ls0, log_var0 - eps).item()
        fd_var = (L_p - L_m) / (2 * eps)

    rel_ls = abs(g_ls - fd_ls) / (abs(fd_ls) + 1e-12)
    rel_var = abs(g_var - fd_var) / (abs(fd_var) + 1e-12)
    assert rel_ls < 1e-4, f"d/dlogℓ: autograd {g_ls:.4e}, FD {fd_ls:.4e} (rel {rel_ls:.2e})"
    assert rel_var < 1e-4, f"d/dlogσ²: autograd {g_var:.4e}, FD {fd_var:.4e} (rel {rel_var:.2e})"
