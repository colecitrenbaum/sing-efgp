"""
JAX-native EFGP primitives.

Mirrors the PyTorch primitives used by ``TorchEFGPBackend`` (which wrap
``efgpnd``).  Once these pass parity + timing tests, a ``JaxEFGPBackend``
in :mod:`sing.efgp_backend` will use them so the entire SING E-step can
be ``jax.jit`` + ``lax.scan``'d.

Import-order constraint
-----------------------
``pytorch_finufft`` (used by ``efgpnd``) and ``jax_finufft`` both wrap
the same upstream ``libfinufft``.  If torch's binding is initialised
first, the next ``jax_finufft`` call segfaults the process.  We force
``jax_finufft`` to load *first* by importing it at the top of this
module — but anyone running torch + JAX together in the same process
must therefore import this module *before* importing ``efgpnd`` /
``pytorch_finufft``.  The parity tests work around this by running
torch and JAX in separate subprocesses.

Math symbol → function in this module
-------------------------------------

    Spectral grid  (xis, ws, h, mtot, xcen)         spectral_grid_se
    F_X* (type-1 NUFFT)                             nufft1
    F_X  (type-2 NUFFT)                             nufft2
    BTTB conv vector  Σ_n W_n e^{-2πi ξ x_n}        bttb_conv_vec_weighted
    Toeplitz matvec  T x  via FFT                   make_toeplitz / toeplitz_apply
    A_r v = v + D_θ T D_θ v                         make_A_apply
    Solve A μ = h via CG                            cg_solve
    diag(Φ A⁻¹ Φ*)  via Hutchinson                   hutchinson_diag

# TODO(efgp-jax): this module is the single point of integration with
# JAX-native EFGP.  When efgp_jax (the sibling package) becomes the
# canonical backend, swap the imports below to point at it directly.
"""
from __future__ import annotations

# IMPORTANT: import jax_finufft BEFORE anything that might trigger a
# ``pytorch_finufft`` / libfinufft initialisation in the same process.
import jax_finufft  # noqa: E402, F401

import math
from math import prod
from typing import Callable, NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array


# ---------------------------------------------------------------------------
# Dtype helper
# ---------------------------------------------------------------------------
def _cmplx(real_dtype):
    if real_dtype in (jnp.float32, jnp.complex64):
        return jnp.complex64
    return jnp.complex128


# ---------------------------------------------------------------------------
# 1. Spectral grid for SE kernel
# ---------------------------------------------------------------------------
class JaxGridState(NamedTuple):
    """JAX-side analogue of :class:`sing.efgp_backend.GridState`.

    Fields are JAX arrays where it matters for tracing (xis_flat, ws,
    h_per_dim, xcen) and Python ints elsewhere (mtot_per_dim, M, d).
    Carry it through jit'd functions as a pytree.
    """
    xis_flat: Array        # (M, d) real
    ws: Array              # (M,)   complex
    h_per_dim: Array       # (d,)   real
    mtot_per_dim: Tuple[int, ...]
    xcen: Array            # (d,)   real
    M: int
    d: int


def _bisect_decreasing(f, eps, *, b0=1000.0, max_iter=200, n_bracket=10):
    """Find L such that f(L) ≈ eps for a monotonically-decreasing f.

    Plain numpy / Python — runs at grid-build time, NOT inside any JAX
    trace.  Mirrors ``efgpnd.utils.kernels.GetTruncationBound``.
    """
    b = float(b0)
    for _ in range(n_bracket):
        if f(b) > eps:
            b *= 2
        else:
            break
    a = 0.0
    mid = 0.5 * (a + b)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if f(mid) > eps:
            a = mid
        else:
            b = mid
    return mid


def spectral_grid_se(
    lengthscale: float,
    variance: float,
    X: Array,
    *,
    eps: float = 1e-3,
    dtype=jnp.float32,
) -> JaxGridState:
    """Build the EFGP spectral grid for an SE kernel + cloud extent ``X``.

    Uses the same bisection-based truncation as
    ``efgpnd._resolve_grid(use_integral=True)`` for shape parity.  The
    bisection runs in plain numpy (not under JAX tracing); the resulting
    ``mtot_per_dim`` is a Python tuple of ints, fixed at trace time.
    """
    d = int(X.shape[1])
    cdtype = _cmplx(dtype)

    # Per-dim spatial extent → max difference within cloud
    X_min = np.asarray(X.min(axis=0))
    X_max = np.asarray(X.max(axis=0))
    L = float((X_max - X_min).max())
    L = max(L, 1.0)

    ls = float(lengthscale)
    var = float(variance)

    # Spatial truncation: SE kernel k(r) = var * exp(-0.5 r²/ℓ²)
    def k_se(r):
        return var * math.exp(-0.5 * r * r / (ls * ls))

    Ltime = _bisect_decreasing(k_se, eps)
    h = 1.0 / (L + Ltime)

    # Frequency truncation: SE spectral density (1-D, isotropic radius proxy)
    # S(ξ) = (2π ℓ²)^{d/2} var exp(-2π² ℓ² ξ²)
    s0 = (2 * math.pi * ls * ls) ** (d / 2.0) * var

    def khat_modified(r):
        s = (2 * math.pi * ls * ls) ** (d / 2.0) * var \
            * math.exp(-2 * math.pi * math.pi * ls * ls * r * r)
        return abs(r) ** (d - 1) * s / s0

    Lfreq = _bisect_decreasing(khat_modified, eps)
    hm = math.ceil(Lfreq / h)
    mtot_per_dim = tuple(2 * hm + 1 for _ in range(d))
    M = int(np.prod(mtot_per_dim))

    # 1-D ξ grid, then tensor product → (M, d)
    xis_1d = jnp.arange(-hm, hm + 1, dtype=dtype) * jnp.asarray(h, dtype=dtype)
    if d == 1:
        xis_flat = xis_1d.reshape(-1, 1)
    else:
        grids = jnp.meshgrid(*[xis_1d for _ in range(d)], indexing="ij")
        xis_flat = jnp.stack([g.ravel() for g in grids], axis=-1)

    # SE spectral density: S(ξ) = (2π ℓ²)^{d/2} variance exp(-2π² ℓ² |ξ|²)
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_S = (d / 2.0) * math.log(2 * math.pi * ls * ls) \
            + math.log(var) \
            - 2 * (math.pi ** 2) * ls * ls * xi_norm_sq
    h_d = h ** d
    ws_real = jnp.sqrt(jnp.exp(log_S) * h_d)
    ws = ws_real.astype(cdtype)

    h_per_dim = jnp.full((d,), h, dtype=dtype)
    xcen = 0.5 * jnp.asarray(X_min + X_max, dtype=dtype)

    return JaxGridState(xis_flat=xis_flat, ws=ws, h_per_dim=h_per_dim,
                        mtot_per_dim=mtot_per_dim, xcen=xcen, M=M, d=d)


# ---------------------------------------------------------------------------
# 2-3. NUFFT wrappers (match efgpnd's iflag / modeord conventions exactly)
# ---------------------------------------------------------------------------
def _make_phi(X: Array, xcen: Array, h_per_dim: Array) -> Tuple[Array, ...]:
    """Per-dim phase coords  φ_k[n] = 2π h_k (x_n - xcen)_k.

    Returns a tuple (φ_1, ..., φ_d), one (N,) array per dim, matching
    jax_finufft's expected signature.
    """
    coords = 2 * math.pi * h_per_dim[None, :] * (X - xcen[None, :])
    return tuple(coords[:, i] for i in range(coords.shape[1]))


def nufft1(X: Array, vals: Array, xcen: Array, h_per_dim: Array,
           out_shape: Tuple[int, ...], eps: float = 6e-8) -> Array:
    """Type-1 NUFFT (nonuniform → uniform), iflag=-1.

    f_k = Σ_n vals_n exp(-2πi ξ_k · x_n)  on the natural-order frequency grid.
    """
    phi = _make_phi(X, xcen, h_per_dim)
    cdtype = _cmplx(vals.dtype)
    if not jnp.iscomplexobj(vals):
        vals = vals.astype(cdtype)
    d = len(phi)
    if d == 1:
        return jax_finufft.nufft1(out_shape[0], vals, phi[0],
                                   eps=eps, iflag=-1)
    elif d == 2:
        return jax_finufft.nufft1(out_shape, vals, phi[0], phi[1],
                                   eps=eps, iflag=-1)
    elif d == 3:
        return jax_finufft.nufft1(out_shape, vals, phi[0], phi[1], phi[2],
                                   eps=eps, iflag=-1)
    raise ValueError(f"unsupported d={d}")


def nufft2(X: Array, fk_grid: Array, xcen: Array, h_per_dim: Array,
           eps: float = 6e-8) -> Array:
    """Type-2 NUFFT (uniform → nonuniform), iflag=+1.

    c_n = Σ_k fk_grid[k] exp(+2πi ξ_k · x_n).  ``fk_grid`` shape is
    ``(m1, ..., md)``.
    """
    phi = _make_phi(X, xcen, h_per_dim)
    cdtype = _cmplx(fk_grid.dtype)
    if not jnp.iscomplexobj(fk_grid):
        fk_grid = fk_grid.astype(cdtype)
    d = len(phi)
    if d == 1:
        return jax_finufft.nufft2(fk_grid, phi[0], eps=eps, iflag=1)
    elif d == 2:
        return jax_finufft.nufft2(fk_grid, phi[0], phi[1], eps=eps, iflag=1)
    elif d == 3:
        return jax_finufft.nufft2(fk_grid, phi[0], phi[1], phi[2],
                                   eps=eps, iflag=1)
    raise ValueError(f"unsupported d={d}")


# ---------------------------------------------------------------------------
# 4. Weighted BTTB convolution vector
# ---------------------------------------------------------------------------
def bttb_conv_vec_weighted(X: Array, weights: Array, xcen: Array,
                            h_per_dim: Array, mtot_per_dim: Tuple[int, ...],
                            eps: float = 6e-8) -> Array:
    """Build the convolution vector for the BTTB operator T = F_X* W F_X.

    Output grid has size ``(4m+1)`` per dim, where ``m = (mtot-1)//2``,
    matching ``efgpnd.compute_convolution_vector_vectorized_dD``.

    Returns a complex array of shape ``(4m_1+1, ..., 4m_d+1)``.
    """
    if not jnp.iscomplexobj(weights):
        weights = weights.astype(_cmplx(weights.dtype))
    out_shape = tuple(2 * (m - 1) + 1 for m in mtot_per_dim)  # = 4*((m-1)/2)+1
    return nufft1(X, weights, xcen, h_per_dim, out_shape=out_shape, eps=eps)


# ---------------------------------------------------------------------------
# 5. Toeplitz: re-export from efgp_jax (proven impl)
# ---------------------------------------------------------------------------
class ToeplitzNDJax(NamedTuple):
    """Mirror of :class:`efgp_jax.toeplitz.ToeplitzND`.

    Stored as a JAX pytree so it can flow through ``jit``/``lax.scan``.
    """
    v_fft: Array
    ns: Tuple[int, ...]
    fft_shape: Tuple[int, ...]
    starts: Tuple[int, ...]
    ends: Tuple[int, ...]


def make_toeplitz(v: Array, force_pow2: bool = True) -> ToeplitzNDJax:
    """Build the BTTB FFT cache from the convolution vector ``v``.

    ``v`` shape: ``(L1, ..., Ld)`` with ``Li = 2*ni - 1``.
    """
    if not jnp.iscomplexobj(v):
        v = v.astype(_cmplx(v.dtype))
    Ls = list(v.shape)
    ns = tuple((L + 1) // 2 for L in Ls)

    if force_pow2:
        fft_shape = tuple(1 << (L - 1).bit_length() for L in Ls)
    else:
        fft_shape = tuple(Ls)

    pad_widths = [(0, F - L) for L, F in zip(Ls, fft_shape)]
    v_pad = jnp.pad(v, pad_widths)
    fft_dims = tuple(range(v_pad.ndim))
    v_fft = jnp.fft.fftn(v_pad, axes=fft_dims)

    starts = tuple(n - 1 for n in ns)
    ends = tuple(st + n for st, n in zip(starts, ns))
    return ToeplitzNDJax(v_fft=v_fft, ns=ns, fft_shape=fft_shape,
                         starts=starts, ends=ends)


def toeplitz_apply(top: ToeplitzNDJax, x: Array) -> Array:
    """T @ x via FFT convolution + central-block extract."""
    ns = top.ns
    d = len(ns)
    size = prod(ns)

    orig_flat = False
    if x.shape[-1] == size and (x.ndim == 1 or d > 1):
        orig_flat = True
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, *ns)
    else:
        batch_shape = x.shape[:-d]

    if not jnp.iscomplexobj(x):
        x = x.astype(top.v_fft.dtype)

    pad_widths = [(0, 0)] * len(batch_shape) + [
        (0, F - n) for n, F in zip(ns, top.fft_shape)
    ]
    x_pad = jnp.pad(x, pad_widths)
    fft_dims = tuple(range(len(batch_shape), len(batch_shape) + d))
    x_fft = jnp.fft.fftn(x_pad, axes=fft_dims)
    y_fft = x_fft * top.v_fft
    y = jnp.fft.ifftn(y_fft, axes=fft_dims)

    slices = [slice(None)] * len(batch_shape)
    for st, en in zip(top.starts, top.ends):
        slices.append(slice(st, en))
    y = y[tuple(slices)]
    if orig_flat:
        y = y.reshape(*batch_shape, size)
    return y


# ---------------------------------------------------------------------------
# 6. A_r matvec
# ---------------------------------------------------------------------------
def make_A_apply(ws: Array, top: ToeplitzNDJax,
                 sigmasq: float = 1.0) -> Callable[[Array], Array]:
    """Return ``v -> v + D T D v / σ²`` (with σ²=1 by default since the σ_r
    scaling lives inside the Toeplitz weights for SING-EFGP).
    """
    ns = top.ns
    if jnp.iscomplexobj(ws):
        ws_real = ws.real
    else:
        ws_real = ws
    cdtype = top.v_fft.dtype

    def A_apply(v):
        if not jnp.iscomplexobj(v):
            v = v.astype(cdtype)
        if v.ndim <= 1:
            Tv = toeplitz_apply(top, ws_real * v)
            return ws_real * Tv + sigmasq * v
        else:
            ws_block = ws_real.reshape(1, *ns)
            shape_in = (v.shape[0], *ns)
            v_block = v.reshape(shape_in)
            Tv = toeplitz_apply(top, ws_block * v_block)
            res = ws_block * Tv + sigmasq * v_block
            return res.reshape(v.shape)
    return A_apply


# ---------------------------------------------------------------------------
# 7. CG solve (mirror of efgp_jax/cg.py)
# ---------------------------------------------------------------------------
def cg_solve(A_apply: Callable[[Array], Array], b: Array, *,
             tol: float = 1e-5, max_iter: int = None,
             M_inv_apply: Callable[[Array], Array] = None,
             x0: Array = None) -> Array:
    """JIT-compatible CG solver via lax.while_loop."""
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if max_iter is None:
        max_iter = 2 * n
    div_eps = 1e-16
    precond = M_inv_apply if M_inv_apply is not None else (lambda v: v)

    r0 = b - A_apply(x0)
    z0 = precond(r0)
    p0 = z0
    r_dot_z0 = jnp.vdot(r0, z0).real
    r0_norm = jnp.sqrt(r_dot_z0)
    init = (x0, r0, z0, p0, r_dot_z0, r0_norm, 0)

    def cond_fn(state):
        x, r, z, p, r_dot_z, r0n, i = state
        rel = jnp.sqrt(jnp.abs(r_dot_z)) / (r0n + div_eps)
        return (i < max_iter) & (rel > tol)

    def body_fn(state):
        x, r, z, p, r_dot_z, r0n, i = state
        Ap = A_apply(p)
        pAp = jnp.vdot(p, Ap).real + div_eps
        alpha = r_dot_z / pAp
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = precond(r_new)
        r_dot_z_new = jnp.vdot(r_new, z_new).real
        beta = r_dot_z_new / (r_dot_z + div_eps)
        p_new = z_new + beta * p
        return (x_new, r_new, z_new, p_new, r_dot_z_new, r0n, i + 1)

    final = jax.lax.while_loop(cond_fn, body_fn, init)
    return final[0]


# ---------------------------------------------------------------------------
# 8. Hutchinson estimator for diag(Φ A⁻¹ Φ*)
# ---------------------------------------------------------------------------
def hutchinson_diag(grid: JaxGridState, top: ToeplitzNDJax,
                     X_eval: Array, A_apply: Callable[[Array], Array],
                     key: Array, *, J: int = 16, cg_tol: float = 1e-4,
                     max_cg_iter: int = None,
                     nufft_eps: float = 6e-8) -> Array:
    """Estimate diag(Φ_X_eval A⁻¹ Φ_X_eval*) by Rademacher probes.

    Each probe:
       z (Rademacher on X_eval) → Φ* z   (type-1 NUFFT)
       D z'                             (multiply by ws)
       u = A⁻¹ (D Φ* z)                  (CG solve)
       D u                              (multiply by ws)
       v = Φ (D u)                       (type-2 NUFFT)
       accumulate z ⊙ v
    """
    N_eval = int(X_eval.shape[0])
    if max_cg_iter is None:
        max_cg_iter = 2 * grid.M
    OUT = grid.mtot_per_dim
    ws_real = grid.ws.real

    def one_probe(key_p):
        z_real = (jax.random.bernoulli(key_p, shape=(N_eval,))
                  .astype(grid.xis_flat.dtype) * 2.0 - 1.0)
        z = z_real.astype(grid.ws.dtype)
        phi_star_z = nufft1(X_eval, z, grid.xcen, grid.h_per_dim,
                             out_shape=OUT, eps=nufft_eps).reshape(-1)
        rhs = ws_real.astype(phi_star_z.dtype) * phi_star_z
        u = cg_solve(A_apply, rhs, tol=cg_tol, max_iter=max_cg_iter)
        u_weighted = ws_real.astype(u.dtype) * u
        # nufft2 expects fk_grid in block shape (m1, ..., md)
        fk_block = u_weighted.reshape(*OUT)
        phi_u = nufft2(X_eval, fk_block, grid.xcen, grid.h_per_dim,
                        eps=nufft_eps)
        return z_real * phi_u.real

    keys = jr.split(key, J)
    probes = jax.vmap(one_probe)(keys)
    return probes.mean(axis=0)


# ---------------------------------------------------------------------------
# Cheap diagonal preconditioner for A
# ---------------------------------------------------------------------------
def make_jacobi_precond(ws: Array, sigmasq: float = 1.0
                         ) -> Callable[[Array], Array]:
    """Diagonal preconditioner  M^-1 = 1 / (|ws|² + σ²)."""
    if jnp.iscomplexobj(ws):
        ws_sq = (ws * jnp.conj(ws)).real
    else:
        ws_sq = ws * ws
    diag = ws_sq + sigmasq
    return lambda v: v / diag.astype(v.dtype)
