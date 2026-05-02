"""Generate the torch reference outputs for the JAX-primitive parity tests.

Run this script once (in its own python process — it will import torch +
pytorch_finufft, which would conflict with jax_finufft if both are loaded
in the same process).  Outputs go to tests/_torch_reference.pkl.

Usage:  python tests/_make_torch_reference.py
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np

_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
sys.path.insert(0, str(_GP_QUAD))

import torch  # noqa: E402
from kernels import SquaredExponential, GPParams  # noqa: E402
from efgpnd import (  # noqa: E402
    NUFFT,
    ToeplitzND,
    create_A_mean,
    create_Gv,
    _resolve_grid,
    _cmplx,
)
from cg import ConjugateGradients  # noqa: E402


REF_PATH = Path(__file__).with_name("_torch_reference.pkl")


def _build(d: int = 2, N: int = 60, ls: float = 0.6, var: float = 1.0,
           eps_grid: float = 1e-3, dtype=torch.float64,
           seed: int = 7):
    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((N, d))
    weights_np = rng.uniform(0.5, 1.5, size=N)
    vals_real_np = rng.standard_normal(N)
    vals_cplx_np = vals_real_np + 1j * rng.standard_normal(N)

    cdtype = _cmplx(dtype)
    X = torch.tensor(X_np, dtype=dtype)
    W = torch.tensor(weights_np, dtype=dtype)

    # Spectral grid
    ker = SquaredExponential(dimension=d, init_lengthscale=ls, init_variance=var)
    GPParams(kernel=ker, init_sig2=1.0)
    xis_1d_list, h_t, mtot_per_dim, xis_flat = _resolve_grid(
        ker, X, eps_grid, frozen_grid=None, max_mtot_1d=None,
        rdtype=dtype, device='cpu',
    )
    M = xis_flat.shape[0]
    spec = ker.spectral_density(xis_flat).to(dtype=dtype)
    h_prod = torch.prod(h_t)
    ws = torch.sqrt(spec * h_prod).to(cdtype)
    xcen = 0.5 * (X.max(dim=0).values + X.min(dim=0).values)

    # Use the SAME xcen for both Toeplitz and eval so BTTB structure holds.
    nufft = NUFFT(X, xcen, h_t, eps=6e-8, cdtype=cdtype)

    # type-1 (vals → grid)
    vals_t = torch.tensor(vals_cplx_np, dtype=cdtype)
    out_nufft1 = nufft.type1(vals_t, out_shape=tuple(mtot_per_dim))

    # type-2 (grid → vals); use a simple grid-shape coefficient pattern
    fk_grid = torch.tensor(rng.standard_normal(tuple(mtot_per_dim))
                           + 1j * rng.standard_normal(tuple(mtot_per_dim)),
                           dtype=cdtype)
    # efgpnd's NUFFT.type2 uses ndim>1 as a batching cue, so we must flatten.
    out_nufft2 = nufft.type2(fk_grid.reshape(-1), out_shape=tuple(mtot_per_dim))

    # BTTB conv vec  (size 4m+1 per dim)
    m_per_dim = [(m - 1) // 2 for m in mtot_per_dim]
    OUT_conv = tuple(4 * mi + 1 for mi in m_per_dim)
    W_c = W.to(cdtype)
    v_kernel = nufft.type1(W_c, out_shape=OUT_conv)

    # Toeplitz matvec
    top = ToeplitzND(v_kernel, force_pow2=True, precompute_fft=True)
    rng2 = np.random.default_rng(seed + 1)
    test_vec_np = (rng2.standard_normal(M) + 1j * rng2.standard_normal(M))
    test_vec = torch.tensor(test_vec_np, dtype=cdtype)
    out_toeplitz = top(test_vec)

    # A_apply  (= D T D + I)
    ws_real = ws.real.to(dtype=dtype)
    A_apply = create_A_mean(ws_real, top, sigmasq_scalar=1.0, cdtype=cdtype)
    out_Aapply = A_apply(test_vec)

    # CG solve  A μ = h_r
    h_r = ws_real * nufft.type1(W_c * vals_t, out_shape=tuple(mtot_per_dim)).reshape(-1)
    cg = ConjugateGradients(A_apply, h_r, torch.zeros_like(h_r),
                            tol=1e-10, max_iter=4 * M, early_stopping=True)
    mu_solve = cg.solve()

    return {
        'config': dict(d=d, N=N, ls=ls, var=var, eps_grid=eps_grid, seed=seed),
        # inputs (so JAX side uses the SAME random vectors)
        'X': X_np, 'weights': weights_np,
        'vals_cplx': vals_cplx_np,
        'fk_grid': fk_grid.cpu().numpy(),
        'test_vec': test_vec_np,
        # spectral grid
        'h_per_dim': h_t.cpu().numpy(),
        'mtot_per_dim': tuple(mtot_per_dim),
        'M': int(M),
        'xis_flat': xis_flat.cpu().numpy(),
        'ws': ws.cpu().numpy(),
        'xcen': xcen.cpu().numpy(),
        # primitive outputs
        'out_nufft1': out_nufft1.cpu().numpy(),
        'out_nufft2': out_nufft2.cpu().numpy(),
        'v_kernel': v_kernel.cpu().numpy(),
        'out_toeplitz': out_toeplitz.cpu().numpy(),
        'out_Aapply': out_Aapply.cpu().numpy(),
        'mu_solve': mu_solve.cpu().numpy(),
        'h_r': h_r.cpu().numpy(),
    }


def main():
    ref = _build()
    REF_PATH.write_bytes(pickle.dumps(ref))
    print(f"wrote {REF_PATH}  M={ref['M']}")


if __name__ == "__main__":
    main()
