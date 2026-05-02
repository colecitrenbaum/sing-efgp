"""Time the torch EFGP path on a few problem sizes; dump to JSON.

Runs in its own python process to avoid the torch+jax_finufft load-order
segfault.

Usage:  python tests/_time_torch_path.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
sys.path.insert(0, str(_GP_QUAD))

import torch  # noqa: E402
from kernels import SquaredExponential  # noqa: E402
from efgpnd import (  # noqa: E402
    NUFFT, ToeplitzND, create_A_mean, _resolve_grid, _cmplx,
)
from cg import ConjugateGradients  # noqa: E402

OUT_PATH = Path(__file__).with_name("_torch_timing.json")


def time_at(d: int, N: int, ls: float, var: float, eps_grid: float,
            n_warmup: int = 2, n_iters: int = 10):
    dtype = torch.float64
    cdtype = _cmplx(dtype)
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((N, d))
    weights_np = rng.uniform(0.5, 1.5, size=N)
    X = torch.tensor(X_np, dtype=dtype)
    W = torch.tensor(weights_np, dtype=dtype)

    ker = SquaredExponential(dimension=d, init_lengthscale=ls,
                              init_variance=var)
    xis_1d_list, h_t, mtot_per_dim, xis_flat = _resolve_grid(
        ker, X, eps_grid, frozen_grid=None, max_mtot_1d=None,
        rdtype=dtype, device='cpu')
    M = int(xis_flat.shape[0])
    spec = ker.spectral_density(xis_flat).to(dtype)
    h_prod = torch.prod(h_t)
    ws = torch.sqrt(spec * h_prod).to(cdtype)
    xcen = 0.5 * (X.max(dim=0).values + X.min(dim=0).values)
    nufft = NUFFT(X, xcen, h_t, eps=6e-8, cdtype=cdtype)
    m_per_dim = [(m - 1) // 2 for m in mtot_per_dim]
    OUT_conv = tuple(4 * mi + 1 for mi in m_per_dim)
    W_c = W.to(cdtype)

    # --- one full update_dynamics_params analogue (build top + A + 1 CG) ---
    def step():
        v_kernel = nufft.type1(W_c, out_shape=OUT_conv)
        top = ToeplitzND(v_kernel, force_pow2=True, precompute_fft=True)
        ws_real = ws.real.to(dtype)
        A = create_A_mean(ws_real, top, sigmasq_scalar=1.0, cdtype=cdtype)
        # h_r = D F* (W * y) for some y
        y = torch.randn(N, dtype=cdtype)
        h_r = ws_real * nufft.type1(W_c * y,
                                     out_shape=tuple(mtot_per_dim)).reshape(-1)
        cg = ConjugateGradients(A, h_r, torch.zeros_like(h_r),
                                tol=1e-5, max_iter=4 * M, early_stopping=True)
        return cg.solve()

    # warmup
    for _ in range(n_warmup):
        step()
    t0 = time.time()
    for _ in range(n_iters):
        out = step()
    dt = (time.time() - t0) / n_iters
    return {'M': M, 'ms_per_iter': dt * 1000.0}


def main():
    out = {}
    for M_target in [16, 32, 50]:
        # tweak ls so M ≈ target^d at d=2
        # heuristic: ls ~ extent / sqrt(M_target);  start with ls=1.5, eps=1e-2
        for ls, eps_grid in [(1.0, 5e-2), (0.7, 1e-2), (0.4, 1e-3)]:
            res = time_at(d=2, N=200, ls=ls, var=1.0, eps_grid=eps_grid,
                          n_iters=20)
            key = f"d=2,N=200,ls={ls},eps={eps_grid}"
            out[key] = res
            print(f"  {key:40s} M={res['M']:5d}  {res['ms_per_iter']:7.2f} ms")
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
