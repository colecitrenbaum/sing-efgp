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

    # ---- Full update_dynamics_params analogue with deterministic X cloud ----
    # We build (T, ms, Ss, SSs) and a SHARED pseudo-cloud X_flat outside the
    # sampler so the JAX side can use the identical inputs.
    rng3 = np.random.default_rng(seed + 42)
    T = 50
    ms_np = rng3.standard_normal((T, d)).astype(np.float64) * 0.4
    # Diagonal-ish positive Ss
    Ss_np = (np.tile(0.05 * np.eye(d), (T, 1, 1))
             + 0.01 * rng3.standard_normal((T, d, d)) ** 2).astype(np.float64)
    Ss_np = (Ss_np + Ss_np.transpose(0, 2, 1)) / 2 + 0.05 * np.eye(d)
    SSs_np = 0.04 * np.tile(np.eye(d), (T - 1, 1, 1)).astype(np.float64)
    del_t_np = np.full(T - 1, 0.04, dtype=np.float64)
    sigma_drift_sq = 0.16
    S_marginal = 4

    # Sample pseudo-cloud once via numpy
    eps_samples = rng3.standard_normal((T - 1, S_marginal, d))
    L_chol = np.linalg.cholesky(Ss_np[:-1] + 1e-9 * np.eye(d))
    X_cloud = (ms_np[:-1, None, :]
               + np.einsum('tij,tsj->tsi', L_chol, eps_samples))
    X_flat_np = X_cloud.reshape(-1, d)

    # ---- Run the EFGP-SING torch path on this exact cloud ----
    # Setup like update_dynamics_params (no inputs, σ shared across r)
    X_full = torch.tensor(X_flat_np, dtype=dtype)
    delta_per_pseudo = np.repeat(del_t_np, S_marginal)
    weights_full_np = delta_per_pseudo / (S_marginal * sigma_drift_sq)
    weights_full = torch.tensor(weights_full_np, dtype=dtype)

    # Spectral grid for this cloud
    xis_1d_list2, h_t2, mtot_per_dim2, xis_flat2 = _resolve_grid(
        ker, X_full, eps_grid, frozen_grid=None, max_mtot_1d=None,
        rdtype=dtype, device='cpu')
    M2 = int(xis_flat2.shape[0])
    spec2 = ker.spectral_density(xis_flat2).to(dtype)
    h_prod2 = torch.prod(h_t2)
    ws2 = torch.sqrt(spec2 * h_prod2).to(cdtype)
    xcen2 = 0.5 * (X_full.max(dim=0).values + X_full.min(dim=0).values)

    nufft_full = NUFFT(X_full, xcen2, h_t2, eps=6e-8, cdtype=cdtype)
    OUT_full_conv = tuple(2 * (m - 1) + 1 for m in mtot_per_dim2)  # 4m+1
    weights_c = weights_full.to(cdtype)
    v_kernel_full = nufft_full.type1(weights_c, out_shape=OUT_full_conv)
    top_full = ToeplitzND(v_kernel_full, force_pow2=True, precompute_fft=True)
    ws_real2 = ws2.real.to(dtype)
    A_full = create_A_mean(ws_real2, top_full, sigmasq_scalar=1.0, cdtype=cdtype)

    D_out = d
    D_lat = d
    mu_r_full = torch.zeros((D_out, M2), dtype=cdtype)
    d_a = ms_np[1:] - ms_np[:-1]                                  # (T-1, d)
    C_a = SSs_np - Ss_np[:-1]                                      # (T-1, d, d)
    for r in range(D_out):
        a_r = np.repeat(d_a[:, r], S_marginal) / (S_marginal * sigma_drift_sq)
        a_r_t = torch.tensor(a_r, dtype=cdtype)
        h1 = ws_real2 * nufft_full.type1(
            a_r_t, out_shape=tuple(mtot_per_dim2)).reshape(-1)
        h2 = torch.zeros(M2, dtype=cdtype)
        for j in range(D_lat):
            c_jr = (np.repeat(C_a[:, j, r], S_marginal)
                    / (S_marginal * sigma_drift_sq))
            c_t = torch.tensor(c_jr, dtype=cdtype)
            Fstar_c = nufft_full.type1(
                c_t, out_shape=tuple(mtot_per_dim2)).reshape(-1)
            xi_j = xis_flat2[:, j].to(cdtype)
            h2 = h2 + (2j * math.pi * xi_j) * (ws_real2 * Fstar_c)
        h_r_full = (h1 + h2).to(cdtype)
        cg2 = ConjugateGradients(A_full, h_r_full, torch.zeros_like(h_r_full),
                                  tol=1e-7, max_iter=4 * M2,
                                  early_stopping=True)
        mu_r_full[r] = cg2.solve()

    # Drift moments at ms_np
    ms_t = torch.tensor(ms_np, dtype=dtype)
    Ef_t = torch.zeros((T, D_out), dtype=dtype)
    Edfdx_t = torch.zeros((T, D_out, D_lat), dtype=dtype)
    nufft_eval = NUFFT(ms_t, xcen2, h_t2, eps=6e-8, cdtype=cdtype)
    for r in range(D_out):
        fk = (ws_real2.to(cdtype) * mu_r_full[r])
        out = nufft_eval.type2(fk.reshape(-1), out_shape=tuple(mtot_per_dim2))
        Ef_t[:, r] = out.real
        for j in range(D_lat):
            xi_j = xis_flat2[:, j].to(cdtype)
            fk_j = (2j * math.pi * xi_j) * (ws_real2.to(cdtype) * mu_r_full[r])
            jac = nufft_eval.type2(fk_j.reshape(-1),
                                    out_shape=tuple(mtot_per_dim2))
            Edfdx_t[:, r, j] = jac.real
    Eff_t = (Ef_t ** 2).sum(dim=1)

    full_qf_ref = {
        'T': int(T), 'sigma_drift_sq': float(sigma_drift_sq),
        'S_marginal': int(S_marginal),
        'ms': ms_np, 'Ss': Ss_np, 'SSs': SSs_np, 'del_t': del_t_np,
        'X_flat_full': X_flat_np,
        'mtot_per_dim_full': tuple(mtot_per_dim2),
        'M_full': int(M2),
        'xis_flat_full': xis_flat2.cpu().numpy(),
        'ws_full': ws2.cpu().numpy(),
        'xcen_full': xcen2.cpu().numpy(),
        'h_full': h_t2.cpu().numpy(),
        'mu_r_full': mu_r_full.cpu().numpy(),
        'Ef_full': Ef_t.cpu().numpy(),
        'Eff_full': Eff_t.cpu().numpy(),
        'Edfdx_full': Edfdx_t.cpu().numpy(),
    }

    return {
        'full_qf': full_qf_ref,
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
