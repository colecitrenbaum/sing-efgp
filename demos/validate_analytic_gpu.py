"""Correctness check (CPU) for the GPU-parallel analytic q(f) variants.

Compares mu_r from the new dense-GEMM and grid-NUFFT paths against the
production analytic (cuFINUFFT-Taylor) and gmix paths on synthetic Stein
inputs with controllable per-source S heterogeneity.

Run: JAX_PLATFORMS=cpu python demos/validate_analytic_gpu.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
import sing.efgp_analytic_gpu as gpu


def make_inputs(N_src, het, seed=0):
    """Synthetic flat Stein sources on a 2-D swirl trajectory."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * math.pi, N_src)
    R = 1.0 + 0.3 * t / t.max()
    m = np.stack([R * np.cos(t), R * np.sin(t)], axis=-1)         # (N,2)
    m += 0.02 * rng.standard_normal(m.shape)
    # Per-source covariance: base 0.03²·I scaled by (1 + het·u), u∈[0,1)
    u = rng.random(N_src)
    base = 0.03 ** 2
    scale = 1.0 + het * u
    S = np.zeros((N_src, 2, 2))
    S[:, 0, 0] = base * scale
    S[:, 1, 1] = base * scale * (1.0 + 0.3 * het * rng.random(N_src))
    S[:, 0, 1] = S[:, 1, 0] = 0.2 * np.sqrt(S[:, 0, 0] * S[:, 1, 1]) * rng.random(N_src)
    d = 0.05 * rng.standard_normal((N_src, 2))
    C = 0.01 * rng.standard_normal((N_src, 2, 2))
    w = np.full(N_src, 0.01)
    return (jnp.asarray(m), jnp.asarray(S), jnp.asarray(d),
            jnp.asarray(C), jnp.asarray(w))


def rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    print(f"device={jax.devices()} x64={jax.config.read('jax_enable_x64')}")
    ls, var = 0.4, 1.0
    sds = 0.1 ** 2
    for N_src in (500, 2000):
        for het in (0.0, 2.0, 10.0):
            m, S, d, C, w = make_inputs(N_src, het)
            grid = jp.spectral_grid_se(ls, var, m, eps=1e-3)
            # gmix knobs
            h_spec = float(grid.h_per_dim[0])
            sig_max = float(np.sqrt(np.linalg.eigvalsh(np.asarray(S)).max()))
            m_extent = float((np.asarray(m).max(0) - np.asarray(m).min(0)).max())
            fine_N = pick_grid_size(h_spec=h_spec, m_extent=m_extent,
                                    sigma_max=sig_max, n_resolve=4.0)
            h_grid = 1.0 / (fine_N * h_spec)
            stencil_r = stencil_radius_for(S, h_grid, n_sigma=1.5)

            common = dict(sigma_drift_sq=sds, D_lat=2, D_out=2)
            mu_a1, _, _ = jpd.compute_mu_r_analytic_jax(m, S, d, C, w, grid, order=1, **common)
            mu_a2, _, _ = jpd.compute_mu_r_analytic_jax(m, S, d, C, w, grid, order=2, **common)
            mu_gm, _, _ = jpd.compute_mu_r_gmix_jax(m, S, d, C, w, grid,
                                                    fine_N=fine_N, stencil_r=int(stencil_r), **common)
            mu_gemm, _, _ = gpu.compute_mu_r_densegemm_jax(m, S, d, C, w, grid, **common)
            mu_gemm_c, _, _ = gpu.compute_mu_r_densegemm_jax(m, S, d, C, w, grid, freq_chunk=257, **common)
            mu_sep1, _, _ = gpu.compute_mu_r_sepgemm_jax(m, S, d, C, w, grid, xorder=1, **common)
            mu_sep2, _, _ = gpu.compute_mu_r_sepgemm_jax(m, S, d, C, w, grid, xorder=2, **common)
            mu_grid, _, _ = gpu.compute_mu_r_gridnufft_jax(m, S, d, C, w, grid, order=1, **common)

            print(f"\nN={N_src} het={het}  M={grid.M} fine_N={fine_N} stencil_r={int(stencil_r)}")
            print(f"  gemm   vs analytic-o2 : {rel(mu_gemm, mu_a2):.2e}   (exact-S; o2 is best Taylor)")
            print(f"  gemm   vs analytic-o1 : {rel(mu_gemm, mu_a1):.2e}")
            print(f"  gemm-chunk vs gemm    : {rel(mu_gemm_c, mu_gemm):.2e}   (memory-tiling identity)")
            print(f"  sepgemm-x1 vs gemm    : {rel(mu_sep1, mu_gemm):.2e}   (separable, cross-Taylor o1)")
            print(f"  sepgemm-x2 vs gemm    : {rel(mu_sep2, mu_gemm):.2e}   (separable, cross-Taylor o2)")
            print(f"  gridnufft vs analytic-o1: {rel(mu_grid, mu_a1):.2e}   (same Taylor order)")
            print(f"  gmix   vs gemm(exact) : {rel(mu_gm, mu_gemm):.2e}   (gmix stencil bias)")
            print(f"  analytic-o1 vs o2     : {rel(mu_a1, mu_a2):.2e}")


if __name__ == '__main__':
    main()
