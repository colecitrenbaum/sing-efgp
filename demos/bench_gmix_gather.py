"""Wall-clock + accuracy benchmark for gmix-gather (Type-2 analog of
Gaussian-mixture spreader, used for heterogeneous-S V restoration).

Compares:
  * Homogeneous-S NUFFT-2 (current 'hutch' baseline) -- shared envelope,
    1 batched NUFFT-2 + D adjoint NUFFT-1s.
  * Heterogeneous-S gmix-gather -- 1 IFFT (size N x N) + per-source
    Gaussian gather on (2r+1)^D stencil.

Sweeps stencil_r (the precision-vs-cost knob) and source count N_src.
Reports wall (forward + grad) and rel error vs explicit ground truth.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sing.efgp_gmix_gather import gmix_inverse_nufft_2d
from sing.efgp_gmix_spreader import stencil_radius_for


def _explicit_V_per_source(c, h_spec, m, S, M_per_dim):
    """Brute-force V_i = sum_δ c(δ) e^{+2πi δ^T m_i} e^{-2π² δ^T S_i δ}."""
    j = jnp.arange(-(M_per_dim // 2),
                    M_per_dim - (M_per_dim // 2), dtype=m.dtype)
    JX, JY = jnp.meshgrid(j, j, indexing='ij')
    xi_x = JX * h_spec
    xi_y = JY * h_spec
    def per_source(m_i, S_i):
        phase = jnp.exp(2j * jnp.pi * (xi_x * m_i[0] + xi_y * m_i[1]))
        quad = (S_i[0, 0] * xi_x ** 2 + 2 * S_i[0, 1] * xi_x * xi_y
                + S_i[1, 1] * xi_y ** 2)
        env = jnp.exp(-2 * jnp.pi**2 * quad)
        return jnp.sum(c * phase * env)
    return jax.vmap(per_source)(m, S)


def main():
    rng = np.random.default_rng(42)
    M_per_dim = 33                        # production-ish (≈10^3 grid pts)
    h_spec = 0.30
    xcen = jnp.zeros(2, dtype=jnp.float64)

    c = (rng.standard_normal((M_per_dim, M_per_dim))
         + 1j * rng.standard_normal((M_per_dim, M_per_dim)))
    c = jnp.asarray(c)

    print(f"=== gmix-gather wall-clock + accuracy bench ===")
    print(f"M_per_dim={M_per_dim} (M={M_per_dim**2}), h_spec={h_spec}\n")

    # Production setting: K=10/T=500 -> N_src ~ 5000
    for N_src in [200, 1000, 5000]:
        m = jnp.asarray(rng.uniform(-1.5, 1.5, (N_src, 2)))
        # Heterogeneous S spanning a 4x range: simulates
        # early/late time + sharp transitions of a real q(x).
        S_diags = rng.uniform(0.02, 0.08, (N_src, 2))
        S = jnp.stack([jnp.diag(d) for d in S_diags], axis=0)

        # Ground truth via brute-force sum
        V_truth = _explicit_V_per_source(c, h_spec, m, S, M_per_dim)
        V_truth.block_until_ready()
        mag = float(jnp.max(jnp.abs(V_truth)))

        print(f"N_src = {N_src}, M = {M_per_dim**2}, "
              f"sigma_max ≈ {float(jnp.sqrt(S_diags.max())):.3f}")
        print(f"{'N':>5} {'r':>4} {'fwd(s)':>9} {'grad(s)':>9} "
              f"{'rel_err':>10} {'r_4σ':>5}")
        for N in [64, 128, 256]:
            h_grid = 1.0 / (N * h_spec)
            r_4sigma = stencil_radius_for(S, h_grid, n_sigma=4.0)
            for r_factor in [0.5, 1.0, 1.5]:
                stencil_r = max(int(r_factor * r_4sigma), 2)

                fwd = jax.jit(lambda m_, S_: gmix_inverse_nufft_2d(
                    m_, S_, c, xcen=xcen, h_spec=h_spec,
                    M_per_dim=M_per_dim, N=N, stencil_r=stencil_r))
                grd = jax.jit(jax.grad(lambda m_: gmix_inverse_nufft_2d(
                    m_, S, c, xcen=xcen, h_spec=h_spec,
                    M_per_dim=M_per_dim, N=N, stencil_r=stencil_r).real.sum()))

                # Warm-up (JIT compile)
                _ = fwd(m, S).block_until_ready()
                _ = grd(m).block_until_ready()

                # Time forward
                t0 = time.time()
                for _ in range(5):
                    V = fwd(m, S)
                V.block_until_ready()
                fwd_t = (time.time() - t0) / 5

                # Time grad
                t0 = time.time()
                for _ in range(5):
                    g = grd(m)
                g.block_until_ready()
                grad_t = (time.time() - t0) / 5

                err = float(jnp.max(jnp.abs(V - V_truth)))
                rel = err / (mag + 1e-12)
                print(f"{N:>5} {stencil_r:>4} {fwd_t:>9.3e} {grad_t:>9.3e} "
                      f"{rel:>10.2e} {r_4sigma:>5}")
        print()


if __name__ == '__main__':
    main()
