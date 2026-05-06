"""Test compute_mu_r_gmix_jax vs the MC version (compute_mu_r_jax).

Both should approximate the same closed-form Σ_i E_q(x_i)[...] integrand;
gmix is exact (modulo FFT discretisation), MC is unbiased but noisy.
For a given q(x), the gmix mu_r should be close to the MC mu_r averaged
over many seeds, and to the dense closed-form reference.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax_finufft  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size


D = 2
sigma_drift_sq = 0.1


def _make_qx_and_drift(T, key):
    """Random smooth q(x) (means + cov + cross-cov) and time grid."""
    k_m, k_S, k_SS = jr.split(key, 3)
    t_grid = jnp.linspace(0., 1., T)
    del_t = t_grid[1:] - t_grid[:-1]
    # Smooth means via low-freq sin/cos
    tt = t_grid[:, None]
    ms = jnp.stack([
        jnp.sin(2 * tt[:, 0]) + 0.3 * jnp.cos(4 * tt[:, 0]),
        jnp.cos(3 * tt[:, 0]) + 0.2 * jnp.sin(5 * tt[:, 0]),
    ], axis=1).astype(jnp.float32)
    # Smoothly varying SPD covariances
    base = 0.04 + 0.03 * jnp.sin(2 * t_grid)
    Ss = jnp.stack([
        jnp.array([[b, 0.5 * b], [0.5 * b, 0.7 * b]], dtype=jnp.float32)
        for b in np.array(base)
    ])
    SSs = 0.95 * Ss[:-1]
    return ms, Ss, SSs, del_t


def test_gmix_vs_mc_aggregate():
    T = 200
    key = jr.PRNGKey(0)
    ms, Ss, SSs, del_t = _make_qx_and_drift(T, key)

    # Build EFGP grid
    X_template = (jnp.linspace(-2., 2., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(0.6, 1.0, X_template, eps=1e-2)
    print(f"\ngrid: M={grid.M}, M_per_dim={grid.mtot_per_dim}, h_spec={float(grid.h_per_dim[0]):.4f}")

    # Pre-flatten Stein quantities into supersources (single-trial path).
    trial_mask = jnp.ones((1, T), dtype=bool)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms[None], Ss[None], SSs[None], del_t, trial_mask)

    # MC version with many samples (high accuracy reference)
    mu_mc_avg = None
    n_seeds = 8
    S_marg_high = 32
    for s in range(n_seeds):
        key_s = jr.PRNGKey(100 + s)
        mu_r, _, _ = jpd.compute_mu_r_jax(
            m_src, S_src, d_src, C_src, w_src, grid, key_s,
            sigma_drift_sq=sigma_drift_sq, S_marginal=S_marg_high,
            D_lat=D, D_out=D)
        mu_mc_avg = mu_r if mu_mc_avg is None else mu_mc_avg + mu_r
    mu_mc_avg /= n_seeds
    print(f"  MC reference: avg over {n_seeds} seeds × S_marginal={S_marg_high}")

    # Single-seed MC at production setting
    mu_mc_single, _, _ = jpd.compute_mu_r_jax(
        m_src, S_src, d_src, C_src, w_src, grid, jr.PRNGKey(0),
        sigma_drift_sq=sigma_drift_sq, S_marginal=2,
        D_lat=D, D_out=D)

    # gmix version
    h_spec = float(grid.h_per_dim[0])
    sigma_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(Ss))))
    fine_N = pick_grid_size(h_spec=h_spec, m_extent=4.0, sigma_max=sigma_max)
    h_grid = 1.0 / (fine_N * h_spec)
    stencil_r = stencil_radius_for(Ss, h_grid)
    print(f"  gmix:  fine_N={fine_N}  h_grid={h_grid:.4f}  stencil_r={stencil_r}")

    mu_gmix, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D,
        fine_N=fine_N, stencil_r=stencil_r)

    # Distances
    err_gmix_vs_ref = float(jnp.linalg.norm(mu_gmix - mu_mc_avg)
                              / jnp.linalg.norm(mu_mc_avg))
    err_mc1_vs_ref = float(jnp.linalg.norm(mu_mc_single - mu_mc_avg)
                              / jnp.linalg.norm(mu_mc_avg))
    print(f"  || mu_gmix  - mu_MC_avg || / || mu_MC_avg || = {err_gmix_vs_ref:.4f}")
    print(f"  || mu_MC1   - mu_MC_avg || / || mu_MC_avg || = {err_mc1_vs_ref:.4f}")
    print(f"  (gmix should be much closer to MC_avg than single-seed MC)")
    # gmix should be at LEAST as close to the MC reference as a single-seed MC is.
    assert err_gmix_vs_ref < err_mc1_vs_ref, (
        f"gmix ({err_gmix_vs_ref:.4f}) not closer than single-seed MC "
        f"({err_mc1_vs_ref:.4f}) to MC reference")


if __name__ == "__main__":
    test_gmix_vs_mc_aggregate()
    print("\nPassed.")
