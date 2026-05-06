"""Test custom Gaussian-mixture spreader vs dense closed form.

Variant (A) from efgp_estep_charfun.tex §5.  Unlike the bucketed variant
(B), this one has no within-bucket bias and is exact up to FFT discretisation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax_finufft  # noqa: F401  ensure load order
import numpy as np
import jax
import jax.numpy as jnp

from sing.efgp_gmix_spreader import (
    gmix_nufft_2d, crop_centered, stencil_radius_for, pick_grid_size,
)


def dense_closed_form(m, S, w, xi_out):
    """Direct evaluation at non-uniform output frequencies."""
    phase = np.exp(2j * np.pi * (m @ xi_out.T))
    quad = np.einsum('nd,tde,ne->tn', xi_out, S, xi_out)
    damp = np.exp(-2 * np.pi**2 * quad)
    return (w[:, None] * phase * damp).sum(axis=0)


def _make_problem(T, *, S_scale=0.05, vary=True, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1.5, 1.5, size=(T, 2))
    if vary:
        Ls = rng.standard_normal((T, 2, 2)) * np.sqrt(S_scale)
        S = np.einsum('tij,tkj->tik', Ls, Ls) + S_scale * np.eye(2)
    else:
        L0 = rng.standard_normal((2, 2)) * np.sqrt(S_scale)
        S0 = L0 @ L0.T + S_scale * np.eye(2)
        S = np.broadcast_to(S0, (T, 2, 2)).copy()
    w = rng.standard_normal(T) + 1j * rng.standard_normal(T) * 0.2
    return m, S, w


def _spectral_grid(M_per_dim, h_spec):
    K = (M_per_dim - 1) // 2
    j = np.arange(-K, K + 1)
    JX, JY = np.meshgrid(j, j, indexing='ij')
    return np.stack([(JX * h_spec).ravel(), (JY * h_spec).ravel()], axis=-1)


def test_constant_S_2d():
    """Dense vs spreader on constant S_i.  Should match to FFT precision."""
    T = 200
    h_spec = 0.2
    M_per_dim = 21
    m, S, w = _make_problem(T, S_scale=0.04, vary=False, seed=1)
    xi_out = _spectral_grid(M_per_dim, h_spec)
    ref = dense_closed_form(m, S, w, xi_out)

    sigma_max = float(np.sqrt(np.linalg.eigvalsh(S[0]).max()))
    N = pick_grid_size(h_spec=h_spec, m_extent=4.0, sigma_max=sigma_max)
    h_grid = 1.0 / (N * h_spec)
    r = stencil_radius_for(jnp.asarray(S), h_grid)
    print(f"\n[constant S]  N={N}  h_grid={h_grid:.4f}  stencil_r={r}")

    F = gmix_nufft_2d(jnp.asarray(m), jnp.asarray(S),
                      jnp.asarray(w),
                      xcen=jnp.zeros(2), h_spec=h_spec, N=N, stencil_r=r)
    got = np.asarray(crop_centered(F, M_per_dim)).ravel()
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  rel-err vs dense = {err:.2e}")
    assert err < 1e-3, f"constant-S spreader vs dense: {err:.2e}"


def test_varying_S_2d():
    """Dense vs spreader on per-source varying S — the core advantage of
    variant A over variant B (bucketed).  Should match to FFT precision
    regardless of how much S_i varies."""
    T = 200
    h_spec = 0.2
    M_per_dim = 21
    m, S, w = _make_problem(T, S_scale=0.04, vary=True, seed=2)
    xi_out = _spectral_grid(M_per_dim, h_spec)
    ref = dense_closed_form(m, S, w, xi_out)

    lams = np.array([np.linalg.eigvalsh(S[t]).max() for t in range(T)])
    sigma_max = float(np.sqrt(lams.max()))
    print(f"\n[varying S]   sigma range = [{np.sqrt(lams.min()):.3f}, "
          f"{sigma_max:.3f}]")
    N = pick_grid_size(h_spec=h_spec, m_extent=4.0, sigma_max=sigma_max)
    h_grid = 1.0 / (N * h_spec)
    r = stencil_radius_for(jnp.asarray(S), h_grid)
    print(f"  N={N}  h_grid={h_grid:.4f}  stencil_r={r}")

    F = gmix_nufft_2d(jnp.asarray(m), jnp.asarray(S),
                      jnp.asarray(w),
                      xcen=jnp.zeros(2), h_spec=h_spec, N=N, stencil_r=r)
    got = np.asarray(crop_centered(F, M_per_dim)).ravel()
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  rel-err vs dense = {err:.2e}")
    assert err < 1e-3, f"varying-S spreader vs dense: {err:.2e}"


def test_resolution_convergence():
    """Increasing N (finer grid) should drive the spreader closer to dense."""
    T = 100
    h_spec = 0.2
    M_per_dim = 15
    m, S, w = _make_problem(T, S_scale=0.04, vary=True, seed=3)
    xi_out = _spectral_grid(M_per_dim, h_spec)
    ref = dense_closed_form(m, S, w, xi_out)

    print(f"\n[resolution convergence]")
    sigma_max = float(np.sqrt(max(
        np.linalg.eigvalsh(S[t]).max() for t in range(T))))
    last_err = None
    for N in [32, 64, 128, 256]:
        h_grid = 1.0 / (N * h_spec)
        r = max(2, int(np.ceil(4 * sigma_max / h_grid)))
        F = gmix_nufft_2d(jnp.asarray(m), jnp.asarray(S),
                          jnp.asarray(w),
                          xcen=jnp.zeros(2), h_spec=h_spec, N=N, stencil_r=r)
        got = np.asarray(crop_centered(F, M_per_dim)).ravel()
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"  N={N:>3}  h_grid={h_grid:.4f}  r={r}  rel-err={err:.2e}")
        last_err = err
    assert last_err < 1e-4, f"finest grid should be ~1e-4 or better: {last_err:.2e}"


def test_high_S_variation():
    """Stress test: S spans 10x range (cold-start regime).  This is exactly
    the regime where bucketing fails; the per-source spreader handles it
    natively as long as the spatial periodic-extent 1/h_spec covers the
    source means plus 4*sigma_max."""
    T = 200
    # h_spec = 0.08 → 1/h_spec = 12.5, room for 1.5 + 4*1.2 = 6.3 source/tail
    h_spec = 0.08
    M_per_dim = 21
    rng = np.random.default_rng(4)
    m = rng.uniform(-1.5, 1.5, size=(T, 2))
    # S that varies by 10x in scale
    base_scale = rng.uniform(0.01, 0.1, size=T)  # 10x variation
    Ls = rng.standard_normal((T, 2, 2)) * np.sqrt(base_scale)[:, None, None]
    S = np.einsum('tij,tkj->tik', Ls, Ls) + base_scale[:, None, None] * np.eye(2)
    w = rng.standard_normal(T)

    xi_out = _spectral_grid(M_per_dim, h_spec)
    ref = dense_closed_form(m, S, w, xi_out)

    lams = np.array([np.linalg.eigvalsh(S[t]).max() for t in range(T)])
    sigma_max = float(np.sqrt(lams.max()))
    print(f"\n[high S variation]  sigma range = "
          f"[{np.sqrt(lams.min()):.3f}, {sigma_max:.3f}]  "
          f"(1/h_spec = {1/h_spec:.1f})")

    N = pick_grid_size(h_spec=h_spec, m_extent=4.0, sigma_max=sigma_max)
    h_grid = 1.0 / (N * h_spec)
    r = stencil_radius_for(jnp.asarray(S), h_grid)
    print(f"  N={N}  h_grid={h_grid:.4f}  stencil_r={r}")
    F = gmix_nufft_2d(jnp.asarray(m), jnp.asarray(S),
                      jnp.asarray(w),
                      xcen=jnp.zeros(2), h_spec=h_spec, N=N, stencil_r=r)
    got = np.asarray(crop_centered(F, M_per_dim)).ravel()
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  rel-err vs dense = {err:.2e}")
    assert err < 1e-3, f"high-variation spreader vs dense: {err:.2e}"


if __name__ == '__main__':
    test_constant_S_2d()
    test_varying_S_2d()
    test_resolution_convergence()
    test_high_S_variation()
    print("\nAll tests passed.")
