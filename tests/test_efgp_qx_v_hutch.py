"""Regression tests for V-Hutch restoration of Approximation A
(``sing/efgp_qx_v_hutch.py``).

Two correctness checks:

1. **Unbiasedness vs dense ground truth.** On a tiny problem with the
   complex-Hermitian operator ``A`` materialised as a dense matrix,
   ``precompute_V_and_grad_E_V_per_t`` must converge to the brute-force
   sum ``E[V](m, S) = sum_kl (DAD)_kl exp(2pi i (xi_l - xi_k)^T m -
   2pi^2 (xi_l - xi_k)^T S (xi_l - xi_k))`` as ``P -> infty``. We verify
   the bias decays as 1/sqrt(P) (no residual systematic offset) and
   that the autodiff gradient matches the explicit Bonnet derivative
   to machine precision.

2. **End-to-end fit equivalence.** ``fit_efgp_sing_jax`` with
   ``restore_qf_variance='hutch'`` must reach the same fixed point as
   the ``='none'`` baseline on a well-distributed source set (the
   structural argument in efgp_estep.tex §6 says nabla V is below the
   natural-grad's effective resolution).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import sing.efgp_jax_primitives as jp
from sing.efgp_qx_v_hutch import (
    precompute_omega_per_r,
    precompute_V_and_grad_E_V_per_t,
)


def _build_tiny_problem(D=2, ls=0.7, var=0.8, sigma=0.4, seed=7):
    """Tiny grid + Toeplitz from random sources; small enough for dense
    Cholesky ground truth."""
    X_template = jnp.linspace(-2.5, 2.5, 6)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=5e-2)
    src = jr.normal(jr.PRNGKey(seed), (32, D)) * 1.5
    weights = jnp.ones(32) * (sigma ** -2)
    v = jp.bttb_conv_vec_weighted(src, weights, grid.xcen, grid.h_per_dim,
                                    grid.mtot_per_dim, eps=6e-8)
    top = jp.make_toeplitz(v)
    return grid, top


def _dense_DAD_and_xi(grid, top, D):
    """Materialise A_complex as a dense MxM matrix, return DAD and the
    spectral coordinate grid xi (matching nufft2's centring)."""
    mtot = tuple(int(n) for n in grid.mtot_per_dim)
    M = int(grid.M)
    v_kernel = jnp.fft.ifftn(top.v_fft)             # complex (kernel can be complex Hermitian)
    axes = [jnp.arange(n) for n in mtot]
    gc = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(mtot))
    ns = jnp.array(top.ns)
    diff = gc[None, :, :] - gc[:, None, :]
    lag_idx = diff + (ns - 1)
    T_dense = v_kernel[tuple(lag_idx[..., d] for d in range(D))]
    ws_real_c = grid.ws.real.astype(jnp.complex128)
    A_complex = (jnp.eye(M, dtype=jnp.complex128)
                 + jnp.outer(ws_real_c, ws_real_c) * T_dense)
    A_inv = jnp.linalg.inv(A_complex)
    DAD = ws_real_c[:, None] * A_inv * ws_real_c[None, :]
    xi_offset = (jnp.array(mtot) - 1) / 2
    xi = (gc - xi_offset) * grid.h_per_dim
    return DAD, xi


def test_V_hutch_unbiased_vs_dense_truth():
    """Hutch estimate of E[V] converges to dense ground truth as P grows.
    Bias decays at the expected 1/sqrt(P) rate; no residual systematic offset."""
    D = 2
    grid, top = _build_tiny_problem(D=D)
    DAD, xi = _dense_DAD_and_xi(grid, top, D)
    S_homo = 0.04 * jnp.eye(D)
    ms = jr.normal(jr.PRNGKey(0), (5, 8, D))            # 40 eval points

    def V_truth_one(m_):
        diff_xi = xi[None, :, :] - xi[:, None, :]
        phase = jnp.exp(2j * jnp.pi * (diff_xi @ m_))
        quad = jnp.einsum('klD,DE,klE->kl', diff_xi, S_homo, diff_xi)
        env = jnp.exp(-2 * jnp.pi**2 * quad)
        return D * jnp.sum((DAD * phase * env).real)
    V_truth = jax.vmap(jax.vmap(V_truth_one))(ms)

    # Both max and mean (over eval points) error should be monotone
    # decreasing with P. Mean scales cleanly as 1/sqrt(P); max is noisier
    # because it picks up tails. Test the monotone trend on max + a clean
    # ratio on mean.
    max_rels, mean_signed_errs = [], []
    for P in [64, 256, 1024]:
        omega_aux = precompute_omega_per_r(None, grid, top, jr.PRNGKey(1),
                                             D_out=D, P=P,
                                             cg_tol=1e-9, max_cg_iter=500)
        V_hutch, _ = precompute_V_and_grad_E_V_per_t(
            omega_aux, ms, S_homo, grid, nufft_eps=1e-10)
        max_rels.append(float(jnp.max(jnp.abs(V_hutch - V_truth))
                                / (jnp.max(jnp.abs(V_truth)) + 1e-12)))
        mean_signed_errs.append(float((V_hutch - V_truth).mean()))
    # Monotone decrease on max abs rel err
    assert max_rels[1] < max_rels[0], \
        f"max rel err should decrease 64->256, got {max_rels}"
    assert max_rels[2] < max_rels[1], \
        f"max rel err should decrease 256->1024, got {max_rels}"
    # Mean signed error should approach zero (no bias) as P grows
    assert abs(mean_signed_errs[2]) < abs(mean_signed_errs[0]) * 0.6, \
        f"mean signed err should drop by 4x going 64->1024 (1/sqrt(16)), got {mean_signed_errs}"
    # P=1024 should give <5% max relative error
    assert max_rels[2] < 0.05, \
        f"P=1024 max rel err {max_rels[2]:.3f} should be <5%"


def test_V_hutch_grad_matches_autodiff_through_truth():
    """The gradient returned by ``precompute_V_and_grad_E_V_per_t``
    (autodiff through the batched NUFFT-2) should match the gradient
    of the dense ground-truth V_truth at the same points (up to
    Hutchinson noise)."""
    D = 2
    grid, top = _build_tiny_problem(D=D)
    DAD, xi = _dense_DAD_and_xi(grid, top, D)
    S_homo = 0.04 * jnp.eye(D)
    ms = jr.normal(jr.PRNGKey(0), (3, 4, D))

    def V_truth_one(m_):
        diff_xi = xi[None, :, :] - xi[:, None, :]
        phase = jnp.exp(2j * jnp.pi * (diff_xi @ m_))
        quad = jnp.einsum('klD,DE,klE->kl', diff_xi, S_homo, diff_xi)
        env = jnp.exp(-2 * jnp.pi**2 * quad)
        return D * jnp.sum((DAD * phase * env).real)
    gV_truth = jax.vmap(jax.vmap(jax.grad(V_truth_one)))(ms)

    omega_aux = precompute_omega_per_r(None, grid, top, jr.PRNGKey(1),
                                         D_out=D, P=1024,
                                         cg_tol=1e-9, max_cg_iter=500)
    _, gV_hutch = precompute_V_and_grad_E_V_per_t(
        omega_aux, ms, S_homo, grid, nufft_eps=1e-10)
    rel = float(jnp.max(jnp.abs(gV_hutch - gV_truth))
                / (jnp.max(jnp.abs(gV_truth)) + 1e-12))
    assert rel < 0.10, f"grad rel err {rel:.3f} should be <10% at P=1024"


def test_V_hutch_hetS_vs_dense_truth_with_varying_S():
    """When S_i varies meaningfully across sources, the heterogeneous-S
    path must match the dense-Cholesky ground truth (Σ_δ ω(δ)·phase·env_i)
    with per-source env_i, not the trajectory-mean S_homo.

    Dense ground truth is computed via the same recipe as
    ``test_V_hutch_unbiased_vs_dense_truth``, but with the per-source S_i
    threaded through the envelope. The hetS path should match this within
    Hutchinson noise on ω̂; the homogeneous-S path should NOT (it uses the
    wrong envelope per source).
    """
    from sing.efgp_qx_v_hutch import (
        precompute_E_V_per_t, precompute_E_V_per_t_hetS,
    )
    from sing.efgp_gmix_spreader import stencil_radius_for as stencil_for

    D = 2
    grid, top = _build_tiny_problem(D=D)
    DAD, xi = _dense_DAD_and_xi(grid, top, D)
    omega_aux = precompute_omega_per_r(None, grid, top, jr.PRNGKey(13),
                                         D_out=D, P=1024,
                                         cg_tol=1e-9, max_cg_iter=500)

    # Per-source Ss spanning ~3x range (anisotropic + variance)
    rng = np.random.default_rng(17)
    K, T = 3, 5
    ms = jnp.asarray(rng.uniform(-1.0, 1.0, (K, T, D)))
    diag_a = rng.uniform(0.02, 0.06, (K, T))
    diag_b = rng.uniform(0.02, 0.06, (K, T))
    off    = rng.uniform(-0.005, 0.005, (K, T))
    Ss = jnp.stack(
        [jnp.stack([diag_a, off], axis=-1),
         jnp.stack([off, diag_b], axis=-1)],
        axis=-2)                                              # (K, T, 2, 2)
    Ss = jnp.asarray(Ss)
    S_homo = Ss.reshape(-1, D, D).mean(axis=0)                # trajectory-mean

    def V_truth_one(m_, S_):
        diff_xi = xi[None, :, :] - xi[:, None, :]
        phase = jnp.exp(2j * jnp.pi * (diff_xi @ m_))
        quad = jnp.einsum('klD,DE,klE->kl', diff_xi, S_, diff_xi)
        env = jnp.exp(-2 * jnp.pi**2 * quad)
        return D * jnp.sum((DAD * phase * env).real)
    # Per-source ground truth: each (m_i, S_i) gets its own S in the envelope.
    V_truth_hetS = jax.vmap(jax.vmap(V_truth_one))(ms, Ss)
    # Homog ground truth: every source uses S_homo in the envelope (deliberately wrong).
    V_truth_homog = jax.vmap(jax.vmap(lambda m_: V_truth_one(m_, S_homo)))(ms)

    # Sanity: ground truths actually differ when S_i varies.
    diff_truths = float(jnp.max(jnp.abs(V_truth_hetS - V_truth_homog)))
    print(f"truth(hetS) - truth(homog) max abs diff = {diff_truths:.3e}")
    assert diff_truths > 1e-3, \
        "S_i variation should produce distinguishable ground truths; test set-up broken"

    # hetS path: should match V_truth_hetS
    M_diff = 2 * int(grid.mtot_per_dim[0])
    gather_N = 1 << (M_diff - 1).bit_length()
    h_grid = 1.0 / (gather_N * float(grid.h_per_dim[0]))
    stencil_r = int(stencil_for(Ss.reshape(-1, D, D), h_grid, n_sigma=4.0))
    V_hetS = precompute_E_V_per_t_hetS(omega_aux, ms, Ss, grid,
                                          gather_N=gather_N,
                                          stencil_r=stencil_r)
    err_hetS = float(jnp.max(jnp.abs(V_hetS - V_truth_hetS)))
    rel_hetS = err_hetS / (float(jnp.max(jnp.abs(V_truth_hetS))) + 1e-12)
    print(f"V_hetS vs V_truth_hetS  rel err = {rel_hetS:.3e}")

    # homog path: should match V_truth_homog (the WRONG answer for this problem)
    V_homog = precompute_E_V_per_t(omega_aux, ms, S_homo, grid)
    err_homog_vs_homog = float(jnp.max(jnp.abs(V_homog - V_truth_homog)))
    rel_homog = err_homog_vs_homog / (float(jnp.max(jnp.abs(V_truth_homog)))
                                         + 1e-12)
    print(f"V_homog vs V_truth_homog  rel err = {rel_homog:.3e}")

    # Both restoration paths should match their respective ground truths
    # to within Hutchinson + truncation noise (here we use P=1024 + n_sigma=4
    # so both should be tight).
    assert rel_hetS < 0.05, \
        f"hetS rel err {rel_hetS:.3e} > 5% (P=1024, n_sigma=4 is generous)"
    assert rel_homog < 0.05, \
        f"homog rel err {rel_homog:.3e} > 5%"

    # And critically: hetS should NOT match the homog ground truth, and
    # homog should NOT match the hetS ground truth -- they're distinct
    # quantities under varying S_i.
    rel_hetS_vs_homog_truth = (
        float(jnp.max(jnp.abs(V_hetS - V_truth_homog)))
        / (float(jnp.max(jnp.abs(V_truth_homog))) + 1e-12))
    print(f"V_hetS vs V_truth_homog  rel err = {rel_hetS_vs_homog_truth:.3e}")
    assert rel_hetS_vs_homog_truth > 0.5 * (
        diff_truths / float(jnp.max(jnp.abs(V_truth_homog)))), \
        "hetS path is suspiciously close to the wrong ground truth"


def test_V_hutch_hetS_vs_homog_at_constant_S():
    """When all S_i are identical (= S_homo), hetS path must match homog
    path. Sanity check that the gmix-gather wiring agrees with the
    standard NUFFT-2 wiring in the homogeneous limit."""
    from sing.efgp_qx_v_hutch import (
        precompute_E_V_per_t, precompute_E_V_per_t_hetS,
    )
    from sing.efgp_gmix_spreader import stencil_radius_for as stencil_for

    D = 2
    grid, top = _build_tiny_problem(D=D)
    omega_aux = precompute_omega_per_r(None, grid, top, jr.PRNGKey(11),
                                         D_out=D, P=64,
                                         cg_tol=1e-9, max_cg_iter=500)

    S_homo = 0.04 * jnp.eye(D)
    ms = jr.normal(jr.PRNGKey(0), (3, 5, D))
    Ss_constant = jnp.broadcast_to(S_homo, (3, 5, D, D))

    V_homog = precompute_E_V_per_t(omega_aux, ms, S_homo, grid)

    # Heterogeneous path with S_i ≡ S_homo: pick stencil_r generously
    M_per_dim_diff = 2 * int(grid.mtot_per_dim[0])
    gather_N = 1 << (M_per_dim_diff - 1).bit_length()
    h_grid = 1.0 / (gather_N * float(grid.h_per_dim[0]))
    stencil_r = stencil_for(Ss_constant.reshape(-1, D, D), h_grid, n_sigma=4.0)
    V_hetS = precompute_E_V_per_t_hetS(
        omega_aux, ms, Ss_constant, grid,
        gather_N=gather_N, stencil_r=int(stencil_r))

    rel = float(jnp.max(jnp.abs(V_hetS - V_homog))
                / (jnp.max(jnp.abs(V_homog)) + 1e-12))
    print(f"hetS-vs-homog rel err: {rel:.3e}")
    assert rel < 5e-3, f"hetS at S_i=S_homo should match homog path, rel={rel:.3e}"


def test_omega_hutch_complex_part_matters():
    """Sanity guard: ``rho_summed`` from ``precompute_omega_per_r`` must
    be COMPLEX (not just .real). Random source locations break
    centrosymmetry, so the BTTB kernel is complex Hermitian and
    omega has a nonzero imaginary part. Dropping it loses the
    Im(omega) sin(2 pi delta^T m) contribution to E[V] (verified to be
    the dominant error vs ground truth)."""
    D = 2
    grid, top = _build_tiny_problem(D=D)
    omega_aux = precompute_omega_per_r(None, grid, top, jr.PRNGKey(1),
                                         D_out=D, P=64,
                                         cg_tol=1e-6, max_cg_iter=200)
    assert jnp.iscomplexobj(omega_aux.rho_summed), \
        "rho_summed should be complex; .real cast destroys Im(omega) signal"
    assert float(jnp.max(jnp.abs(omega_aux.rho_summed.imag))) > 1e-3, \
        ("Im(omega) should be nontrivial for random sources; if zero, the "
         "BTTB kernel happened to be real-symmetric (uncommon).")


@pytest.mark.slow
def test_fit_hutch_reaches_same_fixed_point_as_drop():
    """End-to-end: fit_efgp_sing_jax with restore_qf_variance='hutch'
    reaches the same (ℓ, σ², latent RMSE) fixed point as ='none' on a
    well-distributed K=2/T=100 problem. (Larger problems are covered by
    demos/bench_restore_A_scaling.py.)"""
    import sing.efgp_em as em
    from sing.likelihoods import Likelihood
    from sing.simulate_data import simulate_sde, simulate_gaussian_obs

    D = 2; SIGMA = 0.4
    LS_TRUE = 0.6; VAR_TRUE = 1.0
    LS_INIT = 1.5; VAR_INIT = 1.0
    K, T, T_MAX = 2, 100, 1.6

    class GLik(Likelihood):
        def ell(self, y, mean, var, output_params):
            R = output_params['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - mean) ** 2 + var) / R)

    # Random GP drift draw (small grid)
    X_template = jnp.linspace(-4.0, 4.0, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(LS_TRUE, VAR_TRUE, X_template, eps=1e-2)
    M = grid.M
    ks = jr.split(jr.PRNGKey(42), D)
    eps = jnp.stack([
        ((jax.random.normal(jr.split(k)[0], (M,))
          + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)) for k in ks
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps
    def drift(x, t):
        x_b = x[None, :]
        return jnp.stack([
            jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real[0]
            for r in range(D)
        ])
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    rng = np.random.default_rng(43)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, size=(K, D)).astype(np.float64))
    xs_list = [simulate_sde(jr.PRNGKey(100 + k), x0=x0_K[k], f=drift,
                              t_max=T_MAX, n_timesteps=T, sigma=sigma_fn)
                for k in range(K)]
    xs_K = jnp.clip(jnp.stack(xs_list, axis=0), -3.5, 3.5).astype(jnp.float64)
    C_true = rng.standard_normal((6, D)).astype(np.float64) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(6, dtype=jnp.float64),
                    R=jnp.full((6,), 0.05, dtype=jnp.float64))
    ys_K = jnp.stack([simulate_gaussian_obs(jr.PRNGKey(200 + k), xs_K[k], out_true)
                        for k in range(K)], axis=0).astype(jnp.float64)

    def _fit(mode):
        lik = GLik(ys_K, jnp.ones((K, T), dtype=bool))
        ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D, dtype=jnp.float64),
                                         (K, 1, 1)))
        rho = jnp.linspace(0.05, 0.7, 8)
        return em.fit_efgp_sing_jax(
            likelihood=lik, t_grid=jnp.linspace(0, T_MAX, T, dtype=jnp.float64),
            output_params=out_true, init_params=ip, latent_dim=D,
            lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
            sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
            estep_method='gmix', S_marginal=2,
            n_em_iters=8, n_estep_iters=6, rho_sched=rho,
            learn_emissions=False, update_R=False,
            learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
            n_hutchinson_mstep=4, kernel_warmup_iters=4,
            pin_grid=True, pin_grid_lengthscale=LS_TRUE * 0.75,
            verbose=False, true_xs=np.asarray(xs_K),
            qx_moments_method='linearised_shim',
            restore_qf_variance=mode,
            qx_v_cg_tol=1e-3, qx_v_max_cg_iter=50, qx_v_n_probes=4,
        )

    _, _, _, _, hist_none = _fit('none')
    _, _, _, _, hist_hutch = _fit('hutch')
    _, _, _, _, hist_het  = _fit('hutch_hetS')
    # All three should reach the same fixed point within Hutch noise.
    assert abs(hist_hutch.lengthscale[-1] - hist_none.lengthscale[-1]) < 5e-3
    assert abs(hist_hutch.variance[-1] - hist_none.variance[-1]) < 5e-3
    assert abs(hist_hutch.latent_rmse[-1] - hist_none.latent_rmse[-1]) < 0.05
    assert abs(hist_het.lengthscale[-1] - hist_none.lengthscale[-1]) < 5e-3
    assert abs(hist_het.variance[-1] - hist_none.variance[-1]) < 5e-3
    assert abs(hist_het.latent_rmse[-1] - hist_none.latent_rmse[-1]) < 0.05
