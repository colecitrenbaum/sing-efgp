"""Interface test (J): EFGPDrift and SparseGP must agree on q(f) when handed
the SAME q(x).

The two drift methods solve the *same* underlying problem — Stein-corrected
GP regression of latent velocities given smoother marginals (m, S, SS) and
the per-step Δt — using two different finite approximations:

  * SparseGP:  inducing-point Titsias posterior with closed-form Gauss-Hermite
               kernel expectations.
  * EFGPDrift: equispaced Fourier features with Stein-corrected NUFFT RHS,
               BTTB Toeplitz matvec, conjugate gradients.

If the SING ↔ EFGP interface (SS convention, Δt placement, σ_drift² scaling,
Stein-correction sign) is wired correctly, then for the *same* q(x) both
posteriors must approximate the same dense GP-regression answer. They will
differ by:

  * Inducing-grid truncation (SparseGP)
  * Fourier truncation + S_marginal MC noise (EFGP)

Both effects are small at our test settings (8×8 inducing pts; m=16 Fourier
modes/dim; S_marginal=8) on a smooth, in-RBF-span linear drift. So we expect
the cross-method drift RMSE on a held-in test grid to be MUCH smaller than
either method's RMSE vs. the true drift (which is dominated by ℓ-mismatch
and finite data, not by the regression solver).

What this test catches:
  - SS convention bug (Cov(x_t, x_{t+1}) vs Cov(x_{t+1}, x_t))
  - Δt placement / scaling
  - σ_drift² placement
  - Stein adjoint Jacobian sign error (regression test for the May-2026 fix)
  - Spectral-grid weight convention (sqrt(S(ξ) · h^d))

What this test does NOT catch:
  - M-step kernel-gradient bugs (covered by test_collapsed_mstep_gradient_*)
  - ELBO sign errors (covered by future ELBO-monotonicity test)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))
_GP_QUAD = Path.home() / "Documents" / "GPs" / "gp-quadrature"
if str(_GP_QUAD) not in sys.path:
    sys.path.insert(0, str(_GP_QUAD))

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Gaussian
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd


def _eval_sparse_jacobian(sparse_drift, drift_params, gp_post, X_eval):
    """Vectorize SparseGP.dfdx over X_eval.  Returns (N_eval, D, D).

    Note: SparseGP.dfdx ignores S so we pass tiny S_zero. The expectation
    of J_f(x) at fixed x = m collapses to the deterministic Jacobian.
    """
    D = X_eval.shape[1]
    eps_S = jnp.eye(D) * 1e-9
    eval_one = lambda x: sparse_drift.dfdx(
        drift_params, jr.PRNGKey(0), 0.0, x, eps_S, gp_post)
    return jax.vmap(eval_one)(X_eval)


def _setup(T=300, D=2, ls_true=1.0, sigma=0.3, seed=7):
    """Generate a small, well-conditioned linear-drift SDE problem."""
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    t_max = 3.0
    xs = simulate_sde(jr.PRNGKey(seed), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    rng = np.random.default_rng(seed)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(seed + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, out_true, t_max


@pytest.mark.slow
def test_efgp_sparsegp_drift_posterior_agrees_at_fixed_qx():
    """Hand the SAME (m, S, SS) to both q(f) updates; compare posterior mean
    and Jacobian on a held-in test grid."""
    D = 2
    T = 300
    sigma = 0.3
    ls = 1.0           # truth, also what we pin both kernels at
    var_kernel = 1.0   # σ²_kernel
    n_em = 12          # converge the smoother
    n_estep = 6

    xs, ys, lik, out_true, t_max = _setup(T=T, D=D, ls_true=ls, sigma=sigma)
    t_grid = jnp.linspace(0., t_max, T)
    del_t = t_grid[1:] - t_grid[:-1]

    # Init shared across runs
    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    output_params_init = dict(C=Vt[:D].T, d=ys.mean(0),
                              R=jnp.full((ys.shape[1],), 0.1))
    init_params = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    # ---- 1. Run SparseGP to converge a smoother ----
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)            # 64 inducing pts
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_drift_params = dict(length_scales=jnp.full((D,), float(ls)),
                            output_scale=jnp.asarray(math.sqrt(var_kernel)))
    rho_sched = jnp.linspace(0.05, 0.4, n_em)
    lr_sched = jnp.full((n_em,), 0.05)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sp_drift_params,
        init_params=init_params, output_params=output_params_init,
        sigma=sigma,
        rho_sched=rho_sched,
        # n_iters_m must be ≥ 1 because lax.cond traces both branches and the
        # untaken M-step branch contains a lax.scan that fails with 0 iters.
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=lr_sched,
        print_interval=999,
    )

    # ---- 2. Take the converged (m, S, SS) ----
    ms = mp['m'][0]                     # (T, D)
    Ss = mp['S'][0]                     # (T, D, D)
    SSs = mp['SS'][0]                   # (T-1, D, D)
    print(f"\n[interface-J] smoother: m std={np.asarray(ms).std(0)}, "
          f"|S|_avg={float(jnp.linalg.det(Ss).mean()):.2e}")

    # ---- 3. Build q(f) under BOTH methods at the SAME hypers ----
    # SparseGP path
    gp_post_sp = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp, jnp.ones((1, T), dtype=bool),
        sp_drift_params,
        jnp.zeros((1, T, 1)),
        jnp.zeros((D, 1)),
        sigma,
    )

    # EFGP path — build a Fourier grid covering the data cloud
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid_jax = jp.spectral_grid_se(ls, var_kernel, X_template, eps=1e-3)
    print(f"[interface-J] EFGP grid: M={grid_jax.M}, "
          f"mtot_per_dim={grid_jax.mtot_per_dim}")
    trial_mask_K1 = jnp.ones((1, T), dtype=bool)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms[None], Ss[None], SSs[None], del_t, trial_mask_K1)
    mu_r, _, _ = jpd.compute_mu_r_jax(
        m_src, S_src, d_src, C_src, w_src, grid_jax, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D,
        cg_tol=1e-7, max_cg_iter=4 * grid_jax.M,
    )

    # ---- 4. Eval both posteriors on a held-in test grid ----
    ms_np = np.asarray(ms)
    grid_lo = ms_np.min(0) + 0.2 * (ms_np.max(0) - ms_np.min(0))
    grid_hi = ms_np.max(0) - 0.2 * (ms_np.max(0) - ms_np.min(0))
    n_per = 10
    g0 = np.linspace(grid_lo[0], grid_hi[0], n_per)
    g1 = np.linspace(grid_lo[1], grid_hi[1], n_per)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    X_eval_j = jnp.asarray(X_eval)

    # SparseGP mean and Jacobian
    f_sp = np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post_sp, sp_drift_params, X_eval_j))                   # (N, D)
    J_sp = np.asarray(_eval_sparse_jacobian(
        sparse_drift, sp_drift_params, gp_post_sp, X_eval_j))     # (N, D, D)

    # EFGP mean and Jacobian (drift_moments_jax evaluates Φ(X) μ_r)
    f_efgp_K, _, J_efgp_K = jpd.drift_moments_jax(
        mu_r, grid_jax, X_eval_j[None], D_lat=D, D_out=D)
    f_efgp = np.asarray(f_efgp_K[0])                              # (N, D)
    J_efgp = np.asarray(J_efgp_K[0])                              # (N, D, D)

    # ---- 5. Compare ----
    rmse_f = float(np.sqrt(np.mean((f_sp - f_efgp) ** 2)))
    scale_f = float(np.sqrt(np.mean(f_sp ** 2)))
    rmse_J = float(np.sqrt(np.mean((J_sp - J_efgp) ** 2)))
    scale_J = float(np.sqrt(np.mean(J_sp ** 2)))

    # Reference: each method's mean vs. the true linear drift on the grid
    f_true_grid = X_eval @ np.diag([-1.0, -1.5])
    rmse_sp_truth = float(np.sqrt(np.mean((f_sp - f_true_grid) ** 2)))
    rmse_efgp_truth = float(np.sqrt(np.mean((f_efgp - f_true_grid) ** 2)))

    print(f"[interface-J] f scale (SparseGP) = {scale_f:.4f}")
    print(f"[interface-J] f cross-method RMSE = {rmse_f:.4f} "
          f"({100*rmse_f/scale_f:.1f}% of scale)")
    print(f"[interface-J] J scale (SparseGP) = {scale_J:.4f}")
    print(f"[interface-J] J cross-method RMSE = {rmse_J:.4f} "
          f"({100*rmse_J/scale_J:.1f}% of scale)")
    print(f"[interface-J]   for context: SparseGP-vs-truth RMSE = "
          f"{rmse_sp_truth:.4f}, EFGP-vs-truth RMSE = {rmse_efgp_truth:.4f}")

    # ---- Assertions ----
    # Cross-method agreement on E[f]: should be a small fraction of scale.
    # Both methods are approximations of the same dense GP regression posterior;
    # at this grid resolution they should match to ~10% relative.
    assert rmse_f / scale_f < 0.15, (
        f"E[f] cross-method RMSE {rmse_f:.4f} = "
        f"{100*rmse_f/scale_f:.1f}% of f-scale exceeds 15% — likely "
        f"interface bug (SS convention, Δt scaling, or Stein sign).")

    # Cross-method agreement on E[J_f]: looser threshold (Jacobian has more
    # MC noise from Stein correction and finite-difference of GH for SparseGP).
    assert rmse_J / scale_J < 0.30, (
        f"E[J_f] cross-method RMSE {rmse_J:.4f} = "
        f"{100*rmse_J/scale_J:.1f}% of J-scale exceeds 30% — likely "
        f"Stein-correction sign or frequency-multiplication bug.")

    # Sanity: cross-method agreement should be tighter than either-vs-truth,
    # i.e. the methods agree with each other better than they agree with truth
    # (because both are imperfect estimators of the SAME Bayesian posterior,
    # which itself is biased away from truth at finite T).
    assert rmse_f < min(rmse_sp_truth, rmse_efgp_truth), (
        f"Cross-method E[f] disagreement ({rmse_f:.4f}) exceeds vs-truth "
        f"({rmse_sp_truth:.4f}, {rmse_efgp_truth:.4f}) — methods diverge "
        f"more than either misses truth, suggesting interface bug.")
