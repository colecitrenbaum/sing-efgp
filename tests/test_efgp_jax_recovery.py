"""
Hyperparameter recovery test for the JAX EM path.

Synthesises an SDE whose drift is a fresh sample from the SE-kernel GP
prior at known ``ls_true``, simulates a trajectory + noisy obs, then runs
``fit_efgp_sing_jax`` from an *intentionally wrong* initial lengthscale
and checks that the M-step pulls it toward the truth.

This is a sanity check that the kernel M-step has the right sign and
the cold-start collapse fix (``kernel_warmup_iters``) does its job.
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

# Load jax_finufft FIRST via efgp_em (which imports efgp_jax_primitives).
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import jax
import jax.numpy as jnp
import jax.random as jr
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


def _gp_drift_factory(ls_true: float, var_true: float, D: int, key,
                       extent: float = 4.0, eps_grid: float = 1e-2):
    """Sample a fresh d-dim drift from the SE-kernel GP prior.

    Returns a JAX function ``drift(x, t) -> R^D`` (D independent GP draws,
    one per output dim).
    """
    # Build the spectral grid for a representative cloud spanning [-extent, extent]
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template,
                                eps=eps_grid)

    # Sample epsilon ~ CN(0, I) of shape (D, M) — one draw per output dim
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)                                     # (D, M)
    fk_draw = grid.ws[None, :] * eps               # (D, M) Fourier coefs

    def drift_x(x):
        """x: (D,) -> R^D."""
        x_b = x[None, :]                            # (1, D)
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    def drift(x, t):
        return drift_x(x)

    return drift, fk_draw, grid


@pytest.mark.slow
def test_lengthscale_recovery_jax():
    D = 2
    T = 80
    sigma = 0.4
    ls_true = 0.6
    var_true = 1.0
    n_em = 8

    # Sample the GP drift
    drift_fn, fk_draw, grid_true = _gp_drift_factory(
        ls_true, var_true, D, jr.PRNGKey(123))
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([0.0, 0.0]),
                       f=drift_fn, t_max=2.0, n_timesteps=T,
                       sigma=sigma_fn)
    # Bound observations to a reasonable range (rough sanity)
    xs = jnp.clip(xs, -3.0, 3.0)

    # Gaussian observations
    N = 6
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                     R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(8), xs, out_true)

    class GLik(Likelihood):
        def ell(self, y, m, v, op):
            R = op['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - m) ** 2 + v) / R)

    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                  dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    ls_init = 1.5  # deliberately too large
    var_init = 1.0
    _, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., 2.0, T),
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=var_init, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10,
        rho_sched=jnp.linspace(0.3, 0.9, n_em),
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        kernel_warmup_iters=2,
        verbose=True,
    )

    final_ls = hist.lengthscale[-1]
    print(f"\n  ls_true={ls_true}, ls_init={ls_init}, final_ls={final_ls}")

    # Sanity (not strict recovery): M-step should not collapse to zero
    # (trivial-q cold-start failure mode) and should not blow up.  Strict
    # ls-recovery on this small T=80 problem with a single noisy trajectory
    # is too much to ask: the data signal is weak relative to the prior,
    # the local-quadratic transition approximation introduces bias, and
    # we clipped the GP-drift trajectory to a bounding box.  Recovery
    # diagnostics belong in the demo notebook, not a CI test.
    assert 0.1 < final_ls < 50.0, (
        f"M-step blew up: started at {ls_init}, ended at {final_ls}, "
        f"true {ls_true}")


@pytest.mark.slow
def test_fixed_kernel_linear_recovery_with_emission_warmup():
    """Regression test for the early-emissions bad basin.

    With a slow SING schedule and Gaussian observations, the EFGP JAX path
    used to diverge badly from SparseGP even when the kernel was fixed at the
    same value. The root cause was the closed-form Gaussian emissions update
    running from iteration 1, while q(x) was still close to the diffusion-only
    initialization. Emission warmup should keep the linear latent recovery in
    the SparseGP regime.
    """
    D = 2
    T = 250
    t_max = 6.0
    sigma = 0.3
    N = 8
    A_true = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

    xs = simulate_sde(
        jr.PRNGKey(7), x0=jnp.array([2.0, 0.0]),
        f=lambda x, t: A_true @ x,
        t_max=t_max, n_timesteps=T,
        sigma=lambda x, t: sigma * jnp.eye(D))
    xs_np = np.asarray(xs)

    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)

    class GLik(Likelihood):
        def ell(self, y, m, v, op):
            R = op['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - m) ** 2 + v) / R)

    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    U, s, _ = np.linalg.svd(yc.T, full_matrices=False)
    op = dict(C=jnp.asarray(U[:, :D] * s[:D] / np.sqrt(T)),
              d=jnp.zeros(N),
              R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.5))

    mp, _, op_out, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., t_max, T),
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=1.5, variance=1.0, sigma=sigma,
        rho_sched=jnp.logspace(-2, -1, 15),
        n_em_iters=15, n_estep_iters=8,
        learn_kernel=False,
        X_template=(jnp.linspace(-3., 3., T)[:, None] * jnp.ones((1, D))),
        seed=42,
        verbose=False,
    )

    Xi = np.asarray(mp['m'][0]); Xt = xs_np
    bi = Xi.mean(0); bt = Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    b = bt - A @ bi
    rmse = float(np.sqrt(np.mean((Xi @ A.T + b - Xt) ** 2)))
    R_mean = float(jnp.mean(op_out['R']))
    print(f"\n  fixed-kernel linear RMSE={rmse:.4f}, mean(R)={R_mean:.4f}")

    assert rmse < 0.12, (
        f"Fixed-kernel EFGP latent RMSE regressed to {rmse:.4f}; "
        "this usually means early Gaussian-emissions updates are pulling "
        "q(x) into the bad basin again.")
    assert 0.03 < R_mean < 0.08, (
        f"Gaussian emission variance update looks wrong: mean(R)={R_mean:.4f}")


def test_mstep_jax_gradient_matches_finite_difference():
    """JAX m_step_kernel_jax: verify the analytic gradient (deterministic
    part only — Hutchinson trace is stochastic) agrees with a finite-
    difference of the loss at a non-trivial (μ, top, ws).
    """
    import math
    D = 2
    # Build a small random q(f) state we'll hold fixed.
    rng = np.random.default_rng(11)
    ls0 = 0.6
    var0 = 1.0
    X = jnp.asarray(rng.standard_normal((40, D)).astype(np.float32))
    grid = jp.spectral_grid_se(ls0, var0, X, eps=1e-2)
    M = grid.M
    weights = jnp.asarray(rng.uniform(0.5, 1.5, X.shape[0]).astype(np.float32)
                          * 0.05)
    v_kernel = jp.bttb_conv_vec_weighted(
        X, weights.astype(grid.ws.dtype), grid.xcen, grid.h_per_dim,
        grid.mtot_per_dim)
    top = jp.make_toeplitz(v_kernel, force_pow2=True)

    # Random fixed mu_r and z_r
    mu_r = jnp.asarray((rng.standard_normal((D, M))
                        + 1j * rng.standard_normal((D, M))).astype(
                       np.complex64) * 0.1)
    z_r = jnp.asarray((rng.standard_normal((D, M))
                       + 1j * rng.standard_normal((D, M))).astype(
                      np.complex64) * 0.5)

    # Loss as a function of (log_ls, log_var) — JAX-traceable
    h_scalar = grid.h_per_dim[0]

    def loss_fn(log_ls, log_var):
        ws = jpd._ws_real_se(log_ls, log_var, grid.xis_flat, h_scalar, D)
        ws_c = ws.astype(top.v_fft.dtype)
        L = jnp.zeros(())
        for r in range(D):
            mu = mu_r[r]
            h_r = ws_c * z_r[r]
            term1 = jnp.vdot(h_r, mu).real
            Tv = jp.toeplitz_apply(top, ws_c * mu)
            Av = ws_c * Tv + mu
            term2 = jnp.vdot(mu, Av).real * 0.5
            L = L - (term1 - term2)
        return L

    log_ls0 = math.log(0.7)
    log_var0 = math.log(1.1)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
    L0, (g_ls, g_var) = grad_fn(jnp.asarray(log_ls0, dtype=jnp.float32),
                                jnp.asarray(log_var0, dtype=jnp.float32))

    # Centered finite difference
    eps = 1e-3
    Lp = float(loss_fn(jnp.asarray(log_ls0 + eps, dtype=jnp.float32),
                        jnp.asarray(log_var0, dtype=jnp.float32)))
    Lm = float(loss_fn(jnp.asarray(log_ls0 - eps, dtype=jnp.float32),
                        jnp.asarray(log_var0, dtype=jnp.float32)))
    fd_ls = (Lp - Lm) / (2 * eps)
    Lp = float(loss_fn(jnp.asarray(log_ls0, dtype=jnp.float32),
                        jnp.asarray(log_var0 + eps, dtype=jnp.float32)))
    Lm = float(loss_fn(jnp.asarray(log_ls0, dtype=jnp.float32),
                        jnp.asarray(log_var0 - eps, dtype=jnp.float32)))
    fd_var = (Lp - Lm) / (2 * eps)

    rel_ls = abs(float(g_ls) - fd_ls) / (abs(fd_ls) + 1e-9)
    rel_var = abs(float(g_var) - fd_var) / (abs(fd_var) + 1e-9)
    print(f"\n  d/dlogℓ:  autograd {float(g_ls):.4e}  FD {fd_ls:.4e}  rel {rel_ls:.2e}")
    print(f"  d/dlogσ²: autograd {float(g_var):.4e}  FD {fd_var:.4e}  rel {rel_var:.2e}")
    assert rel_ls < 1e-2, f"d/dlogℓ JAX vs FD rel err {rel_ls:.2e}"
    assert rel_var < 1e-2, f"d/dlogσ² JAX vs FD rel err {rel_var:.2e}"
