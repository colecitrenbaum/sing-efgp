"""
T13: Direct measurement of smoother variance bias.

Hypothesis: the smoother shrinks the empirical velocity variance below the
true GP-drift variance. This biases the marginal likelihood toward small σ².

Direct test: simulate from a known GP drift with σ²_true. Run a smoother
at varying hypers (truth, smaller, larger). Measure:
  - Empirical Var(d_a) = Var(m_{a+1} - m_a) across transitions
  - Empirical Var(true_drift(x_t) * Δt) — what the velocity should be
  - Ratio: how much does the smoother shrink variance?

If smoothed variance is much smaller than true GP-drift variance, the
GP regression will under-estimate σ².

Run:
  ~/myenv/bin/python demos/diag_smoother_bias.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps

    def drift_at(x):
        x_b = x[None, :] if x.ndim == 1 else x
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr)
        return jnp.stack(out, axis=-1)   # (N, D) for batch input

    def drift(x, t):
        return drift_at(x).squeeze(0)

    return drift, drift_at


def setup():
    drift, drift_at = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs_clipped = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs_clipped, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=vt[:D].T, d=ys.mean(0),
                   R=jnp.full((N_OBS,), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs_clipped, ys, lik, op_init, ip_init, t_grid, drift_at


def converge_smoother(lik, op, ip, t_grid, ls_pin, var_pin, n_em=12):
    """Fixed-hyper EFGP EM to converge q(x) at given (ls, var)."""
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    mp, *_ = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_pin, variance=var_pin, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False, learn_kernel=False,
        kernel_warmup_iters=n_em, verbose=False)
    return mp


def measure(label, xs_true, mp, t_grid, drift_at):
    """Compute several variance statistics."""
    ms = np.asarray(mp['m'][0])         # (T, D)
    Ss = np.asarray(mp['S'][0])         # (T, D, D)
    SSs = np.asarray(mp['SS'][0])       # (T-1, D, D) cross-cov
    del_t = np.asarray(t_grid[1:] - t_grid[:-1])  # (T-1,)

    # Smoothed velocities: d_a = m_{a+1} - m_a
    d_smooth = ms[1:] - ms[:-1]   # (T-1, D)
    velocity_smooth = d_smooth / del_t[:, None]   # (T-1, D)

    # True drift evaluated at smoothed locations: f(m_a)
    drift_at_m = np.asarray(drift_at(jnp.asarray(ms[:-1])))   # (T-1, D)
    # True velocity expectation: f(m_a) (Δ-scaled)
    # The variance of the true velocity per step is dominated by σ_drift²/Δt

    # Posterior smoother variance
    var_S_diag = np.diagonal(Ss, axis1=1, axis2=2)   # (T, D)

    # Reported numbers per output dim r (averaged across t)
    print(f"\n  [{label}]", flush=True)
    print(f"    avg |S_diag| over t,r:           "
          f"{var_S_diag.mean():.5f}  (smoother posterior variance)",
          flush=True)
    print(f"    Var of smoothed velocities d_a/Δt: "
          f"{velocity_smooth.var(axis=0)} (per dim, sample std≈"
          f"{velocity_smooth.std()/math.sqrt(2):.3f})",
          flush=True)
    print(f"    Var of true-drift @ m_a:           "
          f"{drift_at_m.var(axis=0)} (per dim, sample std="
          f"{drift_at_m.std():.3f})",
          flush=True)
    print(f"    Implied 'visible σ²' from velocity: "
          f"{velocity_smooth.var(axis=0).mean():.4f}",
          flush=True)
    print(f"    Truth σ² = {VAR_TRUE} (the GP variance scaled by 1)",
          flush=True)


def main():
    print(f"[T13] Smoother variance bias test", flush=True)
    print(f"[T13] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)
    xs, ys, lik, op_init, ip_init, t_grid, drift_at = setup()

    # True trajectory variance for reference
    xs_np = np.asarray(xs)
    print(f"\n  [TRUE TRAJECTORY (no smoothing)]", flush=True)
    raw_d = xs_np[1:] - xs_np[:-1]
    raw_vel = raw_d / np.asarray(t_grid[1:] - t_grid[:-1])[:, None]
    print(f"    Var of raw velocities (xs[a+1]-xs[a])/Δt: "
          f"{raw_vel.var(axis=0)} (per dim, sample std="
          f"{raw_vel.std():.3f})", flush=True)
    drift_at_xs = np.asarray(drift_at(jnp.asarray(xs_np[:-1])))
    print(f"    Var of true-drift @ xs[:-1]:               "
          f"{drift_at_xs.var(axis=0)} (per dim, sample std="
          f"{drift_at_xs.std():.3f})", flush=True)

    # Smoother at TRUE hypers
    print(f"\n[T13] Smoother at TRUE hypers...", flush=True)
    mp_truth = converge_smoother(lik, op_init, ip_init, t_grid,
                                   LS_TRUE, VAR_TRUE, n_em=12)
    measure("smoother @ truth", xs, mp_truth, t_grid, drift_at)

    # Smoother at smaller σ² (under-estimate)
    print(f"\n[T13] Smoother at σ²=0.3 (under-estimate)...", flush=True)
    mp_under = converge_smoother(lik, op_init, ip_init, t_grid,
                                   LS_TRUE, 0.3, n_em=12)
    measure("smoother @ σ²=0.3", xs, mp_under, t_grid, drift_at)

    # Smoother at larger σ² (over-estimate)
    print(f"\n[T13] Smoother at σ²=3.0 (over-estimate)...", flush=True)
    mp_over = converge_smoother(lik, op_init, ip_init, t_grid,
                                  LS_TRUE, 3.0, n_em=12)
    measure("smoother @ σ²=3.0", xs, mp_over, t_grid, drift_at)


if __name__ == '__main__':
    main()
