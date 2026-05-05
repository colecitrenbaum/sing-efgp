"""
T4: Robustness of the local-quadratic transition approximation in q(x).

Background. SING-EFGP exposes the GP-drift moments to SING's natural-grad
q(x) update via three custom-VJP shims (``ef_with_jac_grad``,
``eff_with_grads``, ``FrozenEFGPDrift.dfdx``) that approximate
expectations over the marginal q(x_t) = N(m_t, S_t) by:

    E_q[f(x)]      ≈ f(m)
    E_q[||f||²]    ≈ ||f(m)||²              (with grad_S = J_f^T J_f)
    E_q[J_f(x)]    ≈ J_f(m)

This is a first-order Taylor expansion around m. It misses the
curvature term  ½ Tr(H_f(m) S)  for E[f] and the S-dependence of
E[J_f] entirely.

Question. How much does this approximation cost on real q(x)?  We
compare the linearized values against Monte Carlo "ground-truth"
expectations  E_q[·] = (1/N) Σ_n ·(x^(n)),  x^(n) ~ N(m_t, S_t),
N=large, on three regimes (linear, Duffing, anharmonic) and across
two |S| levels (smoother S, and 4× larger S to stress-test).

Run:
  ~/myenv/bin/python demos/diag_linearization_robustness.py
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


D = 2
T = 200
SIGMA = 0.3
N_OBS = 6
LS = 1.0
VAR = 1.0
N_EM = 12
N_ESTEP = 6
N_MC = 256                # MC samples per time step for "truth" expectation
SEED = 7

REGIMES = ["linear", "duffing", "anharmonic"]
S_SCALES = [1.0, 4.0]     # 1× = smoother S; 4× = stress test


def drift_for(regime):
    if regime == "linear":
        A_true = jnp.diag(jnp.array([-1.0, -1.5]))
        return lambda x, t: A_true @ x
    if regime == "duffing":
        return lambda x, t: jnp.array(
            [x[1], 1.0 * x[0] - 1.0 * x[0]**3 - 0.5 * x[1]])
    if regime == "anharmonic":
        return lambda x, t: jnp.array(
            [x[1], -x[0] - 0.3 * x[1] - 0.5 * x[0]**3])
    raise ValueError(regime)


def setup(regime):
    drift_true = drift_for(regime)
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 3.0 if regime == "linear" else T * 0.05
    x0 = jnp.array([1.0, -0.6]) if regime != "duffing" else jnp.array([1.5, 0.0])
    xs = simulate_sde(jr.PRNGKey(SEED), x0=x0, f=drift_true,
                      t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, t_max


def converge_smoother(lik, t_max):
    yc = lik.ys_obs[0] - lik.ys_obs[0].mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=Vt[:D].T, d=lik.ys_obs[0].mean(0),
                   R=jnp.full((lik.ys_obs.shape[2],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(LS)),
                 output_scale=jnp.asarray(math.sqrt(VAR)))
    rho_sched = jnp.linspace(0.05, 0.4, N_EM)
    t_grid = jnp.linspace(0., t_max, T)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=ip_init, output_params=op_init, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((N_EM,), 0.05), print_interval=999,
    )
    return mp, t_grid


def build_efgp_qf(mp, t_grid):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
        K_per_dim=12, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-7, max_cg_iter=4 * grid.M,
    )
    return grid, mu_r


def linearized_moments(grid, mu_r, ms):
    """E_q[f] ≈ f(m), E_q[J_f] ≈ J_f(m). Returns (Ef_lin, EJ_lin) of
    shapes (T, D), (T, D, D)."""
    Ef, _, EJ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(ms), D_lat=D, D_out=D)
    return np.asarray(Ef), np.asarray(EJ)


def mc_moments(grid, mu_r, ms, Ss, *, n_mc, key, S_scale=1.0):
    """E_q[f] and E_q[J_f] via N MC samples per time step. Returns
    (Ef_mc, EJ_mc, Ef_var_mc) — Ef_var is the per-time MC stderr (rough)."""
    T_local = ms.shape[0]
    Ss_eff = Ss * S_scale
    L = jnp.linalg.cholesky(Ss_eff + 1e-9 * jnp.eye(D))   # (T, D, D)
    eps = jr.normal(key, (n_mc, T_local, D))
    # x[s, t] = m[t] + L[t] @ eps[s, t]
    x_samples = ms[None, :, :] + jnp.einsum('tij,stj->sti', L, eps)
    x_flat = x_samples.reshape(n_mc * T_local, D)
    Ef_flat, _, EJ_flat = jpd.drift_moments_jax(
        mu_r, grid, x_flat, D_lat=D, D_out=D)
    Ef_per = Ef_flat.reshape(n_mc, T_local, D)
    EJ_per = EJ_flat.reshape(n_mc, T_local, D, D)
    Ef_mean = np.asarray(Ef_per.mean(axis=0))
    EJ_mean = np.asarray(EJ_per.mean(axis=0))
    Ef_se = np.asarray(Ef_per.std(axis=0) / math.sqrt(n_mc))
    return Ef_mean, EJ_mean, Ef_se


def report(regime, S_scale, ms_np, Ss_np, Ef_lin, EJ_lin, Ef_mc, EJ_mc, Ef_se):
    """Compute % differences and print a short summary."""
    rms_f = float(np.sqrt(np.mean((Ef_lin - Ef_mc) ** 2)))
    scale_f = float(np.sqrt(np.mean(Ef_mc ** 2)) + 1e-30)
    rms_J = float(np.sqrt(np.mean((EJ_lin - EJ_mc) ** 2)))
    scale_J = float(np.sqrt(np.mean(EJ_mc ** 2)) + 1e-30)
    rel_f = 100 * rms_f / scale_f
    rel_J = 100 * rms_J / scale_J
    mean_se_f = float(np.mean(Ef_se))
    avg_S_diag = float(np.mean(np.diagonal(Ss_np, axis1=1, axis2=2)))
    max_S_diag = float(np.max(np.diagonal(Ss_np, axis1=1, axis2=2)))
    print(f"  {regime:11s}  S_scale={S_scale:>3.1f}  "
          f"avg|S_diag|={avg_S_diag:.4f}  max|S_diag|={max_S_diag:.4f}  "
          f"|| ||  E[f] rel={rel_f:5.2f}%  "
          f"E[J] rel={rel_J:5.2f}%  "
          f"(mc_se_f={mean_se_f:.4f})", flush=True)
    return dict(regime=regime, S_scale=S_scale,
                rel_f=rel_f, rel_J=rel_J,
                avg_S_diag=avg_S_diag, max_S_diag=max_S_diag)


def main():
    print(f"[T4] JAX devices: {jax.devices()}", flush=True)
    print(f"[T4] D={D}, T={T}, n_mc={N_MC}", flush=True)
    rows = []
    for regime in REGIMES:
        print(f"\n[T4] === regime: {regime} ===", flush=True)
        xs, ys, lik, t_max = setup(regime)
        mp, t_grid = converge_smoother(lik, t_max)
        ms = mp['m'][0]; Ss = mp['S'][0]
        ms_np = np.asarray(ms); Ss_np = np.asarray(Ss)
        print(f"  smoother m std: {ms_np.std(axis=0)}, "
              f"|S_diag| mean={np.mean(np.diagonal(Ss_np, axis1=1, axis2=2)):.4f}",
              flush=True)
        grid, mu_r = build_efgp_qf(mp, t_grid)
        Ef_lin, EJ_lin = linearized_moments(grid, mu_r, ms)

        for S_scale in S_SCALES:
            Ef_mc, EJ_mc, Ef_se = mc_moments(
                grid, mu_r, ms, Ss, n_mc=N_MC,
                key=jr.PRNGKey(0), S_scale=S_scale)
            rows.append(report(regime, S_scale, ms_np, Ss_np * S_scale,
                                Ef_lin, EJ_lin, Ef_mc, EJ_mc, Ef_se))

    print("\n[T4] Summary table:", flush=True)
    print(f"  {'regime':12s}  {'S×':>4s}  {'avg S_diag':>10s}  "
          f"{'max S_diag':>10s}  {'rel E[f]%':>9s}  {'rel E[J]%':>9s}",
          flush=True)
    for r in rows:
        print(f"  {r['regime']:12s}  {r['S_scale']:>4.1f}  "
              f"{r['avg_S_diag']:>10.4f}  {r['max_S_diag']:>10.4f}  "
              f"{r['rel_f']:>9.2f}  {r['rel_J']:>9.2f}", flush=True)

    print("\n[T4] Interpretation:", flush=True)
    print("  - E[f] linearization error ∝ ½ Tr(H_f S) — grows with |S| × curvature.",
          flush=True)
    print("  - E[J_f] error grows with |S| × Hessian-of-Jacobian.", flush=True)
    print("  - Acceptable if rel < ~10% across all rows; concerning if > 25% on",
          flush=True)
    print("    realistic |S| (S_scale=1.0) for any nonlinear regime.", flush=True)


if __name__ == "__main__":
    main()
