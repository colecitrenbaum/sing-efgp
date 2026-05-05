"""
T1b: Cross-method drift agreement at PRODUCTION tolerances.

The committed test ``tests/test_efgp_sparsegp_drift_agreement.py`` runs
with very tight settings (``cg_tol=1e-7``, ``S_marginal=8``,
``X_template m=16``) — much tighter than the notebook actually uses
(``cg_tol=1e-5``, ``S_marginal=2``, ``K=auto``). This script mirrors
the test setup but at the looser production tolerances, to verify that
EFGP and SparseGP still agree on the drift posterior at the settings
real users hit.

If agreement at production tolerances is significantly worse than the
1e-7 reference (say RMSE > 25% of E[f] scale or > 50% of E[J_f] scale),
that's evidence that production tolerances are too loose and need to be
tightened for correctness — independent of any optimizer/protocol issue.

Run:
  ~/myenv/bin/python demos/diag_drift_agreement_production.py
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
T = 300
SIGMA = 0.3
LS = 1.0
VAR = 1.0
N_EM = 12
N_ESTEP = 6
SEED = 7


def _eval_sparse_jacobian(sparse_drift, drift_params, gp_post, X_eval):
    eps_S = jnp.eye(D) * 1e-9
    eval_one = lambda x: sparse_drift.dfdx(
        drift_params, jr.PRNGKey(0), 0.0, x, eps_S, gp_post)
    return jax.vmap(eval_one)(X_eval)


def setup():
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 3.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, out_true, t_max


def run_setting(*, S_marginal, cg_tol, qf_nufft_eps, K_per_dim,
                eps_grid, label):
    """Run agreement check at one set of tolerances. Returns dict of metrics."""
    xs, ys, lik, _, t_max = setup()
    t_grid = jnp.linspace(0., t_max, T)
    del_t = t_grid[1:] - t_grid[:-1]

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    output_params_init = dict(C=Vt[:D].T, d=ys.mean(0),
                              R=jnp.full((ys.shape[1],), 0.1))
    init_params = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    # SparseGP smoother
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(LS)),
                 output_scale=jnp.asarray(math.sqrt(VAR)))
    rho_sched = jnp.linspace(0.05, 0.4, N_EM)
    lr_sched = jnp.full((N_EM,), 0.05)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=init_params, output_params=output_params_init,
        sigma=SIGMA, rho_sched=rho_sched,
        n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=lr_sched, print_interval=999,
    )
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]

    # SparseGP q(f)
    gp_post_sp = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp, jnp.ones((1, T), dtype=bool),
        sp_dp, jnp.zeros((1, T, 1)), jnp.zeros((D, 1)), SIGMA)

    # EFGP grid at this K
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    if K_per_dim is None:
        # auto: same path the EM driver takes
        grid_init = jp.spectral_grid_se(LS, VAR, X_template, eps=eps_grid)
        K_per_dim = (int(grid_init.mtot_per_dim[0]) - 1) // 2
        X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
        xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                           dtype=jnp.float32)
        grid = jp.spectral_grid_se_fixed_K(
            log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
            log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
            K_per_dim=K_per_dim, X_extent=X_extent, xcen=xcen, d=D,
            eps=eps_grid,
        )
    else:
        X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
        xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                           dtype=jnp.float32)
        grid = jp.spectral_grid_se_fixed_K(
            log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
            log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
            K_per_dim=K_per_dim, X_extent=X_extent, xcen=xcen, d=D,
            eps=eps_grid,
        )

    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=S_marginal,
        D_lat=D, D_out=D, cg_tol=cg_tol, max_cg_iter=4 * grid.M,
        nufft_eps=qf_nufft_eps,
    )

    # Eval grid (drop the boundary)
    ms_np = np.asarray(ms)
    grid_lo = ms_np.min(0) + 0.2 * (ms_np.max(0) - ms_np.min(0))
    grid_hi = ms_np.max(0) - 0.2 * (ms_np.max(0) - ms_np.min(0))
    n_per = 10
    g0 = np.linspace(grid_lo[0], grid_hi[0], n_per)
    g1 = np.linspace(grid_lo[1], grid_hi[1], n_per)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    X_eval_j = jnp.asarray(X_eval)

    f_sp = np.asarray(sparse_drift.get_posterior_f_mean(gp_post_sp, sp_dp, X_eval_j))
    J_sp = np.asarray(_eval_sparse_jacobian(sparse_drift, sp_dp, gp_post_sp, X_eval_j))

    f_efgp_j, _, J_efgp_j = jpd.drift_moments_jax(
        mu_r, grid, X_eval_j, D_lat=D, D_out=D, nufft_eps=qf_nufft_eps)
    f_efgp = np.asarray(f_efgp_j)
    J_efgp = np.asarray(J_efgp_j)

    rmse_f = float(np.sqrt(np.mean((f_sp - f_efgp) ** 2)))
    scale_f = float(np.sqrt(np.mean(f_sp ** 2)))
    rmse_J = float(np.sqrt(np.mean((J_sp - J_efgp) ** 2)))
    scale_J = float(np.sqrt(np.mean(J_sp ** 2)))

    # vs truth
    f_true_grid = X_eval @ np.diag([-1.0, -1.5])
    rmse_sp_truth = float(np.sqrt(np.mean((f_sp - f_true_grid) ** 2)))
    rmse_efgp_truth = float(np.sqrt(np.mean((f_efgp - f_true_grid) ** 2)))

    return dict(
        label=label, S_marginal=S_marginal, cg_tol=cg_tol,
        qf_nufft_eps=qf_nufft_eps, K_per_dim=K_per_dim,
        M=int(grid.M),
        rmse_f=rmse_f, scale_f=scale_f, rel_f=100 * rmse_f / scale_f,
        rmse_J=rmse_J, scale_J=scale_J, rel_J=100 * rmse_J / scale_J,
        rmse_sp_truth=rmse_sp_truth, rmse_efgp_truth=rmse_efgp_truth,
    )


def main():
    print(f"[T1b-prod] JAX devices: {jax.devices()}", flush=True)
    print(f"[T1b-prod] D={D}, T={T}, sigma={SIGMA}, ls={LS}, var={VAR}",
          flush=True)

    settings = [
        # The committed test config (tight reference)
        dict(label="reference (tight)",
             S_marginal=8, cg_tol=1e-7, qf_nufft_eps=6e-8, K_per_dim=16,
             eps_grid=1e-3),
        # Production / notebook config (looser)
        dict(label="production (notebook)",
             S_marginal=2, cg_tol=1e-5, qf_nufft_eps=6e-8, K_per_dim=None,
             eps_grid=1e-3),
        # Production with the user's tighter NUFFT eps that turned out to not help
        dict(label="production + nufft 1e-7",
             S_marginal=2, cg_tol=1e-5, qf_nufft_eps=1e-7, K_per_dim=None,
             eps_grid=1e-3),
        # Even looser CG tol
        dict(label="production + cg 1e-4",
             S_marginal=2, cg_tol=1e-4, qf_nufft_eps=6e-8, K_per_dim=None,
             eps_grid=1e-3),
    ]
    results = []
    for s in settings:
        print(f"\n[T1b-prod] === {s['label']} ===", flush=True)
        r = run_setting(**s)
        results.append(r)
        print(f"  S_marg={r['S_marginal']}  cg_tol={r['cg_tol']:.0e}  "
              f"nufft_eps={r['qf_nufft_eps']:.0e}  K={r['K_per_dim']}  M={r['M']}",
              flush=True)
        print(f"  E[f]  rmse={r['rmse_f']:.4f}  scale={r['scale_f']:.4f}  "
              f"rel={r['rel_f']:.1f}%", flush=True)
        print(f"  E[J]  rmse={r['rmse_J']:.4f}  scale={r['scale_J']:.4f}  "
              f"rel={r['rel_J']:.1f}%", flush=True)
        print(f"  context: SparseGP-vs-truth={r['rmse_sp_truth']:.4f}  "
              f"EFGP-vs-truth={r['rmse_efgp_truth']:.4f}", flush=True)

    print("\n[T1b-prod] Summary:", flush=True)
    print(f"  {'config':30s}  {'M':>4}  {'rel E[f] %':>11}  {'rel E[J] %':>11}",
          flush=True)
    for r in results:
        print(f"  {r['label']:30s}  {r['M']:>4}  {r['rel_f']:>11.1f}  "
              f"{r['rel_J']:>11.1f}", flush=True)

    # Pass criterion
    prod = next(r for r in results if r['label'] == 'production (notebook)')
    print("\n[T1b-prod] Verdict:", flush=True)
    if prod['rel_f'] < 25.0 and prod['rel_J'] < 50.0:
        print(f"[T1b-prod] PASS: production-tol agreement is acceptable "
              f"(E[f] {prod['rel_f']:.1f}% < 25%, E[J] {prod['rel_J']:.1f}% < 50%).",
              flush=True)
    else:
        print(f"[T1b-prod] FAIL: production-tol agreement is loose "
              f"(E[f] {prod['rel_f']:.1f}%, E[J] {prod['rel_J']:.1f}%) — "
              f"consider tighter tolerances.", flush=True)


if __name__ == "__main__":
    main()
