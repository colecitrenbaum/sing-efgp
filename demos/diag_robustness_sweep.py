"""
T5: Robustness sweep — does EFGP correctness hold across regimes?

The user's mandate: "different regimes in your tests, so we don't find that
we sort of piece mealed something together that works for one particular
SDE drift, and size T, but isn't robust."

This script runs the two strongest correctness checks (K-monotone at
fixed h, cross-method drift agreement at production tols) across a small
sweep of (drift, T, ℓ_true) regimes. If any (drift, T, ℓ) combination
shows non-monotone K or > 25% cross-method E[f] disagreement at
production tols, that's evidence the prior single-regime PASS results
are not representative.

Run:
  ~/myenv/bin/python demos/diag_robustness_sweep.py
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
SIGMA = 0.3
N_OBS = 6
SEED = 7

# (regime, T, ls_kernel) — ls_kernel is BOTH the SparseGP kernel hyper and EFGP's
REGIMES = [
    ("linear",     200,  1.0),
    ("linear",     500,  1.0),
    ("linear",     200,  0.6),     # smaller ℓ → sharper kernel → tighter
    ("duffing",    200,  1.0),
    ("duffing",    500,  1.0),
    ("anharmonic", 200,  1.0),
    ("anharmonic", 500,  1.0),
]

K_LIST = [4, 6, 8, 12, 16]   # K_max=16 (M=1089) for speed; T1a confirmed K=6+ is 1e-4 already


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


def setup(regime, T):
    drift_true = drift_for(regime)
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 3.0 if regime == "linear" else T * 0.05
    x0 = jnp.array([1.5, 0.0]) if regime == "duffing" else jnp.array([1.0, -0.6])
    xs = simulate_sde(jr.PRNGKey(SEED), x0=x0, f=drift_true,
                      t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, t_max


def converge_smoother(lik, t_max, T, ls):
    yc = lik.ys_obs[0] - lik.ys_obs[0].mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=Vt[:D].T, d=lik.ys_obs[0].mean(0),
                   R=jnp.full((lik.ys_obs.shape[2],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(ls)),
                 output_scale=jnp.asarray(1.0))
    n_em = 12
    rho_sched = jnp.linspace(0.05, 0.4, n_em)
    t_grid = jnp.linspace(0., t_max, T)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=ip_init, output_params=op_init, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=n_em, n_iters_e=6, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((n_em,), 0.05), print_interval=999,
    )
    return sparse_drift, sp_dp, mp, t_grid


def make_baseline_grid(K_max, ls):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    return jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(1.0, dtype=jnp.float32)),
        K_per_dim=K_max, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )


def truncated_grid(baseline, K, ls):
    h_val = float(baseline.h_per_dim[0])
    h = jnp.asarray(h_val, dtype=baseline.h_per_dim.dtype)
    k_lat = jnp.arange(-K, K + 1, dtype=baseline.h_per_dim.dtype)
    xis_1d = h * k_lat
    grids = jnp.meshgrid(*[xis_1d for _ in range(D)], indexing="ij")
    xis_flat = jnp.stack([g.ravel() for g in grids], axis=-1)
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_ls = jnp.log(jnp.asarray(ls, dtype=jnp.float32))
    log_var = jnp.log(jnp.asarray(1.0, dtype=jnp.float32))
    ls_sq = jnp.exp(2 * log_ls)
    log_S = (D / 2.0) * jnp.log(2 * math.pi * ls_sq) + log_var \
            - 2 * (math.pi ** 2) * ls_sq * xi_norm_sq
    ws_real = jnp.sqrt(jnp.exp(log_S) * (h_val ** D))
    ws = ws_real.astype(baseline.ws.dtype)
    mtot = tuple(2 * K + 1 for _ in range(D))
    M = (2 * K + 1) ** D
    h_per_dim = jnp.full((D,), h_val, dtype=baseline.h_per_dim.dtype)
    return jp.JaxGridState(xis_flat=xis_flat, ws=ws, h_per_dim=h_per_dim,
                            mtot_per_dim=mtot, xcen=baseline.xcen,
                            M=M, d=D)


def eval_drift(grid, mp, t_grid, X_eval, *, S_marginal, cg_tol, nufft_eps):
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=S_marginal,
        D_lat=D, D_out=D, cg_tol=cg_tol, max_cg_iter=4 * grid.M,
        nufft_eps=nufft_eps,
    )
    Ef, _, EJ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval), D_lat=D, D_out=D, nufft_eps=nufft_eps)
    return np.asarray(Ef), np.asarray(EJ)


def _eval_sparse_jacobian(sparse_drift, drift_params, gp_post, X_eval):
    eps_S = jnp.eye(D) * 1e-9
    eval_one = lambda x: sparse_drift.dfdx(
        drift_params, jr.PRNGKey(0), 0.0, x, eps_S, gp_post)
    return jax.vmap(eval_one)(X_eval)


def run_one(regime, T, ls):
    print(f"\n[T5] === regime={regime}  T={T}  ls={ls} ===", flush=True)
    xs, ys, lik, t_max = setup(regime, T)
    sparse_drift, sp_dp, mp, t_grid = converge_smoother(lik, t_max, T, ls)
    ms_np = np.asarray(mp['m'][0])
    print(f"  smoother m std: {ms_np.std(axis=0)}", flush=True)

    K_max = max(K_LIST)
    baseline = make_baseline_grid(K_max, ls)

    # Eval grid (drop the boundary)
    grid_lo = ms_np.min(0) + 0.2 * (ms_np.max(0) - ms_np.min(0))
    grid_hi = ms_np.max(0) - 0.2 * (ms_np.max(0) - ms_np.min(0))
    n_per = 10
    g0 = np.linspace(grid_lo[0], grid_hi[0], n_per)
    g1 = np.linspace(grid_lo[1], grid_hi[1], n_per)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)

    # ---- (a) K-monotone test at fixed h, tight tols ----
    Ef_ref, EJ_ref = eval_drift(
        baseline, mp, t_grid, X_eval,
        S_marginal=8, cg_tol=1e-7, nufft_eps=6e-8)
    scale_f = float(np.sqrt(np.mean(Ef_ref ** 2)))
    rms_f_seq = []
    for K in K_LIST:
        gK = truncated_grid(baseline, K, ls)
        Ef_K, _ = eval_drift(
            gK, mp, t_grid, X_eval,
            S_marginal=8, cg_tol=1e-7, nufft_eps=6e-8)
        rms = float(np.sqrt(np.mean((Ef_K - Ef_ref) ** 2)))
        rms_f_seq.append(rms)
    f_diffs = np.diff(rms_f_seq[:-1])
    monotone = bool(np.all(f_diffs <= 0.01 * scale_f))

    # ---- (b) Cross-method agreement at production tols ----
    gp_post_sp = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp, jnp.ones((1, T), dtype=bool),
        sp_dp, jnp.zeros((1, T, 1)), jnp.zeros((D, 1)), SIGMA)
    f_sp = np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post_sp, sp_dp, jnp.asarray(X_eval)))

    # EFGP at production tols, K=auto matches the EM driver default
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid_init = jp.spectral_grid_se(ls, 1.0, X_template, eps=1e-3)
    K_auto = (int(grid_init.mtot_per_dim[0]) - 1) // 2
    grid_prod = truncated_grid(baseline, min(K_auto, K_max), ls)
    Ef_prod, _ = eval_drift(
        grid_prod, mp, t_grid, X_eval,
        S_marginal=2, cg_tol=1e-5, nufft_eps=6e-8)
    rmse_cross = float(np.sqrt(np.mean((f_sp - Ef_prod) ** 2)))
    scale_sp = float(np.sqrt(np.mean(f_sp ** 2)))
    rel_cross = 100 * rmse_cross / max(scale_sp, 1e-6)

    print(f"  K-monotone @ fixed h?  {'YES' if monotone else 'NO'}  "
          f"(seq: {[f'{r:.2e}' for r in rms_f_seq]})", flush=True)
    print(f"  cross-method E[f] rel = {rel_cross:.2f}%  "
          f"(K_auto={K_auto}, K_used={min(K_auto, K_max)})", flush=True)
    return dict(regime=regime, T=T, ls=ls, monotone=monotone,
                rel_cross=rel_cross, K_auto=K_auto)


def main():
    print(f"[T5] JAX devices: {jax.devices()}", flush=True)
    print(f"[T5] Sweeping {len(REGIMES)} regimes, K_LIST={K_LIST}", flush=True)
    rows = [run_one(*r) for r in REGIMES]

    print("\n[T5] Final table:", flush=True)
    print(f"  {'regime':12s}  {'T':>4s}  {'ls':>4s}  {'K_auto':>6s}  "
          f"{'monotone':>9s}  {'cross E[f] %':>13s}", flush=True)
    n_fail_mono = 0; n_fail_cross = 0
    for r in rows:
        ok_mono = '✓' if r['monotone'] else '✗'
        ok_cross = '✓' if r['rel_cross'] < 25.0 else '✗'
        if not r['monotone']:
            n_fail_mono += 1
        if r['rel_cross'] >= 25.0:
            n_fail_cross += 1
        print(f"  {r['regime']:12s}  {r['T']:>4d}  {r['ls']:>4.1f}  "
              f"{r['K_auto']:>6d}  {ok_mono} {'YES' if r['monotone'] else 'NO':>5s}  "
              f"{r['rel_cross']:>10.2f}% {ok_cross}", flush=True)

    print(f"\n[T5] {len(REGIMES)-n_fail_mono}/{len(REGIMES)} regimes pass "
          f"K-monotonicity at fixed h", flush=True)
    print(f"[T5] {len(REGIMES)-n_fail_cross}/{len(REGIMES)} regimes pass "
          f"cross-method E[f] < 25%", flush=True)
    if n_fail_mono == 0 and n_fail_cross == 0:
        print("[T5] OVERALL: PASS — EFGP correctness is robust across regimes.",
              flush=True)
    else:
        print("[T5] OVERALL: FAIL — investigate failing regime(s).", flush=True)


if __name__ == "__main__":
    main()
