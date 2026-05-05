"""
T1a-extended: Repeat the fixed-h K monotonicity test, but with PRODUCTION
tolerances (cg_tol=1e-5, S_marginal=2, qf_nufft_eps=6e-8) instead of
the tight reference (cg_tol=1e-9, S_marginal=8) used in
``diag_K_monotone_fixed_h.py``.

T1a passed at tight tols — but the production path runs much looser.
If production tol noise is large enough to obscure the K-monotone trend
(e.g. forces 1%+ relative error even at K=32), then the production
path's K choice matters less than the noise floor and the M-step is
fitting noise.

Run:
  ~/myenv/bin/python demos/diag_K_monotone_production_tols.py
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

K_LIST = [4, 6, 8, 12, 16, 24, 32]


def setup_problem():
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 3.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
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


def make_baseline_grid(K_max):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    return jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
        K_per_dim=K_max, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )


def truncated_grid(baseline, K):
    h_val = float(baseline.h_per_dim[0])
    h = jnp.asarray(h_val, dtype=baseline.h_per_dim.dtype)
    k_lat = jnp.arange(-K, K + 1, dtype=baseline.h_per_dim.dtype)
    xis_1d = h * k_lat
    grids = jnp.meshgrid(*[xis_1d for _ in range(D)], indexing="ij")
    xis_flat = jnp.stack([g.ravel() for g in grids], axis=-1)
    xi_norm_sq = (xis_flat * xis_flat).sum(axis=1)
    log_ls = jnp.log(jnp.asarray(LS, dtype=jnp.float32))
    log_var = jnp.log(jnp.asarray(VAR, dtype=jnp.float32))
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


def run_at_tol(label, *, S_marginal, cg_tol, nufft_eps,
               mp, t_grid, X_eval, baseline):
    print(f"\n[T1a-prod] === {label}  S_marg={S_marginal}  cg_tol={cg_tol}  "
          f"nufft={nufft_eps} ===", flush=True)
    Ef_ref, EJ_ref = eval_drift(
        baseline, mp, t_grid, X_eval,
        S_marginal=S_marginal, cg_tol=cg_tol, nufft_eps=nufft_eps)
    scale_f = float(np.sqrt(np.mean(Ef_ref ** 2)))
    scale_J = float(np.sqrt(np.mean(EJ_ref ** 2)))
    print(f"  reference (K_max=32): E[f] scale={scale_f:.4f}, "
          f"E[J] scale={scale_J:.4f}", flush=True)
    print(f"  {'K':>4}  {'M':>6}  {'rel_f%':>8}  {'rel_J%':>8}", flush=True)
    rms_f_seq, rms_J_seq = [], []
    for K in K_LIST:
        gK = truncated_grid(baseline, K)
        Ef_K, EJ_K = eval_drift(
            gK, mp, t_grid, X_eval,
            S_marginal=S_marginal, cg_tol=cg_tol, nufft_eps=nufft_eps)
        rms_f = float(np.sqrt(np.mean((Ef_K - Ef_ref) ** 2)))
        rms_J = float(np.sqrt(np.mean((EJ_K - EJ_ref) ** 2)))
        rel_f = 100 * rms_f / scale_f
        rel_J = 100 * rms_J / scale_J
        rms_f_seq.append(rms_f); rms_J_seq.append(rms_J)
        print(f"  {K:>4}  {gK.M:>6}  {rel_f:>8.3f}  {rel_J:>8.3f}", flush=True)

    # Monotonicity check (1% slack so we don't false-fire on float32 noise)
    f_diffs = np.diff(rms_f_seq[:-1])
    J_diffs = np.diff(rms_J_seq[:-1])
    f_violations = int(np.sum(f_diffs > 0.01 * scale_f))
    J_violations = int(np.sum(J_diffs > 0.01 * scale_J))
    monotone = (f_violations == 0 and J_violations == 0)
    print(f"  monotone (1% tol): E[f]={'YES' if f_violations == 0 else 'NO'}  "
          f"E[J]={'YES' if J_violations == 0 else 'NO'}", flush=True)
    return monotone


def main():
    print(f"[T1a-prod] JAX devices: {jax.devices()}", flush=True)
    xs, ys, lik, t_max = setup_problem()
    print("[T1a-prod] Converging shared smoother...", flush=True)
    mp, t_grid = converge_smoother(lik, t_max)

    K_max = max(K_LIST)
    baseline = make_baseline_grid(K_max)

    ms_np = np.asarray(mp['m'][0])
    grid_lo = ms_np.min(0) + 0.2 * (ms_np.max(0) - ms_np.min(0))
    grid_hi = ms_np.max(0) - 0.2 * (ms_np.max(0) - ms_np.min(0))
    n_per = 10
    g0 = np.linspace(grid_lo[0], grid_hi[0], n_per)
    g1 = np.linspace(grid_lo[1], grid_hi[1], n_per)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    X_eval = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)

    settings = [
        ("tight reference",     dict(S_marginal=8, cg_tol=1e-9,  nufft_eps=6e-8)),
        ("production",          dict(S_marginal=2, cg_tol=1e-5,  nufft_eps=6e-8)),
        ("production (loose)",  dict(S_marginal=2, cg_tol=1e-4,  nufft_eps=6e-8)),
    ]
    verdicts = {}
    for label, kw in settings:
        verdicts[label] = run_at_tol(label, mp=mp, t_grid=t_grid,
                                      X_eval=X_eval, baseline=baseline, **kw)

    print("\n[T1a-prod] Final verdict per setting:", flush=True)
    for label, ok in verdicts.items():
        print(f"  {label:25s} {'PASS' if ok else 'NON-MONOTONE'}", flush=True)


if __name__ == "__main__":
    main()
