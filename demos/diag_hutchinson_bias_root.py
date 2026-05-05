"""
T1c-debug: Diagnose the Hutchinson trace bias on the log_ls direction.

T1c found: at θ=(log_ls=0.3, log_var=-0.2) and K=8 (M=289), Hutchinson
trace estimates for ∂log|A|/∂log_ls are consistently positive (+0.27..+0.69)
while exact (autodiff slogdet) gives -0.42 — a sign-flipping bias.

Possible causes:
  (a) ``dA_apply`` formula bug (mathematical, not numerical)
  (b) CG not converging well enough at this θ (numerical: residual ≠ 0)
  (c) Float32 cancellation in dws_dlogls × T·D·v sums
  (d) Discrepancy between dws_dlogls analytical formula and autodiff

This script isolates each:
  (1) Build ∂A explicitly via finite-difference of build_A_matrix
       and compare to ``dA_apply``-derived dense matrix.
  (2) Compare exact ``trace(jnp.linalg.solve(A, ∂A))`` against Hutchinson.
  (3) Test with autodiff-derived dws (jax.jacrev) vs analytic dws.
  (4) Sweep cg_tol to see if loose CG explains the bias.

Run:
  ~/myenv/bin/python demos/diag_hutchinson_bias_root.py
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
from sing.efgp_jax_drift import _ws_real_se

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
LS = 1.0
VAR = 1.0
SEED = 7
K_FD = 8
THETA_TEST = (0.3, -0.2)   # the worst case from T1c
N_HUTCHINSON = 1024


def setup():
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 2.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, t_max


def build_state():
    xs, ys, lik, t_max = setup()
    t_grid = jnp.linspace(0., t_max, T)
    del_t = t_grid[1:] - t_grid[:-1]
    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=Vt[:D].T, d=ys.mean(0),
                   R=jnp.full((ys.shape[1],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(LS)),
                 output_scale=jnp.asarray(math.sqrt(VAR)))
    rho_sched = jnp.linspace(0.05, 0.4, 10)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=ip_init, output_params=op_init, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=10, n_iters_e=6, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((10,), 0.05), print_interval=999,
    )
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    _, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M,
    )
    return grid, top


def build_A_matrix(ws_real, top):
    M = int(ws_real.shape[0])
    cdtype = top.v_fft.dtype
    ws_c = ws_real.astype(cdtype)
    eye_c = jnp.eye(M, dtype=cdtype)
    T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
    T_mat = T_cols.T
    D_w_diag = ws_c[:, None] * jnp.eye(M, dtype=cdtype)
    A = eye_c + D_w_diag @ T_mat @ D_w_diag
    return A, T_mat


def main():
    print(f"[debug] JAX devices: {jax.devices()}", flush=True)
    grid, top = build_state()
    M = int(grid.M)
    h_scalar = grid.h_per_dim[0]
    log_ls, log_var = THETA_TEST

    print(f"\n[debug] At θ=(log_ls={log_ls}, log_var={log_var}), K={K_FD}, M={M}",
          flush=True)

    # ---- (1) ws values & dws_dlogls (analytic) ----
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                          jnp.asarray(log_var, dtype=jnp.float32),
                          grid.xis_flat, h_scalar, D)
    A, T_mat = build_A_matrix(ws_real, top)

    ls = math.exp(log_ls)
    ls_sq = ls * ls
    xi_norm_sq = (grid.xis_flat * grid.xis_flat).sum(axis=1)
    dws_dlogls_analytic = ws_real * 0.5 * (D - 4 * math.pi * math.pi * ls_sq * xi_norm_sq)

    # autodiff dws
    def ws_of_logls(ll):
        return _ws_real_se(ll, jnp.asarray(log_var, dtype=jnp.float32),
                           grid.xis_flat, h_scalar, D)
    dws_dlogls_autodiff = jax.jacrev(ws_of_logls)(jnp.asarray(log_ls, dtype=jnp.float32))

    diff = float(jnp.max(jnp.abs(dws_dlogls_analytic - dws_dlogls_autodiff)))
    print(f"\n[debug] (1) dws/d(log_ls): analytic vs autodiff max-diff = {diff:.4e}",
          flush=True)
    print(f"   analytic  range [{float(jnp.min(dws_dlogls_analytic)):+.4f}, "
          f"{float(jnp.max(dws_dlogls_analytic)):+.4f}]", flush=True)
    print(f"   autodiff  range [{float(jnp.min(dws_dlogls_autodiff)):+.4f}, "
          f"{float(jnp.max(dws_dlogls_autodiff)):+.4f}]", flush=True)
    if diff < 1e-5:
        print("   PASS: analytic dws matches autodiff", flush=True)
    else:
        print("   FAIL: analytic dws differs from autodiff", flush=True)

    # ---- (2) Build dA explicitly (dense) and compare to dA_apply ----
    cdtype = top.v_fft.dtype
    ws_c = ws_real.astype(cdtype)
    dws_c = dws_dlogls_analytic.astype(cdtype)

    # Dense ∂A: ∂(D T D)/∂θ = (∂D) T D + D T (∂D)
    def dA_dense():
        D_w = ws_c[:, None] * jnp.eye(M, dtype=cdtype)
        dD_w = dws_c[:, None] * jnp.eye(M, dtype=cdtype)
        return dD_w @ T_mat @ D_w + D_w @ T_mat @ dD_w
    dA_mat = dA_dense()

    # Apply dA via dA_apply on basis vectors → reconstruct dA
    def dA_apply_func(v):
        Dv = ws_c * v
        TDv = jp.toeplitz_apply(top, Dv)
        first = dws_c * TDv
        second = ws_c * jp.toeplitz_apply(top, dws_c * v)
        return first + second
    eye_c = jnp.eye(M, dtype=cdtype)
    dA_via_apply_cols = jax.vmap(dA_apply_func)(eye_c)
    dA_via_apply = dA_via_apply_cols.T   # column = dA @ e_i

    diff_dA = float(jnp.max(jnp.abs(dA_mat - dA_via_apply)))
    rel = float(diff_dA / (jnp.abs(dA_mat).max() + 1e-30))
    print(f"\n[debug] (2) ∂A: dense vs via-apply max-diff = {diff_dA:.4e}  "
          f"(rel {rel*100:.4f}%)", flush=True)

    # ---- (3) Exact trace via jnp.linalg.solve(A, dA) ----
    A_inv_dA = jnp.linalg.solve(A, dA_mat)
    tr_exact = float(jnp.trace(A_inv_dA).real)
    # also via slogdet autodiff (the "ground truth" reference from T1c)
    def logdet_term(ll):
        ws_real_ = _ws_real_se(ll, jnp.asarray(log_var, dtype=jnp.float32),
                               grid.xis_flat, h_scalar, D)
        A_, _ = build_A_matrix(ws_real_, top)
        sign, logdet_ = jnp.linalg.slogdet(A_)
        return logdet_.real
    g_via_grad = float(jax.grad(logdet_term)(jnp.asarray(log_ls, dtype=jnp.float32)))
    print(f"\n[debug] (3) Exact trace[A^-1 ∂A]:", flush=True)
    print(f"   via dense solve(A, dA) and trace = {tr_exact:.4f}", flush=True)
    print(f"   via jax.grad(slogdet(A))         = {g_via_grad:.4f}", flush=True)
    if abs(tr_exact - g_via_grad) < 0.05 * abs(g_via_grad):
        print("   PASS: dense-trace matches autodiff-slogdet", flush=True)
    else:
        print("   WARN: dense-trace ≠ autodiff-slogdet", flush=True)

    # ---- (4) Hutchinson on the DENSE matrices (no toeplitz_apply, no CG) ----
    def hutch_dense(n_probes, key):
        keys = jr.split(key, n_probes)

        def one_probe(key_p):
            v_real = (jax.random.bernoulli(key_p, shape=(M,))
                      .astype(jnp.float32) * 2.0 - 1.0)
            v = v_real.astype(cdtype)
            b = dA_mat @ v   # (∂A) v
            u = jnp.linalg.solve(A, b)   # A^-1 (∂A) v
            return jnp.vdot(v, u).real

        return float(jnp.mean(jax.vmap(one_probe)(keys)))

    tr_hutch_dense = hutch_dense(N_HUTCHINSON, jr.PRNGKey(0))
    print(f"\n[debug] (4) Hutchinson on DENSE matrices (n={N_HUTCHINSON}):", flush=True)
    print(f"   {tr_hutch_dense:.4f}  (vs exact {tr_exact:.4f})", flush=True)
    if abs(tr_hutch_dense - tr_exact) < 0.20 * abs(tr_exact):
        print("   PASS: dense Hutchinson within 20% of exact", flush=True)
    else:
        print("   WARN: dense Hutchinson differs from exact", flush=True)

    # ---- (5) Hutchinson with via-apply ∂A but DENSE A^-1 (i.e., no CG) ----
    def hutch_apply_then_dense_solve(n_probes, key):
        keys = jr.split(key, n_probes)
        def one_probe(key_p):
            v_real = (jax.random.bernoulli(key_p, shape=(M,))
                      .astype(jnp.float32) * 2.0 - 1.0)
            v = v_real.astype(cdtype)
            b = dA_apply_func(v)   # via toeplitz_apply
            u = jnp.linalg.solve(A, b)   # exact solve
            return jnp.vdot(v, u).real
        return float(jnp.mean(jax.vmap(one_probe)(keys)))

    tr_hutch_apply = hutch_apply_then_dense_solve(N_HUTCHINSON, jr.PRNGKey(0))
    print(f"\n[debug] (5) Hutchinson with apply-∂A + dense solve (n={N_HUTCHINSON}):",
          flush=True)
    print(f"   {tr_hutch_apply:.4f}  (vs exact {tr_exact:.4f})", flush=True)
    if abs(tr_hutch_apply - tr_exact) < 0.20 * abs(tr_exact):
        print("   PASS: apply-∂A + dense-solve matches exact", flush=True)
    else:
        print("   WARN: apply-∂A path differs from exact", flush=True)

    # ---- (6) Hutchinson with apply-∂A + CG, sweep cg_tol ----
    print(f"\n[debug] (6) Hutchinson with apply-∂A + CG  (sweep cg_tol):",
          flush=True)
    one_vec = jnp.ones(M, dtype=cdtype)
    T_diag_proxy = jp.toeplitz_apply(top, one_vec).real.mean().astype(jnp.float32)
    ws_sq_real = (ws_c * jnp.conj(ws_c)).real.astype(jnp.float32)
    diag_A_inner = 1.0 + ws_sq_real * T_diag_proxy
    M_inv_inner = (1.0 / diag_A_inner).astype(cdtype)
    M_inv_apply = lambda v: M_inv_inner * v

    def A_apply_d(v):
        tv = jp.toeplitz_apply(top, ws_c * v)
        return ws_c * tv + v

    for cg_tol in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
        keys = jr.split(jr.PRNGKey(0), N_HUTCHINSON)
        def one_probe(key_p):
            v_real = (jax.random.bernoulli(key_p, shape=(M,))
                      .astype(jnp.float32) * 2.0 - 1.0)
            v = v_real.astype(cdtype)
            b = dA_apply_func(v)
            u = jp.cg_solve(A_apply_d, b, tol=cg_tol, max_iter=8 * M,
                             M_inv_apply=M_inv_apply)
            return jnp.vdot(v, u).real
        tr_h = float(jnp.mean(jax.vmap(one_probe)(keys)))
        err = (tr_h - tr_exact) / abs(tr_exact)
        print(f"   cg_tol={cg_tol:.0e}: tr_hutch={tr_h:.4f}  "
              f"err={err*100:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
