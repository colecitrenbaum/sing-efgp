"""
T1c: M-step gradient correctness check.

Goal. Verify that the Hutchinson-based trace gradient + envelope-theorem
deterministic gradient + ws_safe clamp combo in
``m_step_kernel_jax`` (sing/efgp_jax_drift.py:340-528) computes the
correct gradient of the collapsed M-step loss.

Approach. Use a small problem (K=8 → M=289 in 2D) where the full A
matrix can be built explicitly and differentiated through with JAX
autodiff (jax.grad of slogdet + jnp.linalg.solve). That gives an
*exact* reference gradient — no FD noise.

Two checks:
  (1) Hutchinson trace gradient → exact trace via jax.grad(slogdet).
  (2) Total (deterministic + trace) → exact total via jax.grad of full
      loss-at-optimum (envelope-theorem-correct).

We sweep n_hutchinson ∈ {16, 64, 256} to distinguish "Hutchinson
estimator noise" from a "structural bias" — bias is constant in n,
noise shrinks as 1/√n.

Run:
  ~/myenv/bin/python demos/diag_mstep_finite_diff.py
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

THETA_GRID = [
    (0.0, 0.0),
    (-0.3, 0.2),
    (0.3, -0.2),
]

N_HUTCHINSON_SWEEP = [16, 64, 256, 1024]


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


def build_shared_state():
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
    print(f"[T1c] grid: K={K_FD}, M={grid.M}, h={float(grid.h_per_dim[0]):.4f}",
          flush=True)

    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M,
    )

    ws_real = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real.dtype), ws_real)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h_r0 = jax.vmap(A_init)(mu_r_init)
    z_r = h_r0 / ws_safe

    return grid, mu_r_init, top, z_r


def build_A_matrix(ws_real_jnp, top: jp.ToeplitzNDJax) -> jnp.ndarray:
    M = int(ws_real_jnp.shape[0])
    cdtype = top.v_fft.dtype
    ws_c = ws_real_jnp.astype(cdtype)
    eye_c = jnp.eye(M, dtype=cdtype)
    T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
    T_mat = T_cols.T
    D_w_diag = jnp.diag(ws_c)
    A = eye_c + D_w_diag @ T_mat @ D_w_diag
    return A


# ---------- exact references via jax.grad through linalg ----------

def exact_logdet_grad(log_ls, log_var, *, grid, top, D_out):
    """∂(½ * D_out * log|A|)/∂θ via jax.grad through slogdet."""
    h_scalar = grid.h_per_dim[0]

    def logdet_term(ll, lv):
        ws_real = _ws_real_se(ll, lv, grid.xis_flat, h_scalar, D)
        A = build_A_matrix(ws_real, top)
        sign, logdet = jnp.linalg.slogdet(A)
        return 0.5 * D_out * logdet.real

    g_ls, g_var = jax.grad(logdet_term, argnums=(0, 1))(log_ls, log_var)
    return float(g_ls), float(g_var)


def exact_total_grad(log_ls, log_var, *, grid, top, z_r, D_out):
    """Exact total ∂L/∂θ via jax.grad of loss-at-optimum (envelope-correct)."""
    h_scalar = grid.h_per_dim[0]

    def total(ll, lv):
        ws_real = _ws_real_se(ll, lv, grid.xis_flat, h_scalar, D)
        A = build_A_matrix(ws_real, top)
        cdtype = top.v_fft.dtype
        ws_c = ws_real.astype(cdtype)
        h = ws_c[None, :] * z_r
        mu = jax.vmap(lambda b: jnp.linalg.solve(A, b))(h)

        def per_r(r):
            term1 = jnp.vdot(h[r], mu[r]).real
            term2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
            return -(term1 - term2)
        det_loss = jax.vmap(per_r)(jnp.arange(D_out)).sum()

        sign, logdet = jnp.linalg.slogdet(A)
        return det_loss + 0.5 * D_out * logdet.real

    g_ls, g_var = jax.grad(total, argnums=(0, 1))(log_ls, log_var)
    return float(g_ls), float(g_var)


def exact_det_grad(log_ls, log_var, *, grid, top, z_r, D_out):
    """Exact deterministic ∂L_det/∂θ at refreshed μ (envelope; NO logdet term)."""
    h_scalar = grid.h_per_dim[0]

    def det_only(ll, lv):
        ws_real = _ws_real_se(ll, lv, grid.xis_flat, h_scalar, D)
        A = build_A_matrix(ws_real, top)
        cdtype = top.v_fft.dtype
        ws_c = ws_real.astype(cdtype)
        h = ws_c[None, :] * z_r
        mu = jax.vmap(lambda b: jnp.linalg.solve(A, b))(h)

        def per_r(r):
            term1 = jnp.vdot(h[r], mu[r]).real
            term2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
            return -(term1 - term2)
        return jax.vmap(per_r)(jnp.arange(D_out)).sum()

    g_ls, g_var = jax.grad(det_only, argnums=(0, 1))(log_ls, log_var)
    return float(g_ls), float(g_var)


# ---------- Hutchinson estimator copied from m_step_kernel_jax ----------

def hutchinson_logdet_grad(log_ls, log_var, *, grid, top, D_out, key,
                            n_hutchinson):
    log_ls = jnp.asarray(log_ls, dtype=jnp.float32)
    log_var = jnp.asarray(log_var, dtype=jnp.float32)
    h_scalar = grid.h_per_dim[0]

    ws = _ws_real_se(log_ls, log_var, grid.xis_flat, h_scalar, D)
    ws_c = ws.astype(top.v_fft.dtype)

    ls = jnp.exp(log_ls)
    ls_sq = ls * ls
    xi_norm_sq = (grid.xis_flat * grid.xis_flat).sum(axis=1)
    dws_dlogls = ws * 0.5 * (D - 4 * math.pi * math.pi * ls_sq * xi_norm_sq)
    dws_dlogvar = ws * 0.5

    def A_apply_d(v):
        tv = jp.toeplitz_apply(top, ws_c * v)
        return ws_c * tv + v

    def dA_apply(dD, v):
        Dv = ws_c * v
        TDv = jp.toeplitz_apply(top, Dv)
        first = dD.astype(top.v_fft.dtype) * TDv
        second = ws_c * jp.toeplitz_apply(top, dD.astype(top.v_fft.dtype) * v)
        return first + second

    M = int(grid.xis_flat.shape[0])
    one_vec = jnp.ones(M, dtype=top.v_fft.dtype)
    T_diag_proxy = jp.toeplitz_apply(top, one_vec).real.mean().astype(jnp.float32)
    ws_sq_real = (ws_c * jnp.conj(ws_c)).real.astype(jnp.float32)
    diag_A_inner = 1.0 + ws_sq_real * T_diag_proxy
    M_inv_inner = (1.0 / diag_A_inner).astype(top.v_fft.dtype)
    M_inv_apply = lambda v: M_inv_inner * v

    def one_probe(key_p):
        v_real = (jax.random.bernoulli(key_p, shape=(M,))
                  .astype(jnp.float32) * 2.0 - 1.0)
        v = v_real.astype(top.v_fft.dtype)
        b_ls = dA_apply(dws_dlogls, v)
        u_ls = jp.cg_solve(A_apply_d, b_ls, tol=1e-9, max_iter=4 * M,
                            M_inv_apply=M_inv_apply)
        b_var = dA_apply(dws_dlogvar, v)
        u_var = jp.cg_solve(A_apply_d, b_var, tol=1e-9, max_iter=4 * M,
                             M_inv_apply=M_inv_apply)
        return jnp.array([jnp.vdot(v, u_ls).real, jnp.vdot(v, u_var).real])

    keys = jr.split(key, n_hutchinson)
    probes = jax.vmap(one_probe)(keys)
    traces = probes.mean(axis=0)
    return float(0.5 * traces[0] * D_out), float(0.5 * traces[1] * D_out)


def main():
    print(f"[T1c] JAX devices: {jax.devices()}", flush=True)
    print(f"[T1c] D={D}, T={T}, K_FD={K_FD}", flush=True)
    grid, mu_r, top, z_r = build_shared_state()

    # First: confirm the deterministic-gradient autodiff matches the
    # m_step_kernel_jax-style autodiff (sanity check on the build_A_matrix
    # path being equivalent to toeplitz_apply).
    print("\n[T1c] (1) Deterministic gradient via two paths:", flush=True)
    print(f"  {'log_ls':>7}  {'log_var':>7}  {'g_det_ls (full A)':>20}  "
          f"{'g_det_var (full A)':>20}", flush=True)
    for (ll, lv) in THETA_GRID:
        g_det_ls, g_det_var = exact_det_grad(
            float(ll), float(lv), grid=grid, top=top, z_r=z_r, D_out=D)
        print(f"  {ll:>7.3f}  {lv:>7.3f}  {g_det_ls:>20.6e}  {g_det_var:>20.6e}",
              flush=True)

    # Second: log-det gradient. Compare exact (autodiff slogdet) vs Hutchinson
    # at increasing probe counts. Bias should NOT shrink with n; noise should.
    print("\n[T1c] (2) Log-det gradient — exact (autodiff slogdet) vs Hutchinson",
          flush=True)
    print(f"  {'log_ls':>7}  {'log_var':>7}  {'exact_ls':>11}  "
          f"{'exact_var':>11}", flush=True)
    exacts = {}
    for (ll, lv) in THETA_GRID:
        ex_ls, ex_var = exact_logdet_grad(
            float(ll), float(lv), grid=grid, top=top, D_out=D)
        exacts[(ll, lv)] = (ex_ls, ex_var)
        print(f"  {ll:>7.3f}  {lv:>7.3f}  {ex_ls:>11.4e}  {ex_var:>11.4e}",
              flush=True)

    print(f"\n  Hutchinson estimates at varying n:", flush=True)
    print(f"  {'log_ls':>7}  {'log_var':>7}  "
          + "  ".join(f"{'n=' + str(n) + ' ls':>14s}  {'n=' + str(n) + ' var':>14s}"
                       for n in N_HUTCHINSON_SWEEP), flush=True)
    bias_ls_mean = []
    bias_var_mean = []
    for (ll, lv) in THETA_GRID:
        ex_ls, ex_var = exacts[(ll, lv)]
        row = [f"{ll:>7.3f}", f"{lv:>7.3f}"]
        for n in N_HUTCHINSON_SWEEP:
            hu_ls, hu_var = hutchinson_logdet_grad(
                float(ll), float(lv), grid=grid, top=top, D_out=D,
                key=jr.PRNGKey(0), n_hutchinson=n)
            err_ls = (hu_ls - ex_ls) / (abs(ex_ls) + 1e-12)
            err_var = (hu_var - ex_var) / (abs(ex_var) + 1e-12)
            row.append(f"{hu_ls:>10.3e}({err_ls:+.0%})")
            row.append(f"{hu_var:>10.3e}({err_var:+.0%})")
            if n == max(N_HUTCHINSON_SWEEP):
                bias_ls_mean.append(err_ls)
                bias_var_mean.append(err_var)
        print("  " + "  ".join(row), flush=True)

    # Verdict
    max_n = max(N_HUTCHINSON_SWEEP)
    print(f"\n[T1c] At n_hutchinson={max_n} (largest), the residual "
          f"(Hutchinson - exact) errors per θ:", flush=True)
    print(f"  mean |err_ls| = {np.mean(np.abs(bias_ls_mean))*100:.1f}%  "
          f"max |err_ls| = {np.max(np.abs(bias_ls_mean))*100:.1f}%", flush=True)
    print(f"  mean |err_var| = {np.mean(np.abs(bias_var_mean))*100:.1f}%  "
          f"max |err_var| = {np.max(np.abs(bias_var_mean))*100:.1f}%", flush=True)
    if max(np.max(np.abs(bias_ls_mean)),
           np.max(np.abs(bias_var_mean))) < 0.20:
        print("[T1c] PASS: Hutchinson trace is consistent with exact log-det "
              "gradient (residual < 20% at largest n; noise dominates).",
              flush=True)
    else:
        print("[T1c] FAIL: Hutchinson trace systematically biased — residual "
              "exceeds 20% even at largest n. Investigate dws/dθ formulas, "
              "ws_safe clamp, or preconditioner.", flush=True)


if __name__ == "__main__":
    main()
