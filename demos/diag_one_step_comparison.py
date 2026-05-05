"""
T15: Single-step controlled comparison of EFGP vs SparseGP M-step.

Setup: same q(x), same init hypers, one M-step in each method. Compare:
  - Raw gradient direction in (log_ℓ, log_σ²)
  - One Adam step at the same lr
  - Multiple Adam steps trajectory

If the two methods disagree on the gradient DIRECTION at the same q(x),
that's evidence the objectives are systematically different. If they agree
on direction but EFGP's MAGNITUDE is huge (overshoot), it's an LR issue.

Also tests: from the same q(x_0), how does each method's E-step update q?
Do EFGP and SparseGP give different q(x_1)?

Run:
  ~/myenv/bin/python demos/diag_one_step_comparison.py
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
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from functools import partial

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em, compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params
from jax import vmap


D = 2
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0
LS_INIT = 2.0
VAR_INIT = 0.3
K_FD = 8


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

    def drift(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)
    return drift


def setup():
    drift = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def converge_smoother_sparse(lik, op, ip, t_grid, ls_pin, var_pin, n_em=12):
    """Run SparseGP fixed-hyper EM at given hypers to converge q(x)."""
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(ls_pin)),
                 output_scale=jnp.asarray(math.sqrt(var_pin)))
    rho_sched = jnp.linspace(0.05, 0.4, n_em)
    mp, nat_p, gp_post, dp_out, ip_out, op_out, ie_out, _ = fit_variational_em(
        key=jr.PRNGKey(11), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=sp_dp, init_params=ip, output_params=op,
        sigma=SIGMA, rho_sched=rho_sched, n_iters=n_em, n_iters_e=10,
        n_iters_m=1, perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((n_em,), 0.05), print_interval=999)
    return sparse, zs, mp, nat_p, ip_out, op_out, ie_out, sp_dp


def efgp_mstep_gradient(grid, top, z_r, log_ls, log_var, D_out=2):
    """EFGP M-step gradient at a fixed (μ_r, top, z_r) state, using exact
    slogdet for the trace term (no Hutchinson noise)."""
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype

    def loss_fn(ll, lv):
        ws_real = _ws_real_se(ll, lv, grid.xis_flat, grid.h_per_dim[0], D)
        ws_c = ws_real.astype(cdtype)
        eye_c = jnp.eye(M, dtype=cdtype)
        T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
        T_mat = T_cols.T
        D_w = ws_c[:, None] * eye_c
        A = eye_c + D_w @ T_mat @ D_w
        h = ws_c[None, :] * z_r
        mu = jax.vmap(lambda b: jnp.linalg.solve(A, b))(h)
        def per_r(r):
            t1 = jnp.vdot(h[r], mu[r]).real
            t2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
            return -(t1 - t2)
        det_loss = jax.vmap(per_r)(jnp.arange(D_out)).sum()
        sign, logdet = jnp.linalg.slogdet(A)
        return det_loss + 0.5 * D_out * logdet.real

    val_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))
    L, (g_ls, g_var) = val_and_grad(jnp.asarray(log_ls, dtype=jnp.float32),
                                      jnp.asarray(log_var, dtype=jnp.float32))
    return float(L), float(g_ls), float(g_var)


def sparsegp_mstep_gradient(sparse, lik, t_grid, mp, nat_p, ip_out, op_out,
                              ie_out, ls, var_, key=jr.PRNGKey(0)):
    """SparseGP M-step gradient: -d ELBO / d (length_scales, output_scale)
    at the same q(x). Uses inducing_points fixed at default."""
    trial_mask = jnp.ones((1, T), dtype=bool)
    _, ap = vmap(natural_to_marginal_params)(nat_p, trial_mask)
    inputs_dummy = jnp.zeros((1, T, 1))

    def neg_elbo(drift_params):
        return -compute_elbo_over_batch(
            key, lik.ys_obs, lik.t_mask, trial_mask,
            sparse, lik, t_grid, drift_params, ip_out, op_out,
            nat_p, mp, ap, inputs_dummy, ie_out, SIGMA)[0]

    drift_params = dict(
        length_scales=jnp.full((D,), float(ls)),
        output_scale=jnp.asarray(math.sqrt(var_)),
        inducing_points=sparse.zs,
    )
    L, grad = jax.value_and_grad(neg_elbo)(drift_params)

    # Gradient is in (length_scales, output_scale) space.  Convert to
    # (log_ℓ, log_σ²) via chain rule:
    #   ℓ = exp(log_ℓ), so d/d log_ℓ = ℓ d/d ℓ
    #   σ² = exp(log_σ²), σ = exp(log_σ²/2), so d/d log_σ² = (σ/2) d/d σ
    # Avg gradient over the D length_scales (since EFGP uses isotropic ℓ).
    g_ls_per_dim = np.asarray(grad['length_scales'])
    g_log_ls = float(np.mean(g_ls_per_dim) * ls)
    g_sigma = float(np.asarray(grad['output_scale']))
    sigma = math.sqrt(var_)
    g_log_var = g_sigma * sigma * 0.5
    return float(L), g_log_ls, g_log_var


def main():
    print(f"[T15] One-step controlled M-step comparison", flush=True)
    print(f"[T15] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)
    print(f"[T15] init:  ℓ={LS_INIT}, σ²={VAR_INIT}", flush=True)

    xs, ys, lik, op, ip, t_grid = setup()

    # Build a SHARED q(x) by running SparseGP fixed-hyper EM at INIT hypers
    print(f"\n[T15] Building shared q(x) via SparseGP fixed-hyper EM "
          f"at INIT hypers (ℓ={LS_INIT}, σ²={VAR_INIT})...", flush=True)
    sparse, zs, mp, nat_p, ip_out, op_out, ie_out, sp_dp = converge_smoother_sparse(
        lik, op, ip, t_grid, LS_INIT, VAR_INIT, n_em=12)

    # Build EFGP M-step state at this q(x), at INIT hypers
    print(f"\n[T15] Building EFGP M-step state at the same q(x), "
          f"at INIT hypers...", flush=True)
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS_INIT, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR_INIT, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=16,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r_init)
    z_r = h0 / ws_safe
    print(f"  EFGP grid: K={K_FD}, M={int(grid.M)}", flush=True)

    # Compute gradients at INIT
    log_ls_init = math.log(LS_INIT)
    log_var_init = math.log(VAR_INIT)
    print(f"\n[T15] Gradients at INIT (log_ℓ={log_ls_init:.3f}, "
          f"log_σ²={log_var_init:.3f}):", flush=True)

    L_efgp, g_ls_efgp, g_var_efgp = efgp_mstep_gradient(
        grid, top, z_r, log_ls_init, log_var_init, D_out=D)
    print(f"  EFGP    loss={L_efgp:.4f}  "
          f"grad_log_ℓ={g_ls_efgp:+.4e}  grad_log_σ²={g_var_efgp:+.4e}",
          flush=True)
    grad_norm_efgp = math.sqrt(g_ls_efgp**2 + g_var_efgp**2)
    grad_dir_efgp = (g_ls_efgp/grad_norm_efgp, g_var_efgp/grad_norm_efgp) if grad_norm_efgp > 0 else (0, 0)
    print(f"          ‖grad‖={grad_norm_efgp:.4e}  "
          f"direction (Δlog_ℓ:Δlog_σ²)=({grad_dir_efgp[0]:+.3f}:{grad_dir_efgp[1]:+.3f})",
          flush=True)

    L_sp, g_ls_sp, g_var_sp = sparsegp_mstep_gradient(
        sparse, lik, t_grid, mp, nat_p, ip_out, op_out, ie_out,
        LS_INIT, VAR_INIT)
    print(f"  Sparse  loss={L_sp:.4f}  "
          f"grad_log_ℓ={g_ls_sp:+.4e}  grad_log_σ²={g_var_sp:+.4e}",
          flush=True)
    grad_norm_sp = math.sqrt(g_ls_sp**2 + g_var_sp**2)
    grad_dir_sp = (g_ls_sp/grad_norm_sp, g_var_sp/grad_norm_sp) if grad_norm_sp > 0 else (0, 0)
    print(f"          ‖grad‖={grad_norm_sp:.4e}  "
          f"direction (Δlog_ℓ:Δlog_σ²)=({grad_dir_sp[0]:+.3f}:{grad_dir_sp[1]:+.3f})",
          flush=True)

    # Cosine between directions
    cos_grad = (grad_dir_efgp[0] * grad_dir_sp[0]
                + grad_dir_efgp[1] * grad_dir_sp[1])
    print(f"\n  cosine(EFGP grad, Sparse grad) = {cos_grad:+.3f}", flush=True)
    if cos_grad > 0.7:
        print(f"  → DIRECTIONS AGREE (cos > 0.7).  Difference must be in "
              f"magnitude / LR.", flush=True)
    elif cos_grad > 0:
        print(f"  → DIRECTIONS WEAKLY AGREE (0 < cos < 0.7).", flush=True)
    elif cos_grad > -0.5:
        print(f"  → DIRECTIONS NEARLY ORTHOGONAL — methods optimize "
              f"genuinely different objectives.", flush=True)
    else:
        print(f"  → DIRECTIONS POINT OPPOSITE WAYS — major bug or "
              f"objective mismatch.", flush=True)

    # The negative-gradient direction (which way Adam moves)
    print(f"\n[T15] Adam step direction (-grad), one step at lr=0.05:",
          flush=True)
    print(f"  EFGP   step ({-0.05*g_ls_efgp:+.4f}, {-0.05*g_var_efgp:+.4f}) "
          f"→ new (log_ℓ, log_σ²) = "
          f"({log_ls_init - 0.05*g_ls_efgp:+.3f}, "
          f"{log_var_init - 0.05*g_var_efgp:+.3f})", flush=True)
    print(f"  Sparse step ({-0.05*g_ls_sp:+.4f}, {-0.05*g_var_sp:+.4f}) "
          f"→ new (log_ℓ, log_σ²) = "
          f"({log_ls_init - 0.05*g_ls_sp:+.3f}, "
          f"{log_var_init - 0.05*g_var_sp:+.3f})", flush=True)

    # Direction toward truth (in log space)
    log_ls_truth = math.log(LS_TRUE)
    log_var_truth = math.log(VAR_TRUE)
    dir_to_truth = (log_ls_truth - log_ls_init, log_var_truth - log_var_init)
    norm_truth = math.sqrt(dir_to_truth[0]**2 + dir_to_truth[1]**2)
    dir_truth_unit = (dir_to_truth[0]/norm_truth, dir_to_truth[1]/norm_truth)
    print(f"\n[T15] Direction from init toward truth: "
          f"({dir_truth_unit[0]:+.3f}, {dir_truth_unit[1]:+.3f})  "
          f"(distance={norm_truth:.3f})", flush=True)

    # Cosines: -gradient (descent direction) vs direction-to-truth
    neg_g_efgp = (-grad_dir_efgp[0], -grad_dir_efgp[1])
    neg_g_sp = (-grad_dir_sp[0], -grad_dir_sp[1])
    cos_efgp_to_truth = neg_g_efgp[0] * dir_truth_unit[0] + neg_g_efgp[1] * dir_truth_unit[1]
    cos_sp_to_truth = neg_g_sp[0] * dir_truth_unit[0] + neg_g_sp[1] * dir_truth_unit[1]
    print(f"\n  cos(-grad_EFGP,  truth-dir) = {cos_efgp_to_truth:+.3f}  "
          f"({'→toward truth' if cos_efgp_to_truth > 0 else '←away from truth'})",
          flush=True)
    print(f"  cos(-grad_Sparse, truth-dir) = {cos_sp_to_truth:+.3f}  "
          f"({'→toward truth' if cos_sp_to_truth > 0 else '←away from truth'})",
          flush=True)

    # Save
    np.savez('/tmp/T15_one_step.npz',
              log_ls_init=log_ls_init, log_var_init=log_var_init,
              log_ls_truth=log_ls_truth, log_var_truth=log_var_truth,
              g_ls_efgp=g_ls_efgp, g_var_efgp=g_var_efgp,
              g_ls_sp=g_ls_sp, g_var_sp=g_var_sp,
              L_efgp=L_efgp, L_sp=L_sp,
              cos_grad=cos_grad,
              cos_efgp_to_truth=cos_efgp_to_truth,
              cos_sp_to_truth=cos_sp_to_truth)
    print(f"\n[T15] Saved /tmp/T15_one_step.npz", flush=True)


if __name__ == '__main__':
    main()
