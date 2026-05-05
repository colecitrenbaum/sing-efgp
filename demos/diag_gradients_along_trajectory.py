"""
T16: Do EFGP and SparseGP gradients agree at MULTIPLE points, not just at INIT?

T15 showed cos=0.95 at the init point. But T11/T14 plots show trajectories
diverge — by iter 10, EFGP and SparseGP are at different (ℓ, σ²) and
their full trajectories don't overlay.

Test: at FIXED q(x) (shared), evaluate both methods' M-step gradients
at MULTIPLE (ℓ, σ²) points sampled along the SparseGP trajectory. If
gradients agree at every point, the trajectory divergence is purely from
different q(x) evolution. If they disagree at some points, the objectives
have systematic differences.

Also: vary kernel_warmup. If EFGP without warmup matches SparseGP, that's
a contributing factor.

Run:
  ~/myenv/bin/python demos/diag_gradients_along_trajectory.py
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
from jax import vmap

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em, compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params


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
    return sparse, zs, mp, nat_p, ip_out, op_out, ie_out


def efgp_grad_at(grid, top, z_r, log_ls, log_var, D_out=2):
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

    L, (g_ls, g_var) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        jnp.asarray(log_ls, dtype=jnp.float32),
        jnp.asarray(log_var, dtype=jnp.float32))
    return float(L), float(g_ls), float(g_var)


def sparse_grad_at(sparse, lik, t_grid, mp, nat_p, ip_out, op_out, ie_out,
                    ls, var_, key=jr.PRNGKey(0)):
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
    g_ls_per_dim = np.asarray(grad['length_scales'])
    g_log_ls = float(np.mean(g_ls_per_dim) * ls)
    g_sigma = float(np.asarray(grad['output_scale']))
    sigma = math.sqrt(var_)
    g_log_var = g_sigma * sigma * 0.5
    return float(L), g_log_ls, g_log_var


def main():
    print(f"[T16] Gradient agreement at multiple points along trajectory",
          flush=True)
    print(f"[T16] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)

    xs, ys, lik, op, ip, t_grid = setup()

    # Build smoother at INIT — same q(x) for all gradient evaluations
    print(f"\n[T16] Building shared smoother at INIT (ℓ={LS_INIT}, σ²={VAR_INIT})...",
          flush=True)
    sparse, zs, mp, nat_p, ip_out, op_out, ie_out = converge_smoother_sparse(
        lik, op, ip, t_grid, LS_INIT, VAR_INIT, n_em=12)

    # Build EFGP M-step state at this q(x)
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
    A_init_op = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init_op)(mu_r_init)
    z_r = h0 / ws_safe

    # Sample points along the line from init toward truth, in log space
    # init: (log_ℓ=0.69, log_σ²=-1.20)
    # truth: (log_ℓ=-0.22, log_σ²=0.0)
    # plus a few off-line points for completeness
    points = [
        ('init',          math.log(2.0),  math.log(0.3)),
        ('25% to truth',  0.69 - 0.25*0.91, -1.20 + 0.25*1.20),
        ('50% to truth',  0.69 - 0.50*0.91, -1.20 + 0.50*1.20),
        ('75% to truth',  0.69 - 0.75*0.91, -1.20 + 0.75*1.20),
        ('truth',         math.log(LS_TRUE), math.log(VAR_TRUE)),
        ('past truth',    math.log(0.4), math.log(2.0)),
        ('SP@iter25',     math.log(1.726), math.log(0.630)),  # from T11 traj
        ('SP@iter50',     math.log(1.601), math.log(0.861)),
        ('SP@iter100',    math.log(1.393), math.log(1.891)),
        ('EFGP@iter25',   math.log(1.991), math.log(0.340)),  # from T14 lr=0.003
        ('EFGP@iter50',   math.log(1.894), math.log(0.417)),
        ('EFGP@iter100',  math.log(1.861), math.log(0.632)),
    ]

    print(f"\n[T16] Comparing gradients at multiple points (shared q(x) "
          f"converged at INIT hypers):\n", flush=True)
    print(f"  {'point':18s}  {'log_ℓ':>7} {'log_σ²':>7}   "
          f"{'EFGP grad (l, σ²)':>26}   {'SP grad (l, σ²)':>26}   "
          f"{'cos':>5}", flush=True)
    rows = []
    for label, ll, lv in points:
        L_e, ge_ls, ge_var = efgp_grad_at(grid, top, z_r, ll, lv, D_out=D)
        L_s, gs_ls, gs_var = sparse_grad_at(
            sparse, lik, t_grid, mp, nat_p, ip_out, op_out, ie_out,
            math.exp(ll), math.exp(lv))
        norm_e = math.sqrt(ge_ls**2 + ge_var**2)
        norm_s = math.sqrt(gs_ls**2 + gs_var**2)
        cos = (ge_ls*gs_ls + ge_var*gs_var) / (norm_e*norm_s + 1e-30)
        print(f"  {label:18s}  {ll:>+7.3f} {lv:>+7.3f}   "
              f"({ge_ls:>+8.3f}, {ge_var:>+8.3f})   "
              f"({gs_ls:>+8.3f}, {gs_var:>+8.3f})   "
              f"{cos:>+.3f}", flush=True)
        rows.append((label, ll, lv, ge_ls, ge_var, gs_ls, gs_var, cos))

    # Save
    np.savez('/tmp/T16_grad_along_traj.npz',
              labels=np.array([r[0] for r in rows]),
              log_ls=np.array([r[1] for r in rows]),
              log_var=np.array([r[2] for r in rows]),
              g_efgp_ls=np.array([r[3] for r in rows]),
              g_efgp_var=np.array([r[4] for r in rows]),
              g_sp_ls=np.array([r[5] for r in rows]),
              g_sp_var=np.array([r[6] for r in rows]),
              cos=np.array([r[7] for r in rows]))
    print(f"\n[T16] Saved /tmp/T16_grad_along_traj.npz", flush=True)


if __name__ == '__main__':
    main()
