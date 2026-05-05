"""
Diagnostic: shared-q(x) hyper-update dynamics on the linear benchmark.

Goal:
  Separate optimizer dynamics from E-step / grid approximation issues by
  forcing both methods to optimize from the SAME fixed smoother q(x).

Protocol:
  1. Run SparseGP with fixed hypers (no M-step) to obtain a stable shared
     smoother q(x).
  2. From those exact same marginals, run isolated hyper M-steps for
     SparseGP and EFGP at several learning rates.
  3. Compare one-step moves and 50-step trajectories in:
       - raw ell
       - raw sigma^2
       - effective linear slope scale alpha = sigma^2 / ell^2

Interpretation:
  If both methods move alpha in the same direction from the same q(x),
  but SparseGP barely changes ell at the notebook learning rate, then the
  notebook hyper-gap is largely an optimizer-scale artifact rather than a
  broken EFGP hyper objective.

Run with:
  ~/myenv/bin/python -u demos/diag_hyper_dynamics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import optax

from sing.likelihoods import Gaussian
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em, compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params

from sing import efgp_jax_primitives as jp
from sing import efgp_jax_drift as jpd


D = 2
INIT_LS = 1.5
INIT_VAR = 1.0
LR_SWEEP = (1e-3, 3e-3, 1e-2)
QF_NUFFT_EPS = 1e-7


def build_problem():
    a_true = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
    cfg = dict(seed=7, x0=jnp.array([2.0, 0.0]), T=400, t_max=8.0,
               sigma=0.3, N_obs=8)
    xs = simulate_sde(
        jr.PRNGKey(cfg["seed"]),
        x0=cfg["x0"],
        f=lambda x, t: a_true @ x,
        t_max=cfg["t_max"],
        n_timesteps=cfg["T"],
        sigma=lambda x, t: cfg["sigma"] * jnp.eye(D),
    )
    rng = np.random.default_rng(cfg["seed"])
    c_true = rng.standard_normal((cfg["N_obs"], D)) * 0.5
    ys = simulate_gaussian_obs(
        jr.PRNGKey(cfg["seed"] + 1),
        xs,
        dict(C=jnp.asarray(c_true), d=jnp.zeros(cfg["N_obs"]),
             R=jnp.full((cfg["N_obs"],), 0.05)),
    )
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=vt[:D].T, d=ys.mean(0), R=jnp.full((cfg["N_obs"],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1),
    )
    t_grid = jnp.linspace(0.0, cfg["t_max"], cfg["T"])
    lik = Gaussian(ys[None], jnp.ones((1, cfg["T"]), dtype=bool))
    return cfg, xs, ys, lik, op_init, ip_init, t_grid


def default_x_template(ip_init, T):
    diag_std0 = jnp.sqrt(jnp.diag(jnp.asarray(ip_init["V0"][0])))
    half_span = jnp.maximum(jnp.full_like(diag_std0, 3.0), 4.0 * diag_std0)
    return (
        jnp.asarray(ip_init["mu0"][0])[None, :]
        + jnp.linspace(-1.0, 1.0, max(T, 16))[:, None] * half_span[None, :]
    )


def alpha_of(ell, var):
    return float(var) / (float(ell) ** 2)


def build_shared_smoother(cfg, lik, op_init, ip_init, t_grid):
    print("Building shared q(x) from SparseGP fixed-hyper run...", flush=True)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    dp0 = dict(
        length_scales=jnp.full((D,), INIT_LS),
        output_scale=jnp.asarray(np.sqrt(INIT_VAR)),
    )
    mp_sp, nat_p_sp, gp_post_sp, dp_sp_out, ip_sp, op_sp, ie_sp, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse,
        likelihood=lik,
        t_grid=t_grid,
        drift_params=dp0,
        init_params=ip_init,
        output_params=op_init,
        sigma=cfg["sigma"],
        rho_sched=jnp.logspace(-2, -1, 50),
        n_iters=50,
        n_iters_e=10,
        n_iters_m=1,
        perform_m_step=False,
        learn_output_params=False,
        learning_rate=1e-3 * jnp.ones(50),
        print_interval=9999,
    )
    print("Shared q(x) ready.", flush=True)
    return sparse, zs, mp_sp, nat_p_sp, ip_sp, op_sp, ie_sp


def build_efgp_shared_mstep_state(cfg, mp_sp, ip_init, t_grid):
    x_template = default_x_template(ip_init, t_grid.shape[0])
    grid0 = jp.spectral_grid_se(INIT_LS, INIT_VAR, x_template, eps=1e-3)
    k0 = (int(grid0.mtot_per_dim[0]) - 1) // 2
    x_min = np.asarray(x_template.min(axis=0))
    x_max = np.asarray(x_template.max(axis=0))
    x_extent = float((x_max - x_min).max())
    xcen = jnp.asarray(0.5 * (x_min + x_max), dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(INIT_LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(INIT_VAR, dtype=jnp.float32)),
        K_per_dim=k0,
        X_extent=x_extent,
        xcen=xcen,
        d=D,
        eps=1e-3,
    )
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_sp, _, top_sp = jpd.compute_mu_r_jax(
        mp_sp["m"][0], mp_sp["S"][0], mp_sp["SS"][0], del_t, grid, jr.PRNGKey(0),
        sigma_drift_sq=cfg["sigma"] ** 2, S_marginal=2, D_lat=D, D_out=D,
        cg_tol=1e-5, nufft_eps=QF_NUFFT_EPS,
    )
    ws0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws0) < 1e-30, jnp.array(1e-30, dtype=ws0.dtype), ws0)
    a0 = jp.make_A_apply(grid.ws, top_sp, sigmasq=1.0)
    h_r0 = jax.vmap(a0)(mu_r_sp)
    z_r = h_r0 / ws_safe
    print(f"EFGP shared-M-step basis: K0={k0}, M={grid.M}", flush=True)
    return grid, mu_r_sp, top_sp, z_r


def build_sparse_loss(cfg, lik, sparse, zs, mp_sp, nat_p_sp, ip_sp, op_sp, ie_sp, t_grid):
    trial_mask = jnp.ones((1, cfg["T"]), dtype=bool)
    _, ap_sp = vmap(natural_to_marginal_params)(nat_p_sp, trial_mask)
    inputs_dummy = jnp.zeros((1, cfg["T"], 1))
    loss_fn = lambda key, drift_params: -compute_elbo_over_batch(
        key, lik.ys_obs, lik.t_mask, trial_mask,
        sparse, lik, t_grid, drift_params, ip_sp, op_sp,
        nat_p_sp, mp_sp, ap_sp, inputs_dummy, ie_sp, cfg["sigma"],
    )[0]
    return jax.jit(jax.value_and_grad(loss_fn, argnums=1))


def run_sparse_shared(loss_grad_sp, zs, lr, n_steps):
    params = dict(
        length_scales=jnp.full((D,), INIT_LS),
        output_scale=jnp.asarray(np.sqrt(INIT_VAR)),
        inducing_points=zs,
    )
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    path = []
    for i in range(n_steps):
        loss, grads = loss_grad_sp(jr.PRNGKey(i), params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ell = float(jnp.mean(params["length_scales"]))
        var = float(params["output_scale"]) ** 2
        path.append((ell, var, alpha_of(ell, var), float(loss)))
        if not np.isfinite(ell) or not np.isfinite(var):
            break
    return path


def run_efgp_shared(mu_r_sp, z_r, top_sp, grid, lr, n_steps):
    log_ls, log_var, losses = jpd.m_step_kernel_jax(
        np.log(INIT_LS), np.log(INIT_VAR),
        mu_r_fixed=mu_r_sp, z_r=z_r, top=top_sp,
        xis_flat=grid.xis_flat, h_per_dim=grid.h_per_dim,
        D_lat=D, D_out=D,
        n_inner=n_steps, lr=lr,
        n_hutchinson=16, include_trace=True,
        key=jr.PRNGKey(42),
    )
    ell = float(jnp.exp(log_ls))
    var = float(jnp.exp(log_var))
    return ell, var, alpha_of(ell, var), losses


def main():
    print("JAX devices:", jax.devices(), flush=True)
    cfg, xs, ys, lik, op_init, ip_init, t_grid = build_problem()
    sparse, zs, mp_sp, nat_p_sp, ip_sp, op_sp, ie_sp = build_shared_smoother(
        cfg, lik, op_init, ip_init, t_grid)
    grid, mu_r_sp, top_sp, z_r = build_efgp_shared_mstep_state(
        cfg, mp_sp, ip_init, t_grid)
    loss_grad_sp = build_sparse_loss(
        cfg, lik, sparse, zs, mp_sp, nat_p_sp, ip_sp, op_sp, ie_sp, t_grid)

    print("\nOne-step isolated M-step deltas from shared q(x):", flush=True)
    for lr in LR_SWEEP:
        sp_path = run_sparse_shared(loss_grad_sp, zs, lr, n_steps=1)
        sp_ell, sp_var, sp_alpha, _ = sp_path[-1]
        ef_ell, ef_var, ef_alpha, _ = run_efgp_shared(
            mu_r_sp, z_r, top_sp, grid, lr, n_steps=1)
        print(
            f"  lr={lr:g}"
            f"  Sparse: ell {sp_ell:.4f}, var {sp_var:.4f}, alpha {sp_alpha:.4f}"
            f" | EFGP: ell {ef_ell:.4f}, var {ef_var:.4f}, alpha {ef_alpha:.4f}",
            flush=True,
        )

    print("\n50-step isolated M-step sweeps from shared q(x):", flush=True)
    for lr in LR_SWEEP:
        sp_path = run_sparse_shared(loss_grad_sp, zs, lr, n_steps=50)
        sp_ell, sp_var, sp_alpha, _ = sp_path[-1]
        sp_mean_abs = (
            float(np.mean(np.abs(np.diff([p[0] for p in sp_path]))))
            if len(sp_path) > 1 else 0.0
        )
        ef_ell, ef_var, ef_alpha, ef_losses = run_efgp_shared(
            mu_r_sp, z_r, top_sp, grid, lr, n_steps=50)
        print(f"  lr={lr:g}", flush=True)
        print(
            f"    Sparse: steps={len(sp_path):2d}"
            f" ell={sp_ell:.4f} var={sp_var:.4f} alpha={sp_alpha:.4f}"
            f" mean|Δell|={sp_mean_abs:.4g}"
            f" stable={np.isfinite(sp_ell) and np.isfinite(sp_var)}",
            flush=True,
        )
        print(
            f"    EFGP:         ell={ef_ell:.4f} var={ef_var:.4f}"
            f" alpha={ef_alpha:.4f}"
            f" loss0={ef_losses[0]:.2f} loss_end={ef_losses[-1]:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
