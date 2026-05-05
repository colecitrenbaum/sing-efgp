"""
Diagnostic: how much error comes from the jit-friendly fixed-K grid policy?

This script compares the production EFGP path

    fixed integer lattice K, adaptive spacing h(theta)

against the more exact diagnostic oracle

    rebuild the tailored spectral grid each outer EM iteration

under the current stable notebook protocol:
  - Gaussian emissions
  - broader default support box
  - eps_grid = qf_nufft_eps = 1e-3
  - qf_cg_tol = 1e-5

The production K is chosen the same way as the library default:
match the initial tailored grid at the requested ``eps``.  The key
question is whether that jit-friendly path is already good enough, or
whether the remaining gap is really a frequency-budget / grid-policy
issue.

Run with:
  ~/myenv/bin/python -u demos/diag_grid_feedback.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Gaussian
from sing.simulate_data import simulate_sde, simulate_gaussian_obs

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd


D = 2
EPS = 1e-3
QF_CG_TOL = 1e-5
QF_NUFFT_EPS = 1e-7


LINEAR = dict(
    name="Damped rotation",
    drift_fn=lambda x, t: jnp.array([[-0.6, 1.0], [-1.0, -0.6]]) @ x,
    x0=jnp.array([2.0, 0.0]),
    T=400,
    t_max=8.0,
    sigma=0.3,
    N_obs=8,
    seed=7,
    n_em=50,
)

ANHARMONIC = dict(
    name="Anharmonic oscillator",
    drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3 * x[1] - 0.5 * x[0] ** 3]),
    x0=jnp.array([1.5, 0.0]),
    T=400,
    t_max=10.0,
    sigma=0.3,
    N_obs=8,
    seed=21,
    n_em=20,
)


def make_problem(cfg):
    xs = simulate_sde(
        jr.PRNGKey(cfg["seed"]),
        x0=cfg["x0"],
        f=cfg["drift_fn"],
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
    op = dict(C=vt[:D].T, d=ys.mean(0), R=jnp.full((cfg["N_obs"],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1),
    )
    t_grid = jnp.linspace(0.0, cfg["t_max"], cfg["T"])
    lik = Gaussian(ys[None], jnp.ones((1, cfg["T"]), dtype=bool))
    return np.asarray(xs), ys, op, ip, t_grid, lik


def default_x_template(ip_init, T):
    diag_std0 = jnp.sqrt(jnp.diag(jnp.asarray(ip_init["V0"][0])))
    half_span = jnp.maximum(jnp.full_like(diag_std0, 3.0), 4.0 * diag_std0)
    return (
        jnp.asarray(ip_init["mu0"][0])[None, :]
        + jnp.linspace(-1.0, 1.0, max(T, 16))[:, None] * half_span[None, :]
    )


def make_truth_grid(xs_np):
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    gx, gy = np.meshgrid(g0, g1, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def procrustes_align(m_inf, m_true):
    xi = np.asarray(m_inf)
    xt = np.asarray(m_true)
    bi = xi.mean(0)
    bt = xt.mean(0)
    a_t, *_ = np.linalg.lstsq(xi - bi, xt - bt, rcond=None)
    a = a_t.T
    b = bt - a @ bi
    return a, b


def eval_efgp_drift(mp, *, t_grid, sigma, ls, var, x_template,
                    grid_pts_inferred, tailored_grid_every_iter, K_per_dim):
    x_min = np.asarray(x_template.min(axis=0))
    x_max = np.asarray(x_template.max(axis=0))
    x_extent = float((x_max - x_min).max())
    xcen = jnp.asarray(0.5 * (x_min + x_max), dtype=jnp.float32)
    if tailored_grid_every_iter:
        grid = jp.spectral_grid_se(ls, var, x_template, eps=EPS)
    else:
        grid = jp.spectral_grid_se_fixed_K(
            log_ls=jnp.log(jnp.asarray(ls, dtype=jnp.float32)),
            log_var=jnp.log(jnp.asarray(var, dtype=jnp.float32)),
            K_per_dim=K_per_dim,
            X_extent=x_extent,
            xcen=xcen,
            d=D,
            eps=EPS,
        )

    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        mp["m"][0], mp["S"][0], mp["SS"][0], del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=2, D_lat=D, D_out=D,
        cg_tol=QF_CG_TOL, nufft_eps=QF_NUFFT_EPS,
    )
    ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts_inferred, dtype=jnp.float32),
        D_lat=D, D_out=D,
    )
    return np.asarray(ef), grid.M


def initial_tailored_k(lengthscale, variance, x_template):
    grid0 = jp.spectral_grid_se(lengthscale, variance, x_template, eps=EPS)
    return (int(grid0.mtot_per_dim[0]) - 1) // 2


def run_variant(cfg, *, tailored_grid_every_iter, K_per_dim):
    xs_np, ys, op_init, ip_init, t_grid, lik = make_problem(cfg)
    x_template = default_x_template(ip_init, cfg["T"])

    fit_cfg = dict(
        likelihood=lik,
        t_grid=t_grid,
        output_params=op_init,
        init_params=ip_init,
        latent_dim=D,
        lengthscale=1.5,
        variance=1.0,
        sigma=cfg["sigma"],
        sigma_drift_sq=cfg["sigma"] ** 2,
        eps_grid=EPS,
        qf_cg_tol=QF_CG_TOL,
        qf_nufft_eps=QF_NUFFT_EPS,
        S_marginal=2,
        n_em_iters=cfg["n_em"],
        n_estep_iters=10,
        rho_sched=jnp.linspace(0.1, 0.3, cfg["n_em"]),
        learn_emissions=True,
        update_R=True,
        emission_warmup_iters=8,
        learn_kernel=True,
        n_mstep_iters=10,
        mstep_lr=0.01,
        n_hutchinson_mstep=16,
        kernel_warmup_iters=8,
        tailored_grid_every_iter=tailored_grid_every_iter,
        K_per_dim=K_per_dim,
        X_template=x_template,
        verbose=False,
    )

    t0 = time.time()
    mp, _, op_out, _, hist = em.fit_efgp_sing_jax(**fit_cfg)
    wall = time.time() - t0

    a, b = procrustes_align(mp["m"][0], xs_np)
    lat_rmse = float(np.sqrt(np.mean(
        (np.asarray(mp["m"][0]) @ a.T + b - xs_np) ** 2
    )))

    grid_true = make_truth_grid(xs_np)
    grid_inf = (grid_true - b) @ np.linalg.inv(a).T
    f_true = np.array([
        np.asarray(cfg["drift_fn"](jnp.asarray(p), 0.0))
        for p in grid_true
    ])
    f_pred, m_eval = eval_efgp_drift(
        mp,
        t_grid=t_grid,
        sigma=cfg["sigma"],
        ls=float(hist.lengthscale[-1]),
        var=float(hist.variance[-1]),
        x_template=x_template,
        grid_pts_inferred=grid_inf,
        tailored_grid_every_iter=tailored_grid_every_iter,
        K_per_dim=K_per_dim,
    )
    f_pred = f_pred @ a.T
    drift_rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))

    return dict(
        wall=wall,
        latent_rmse=lat_rmse,
        drift_rmse=drift_rmse,
        ls=float(hist.lengthscale[-1]),
        var=float(hist.variance[-1]),
        mean_R=float(np.mean(np.asarray(op_out["R"]))),
        grid_eval_M=m_eval,
        ls_path=np.array(hist.lengthscale),
    )


def main():
    print("JAX devices:", jax.devices(), flush=True)
    for cfg in (LINEAR, ANHARMONIC):
        print(f"\n=== {cfg['name']} ===", flush=True)
        _, _, _, ip_init, _, _ = make_problem(cfg)
        prod_k = initial_tailored_k(1.5, 1.0, default_x_template(ip_init, cfg["T"]))
        prod = run_variant(cfg, tailored_grid_every_iter=False, K_per_dim=prod_k)
        oracle = run_variant(cfg, tailored_grid_every_iter=True, K_per_dim=None)
        print(
            f"  jit-friendly adaptive-h, K={prod_k}:"
            f" wall={prod['wall']:.1f}s"
            f"  ell={prod['ls']:.4f}"
            f"  var={prod['var']:.4f}"
            f"  latent={prod['latent_rmse']:.4f}"
            f"  drift={prod['drift_rmse']:.4f}"
            f"  M_eval={prod['grid_eval_M']}",
            flush=True,
        )
        print(
            "  tailored grid every EM iter:"
            f" wall={oracle['wall']:.1f}s"
            f"  ell={oracle['ls']:.4f}"
            f"  var={oracle['var']:.4f}"
            f"  latent={oracle['latent_rmse']:.4f}"
            f"  drift={oracle['drift_rmse']:.4f}"
            f"  M_eval={oracle['grid_eval_M']}",
            flush=True,
        )
        print(
            "  delta(jit - oracle):"
            f" d_ell={prod['ls'] - oracle['ls']:+.4f}"
            f"  d_var={prod['var'] - oracle['var']:+.4f}"
            f"  d_drift={prod['drift_rmse'] - oracle['drift_rmse']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
