"""
Diagnose the failure modes in ``demos/efgp_vs_sparsegp_benchmarks.ipynb``.

This script focuses on the concrete questions that came up during the EFGP
port:

1. Is the notebook's linear-row EFGP/SparseGP hyper gap a q(f) bug?
2. Is the linear-row endpoint at ``n_em=25`` actually converged?
3. Are the notebook's nonlinear SparseGP rows numerically trustworthy?

The output is designed to be read top-to-bottom.

Run with:
  ~/myenv/bin/python demos/diag_notebook_benchmark.py
"""
from __future__ import annotations

import math
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

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        r = output_params["R"]
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi * r)
            - 0.5 * ((y - mean) ** 2 + var) / r
        )


A_ROT = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
LINEAR = dict(
    name="Damped rotation",
    drift_fn=lambda x, t: A_ROT @ x,
    x0=jnp.array([2.0, 0.0]),
    T=400,
    t_max=8.0,
    sigma=0.3,
    N_obs=8,
    seed=7,
)
DUFFING = dict(
    name="Duffing",
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0] ** 3 - 0.5 * x[1]]),
    x0=jnp.array([1.2, 0.0]),
    T=400,
    t_max=15.0,
    sigma=0.2,
    N_obs=8,
    seed=13,
)
ANHARMONIC = dict(
    name="Anharmonic",
    drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3 * x[1] - 0.5 * x[0] ** 3]),
    x0=jnp.array([1.5, 0.0]),
    T=400,
    t_max=10.0,
    sigma=0.3,
    N_obs=8,
    seed=21,
)

D = 2


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
    out_true = dict(
        C=jnp.asarray(c_true),
        d=jnp.zeros(cfg["N_obs"]),
        R=jnp.full((cfg["N_obs"],), 0.05),
    )
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg["seed"] + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    output_params = dict(
        C=vt[:D].T,
        d=ys.mean(0),
        R=jnp.full((cfg["N_obs"],), 0.1),
    )
    init_params = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1),
    )
    t_grid = jnp.linspace(0.0, cfg["t_max"], cfg["T"])
    likelihood = GLik(ys[None], jnp.ones((1, cfg["T"]), dtype=bool))
    return xs, ys, likelihood, output_params, init_params, t_grid


def procrustes_align(m_inf, m_true):
    xi = np.asarray(m_inf)
    xt = np.asarray(m_true)
    bi = xi.mean(0)
    bt = xt.mean(0)
    a_t, *_ = np.linalg.lstsq(xi - bi, xt - bt, rcond=None)
    a = a_t.T
    b = bt - a @ bi
    return a, b


def latent_rmse(m_inf, m_true):
    a, b = procrustes_align(m_inf, m_true)
    xi = np.asarray(m_inf)
    xt = np.asarray(m_true)
    return float(np.sqrt(np.mean((xi @ a.T + b - xt) ** 2)))


def make_truth_grid(xs_np):
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    gx, gy = np.meshgrid(g0, g1, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def linear_truth_field(grid_true):
    return np.array([np.asarray(A_ROT @ jnp.asarray(p)) for p in grid_true])


def make_eval_grid(m_inf, xs_true, grid_true):
    a, b = procrustes_align(m_inf, xs_true)
    grid_inf = (grid_true - b) @ np.linalg.inv(a).T
    return a, b, grid_inf


def fit_efgp(cfg, likelihood, output_params, init_params, t_grid, *, n_em):
    rho = jnp.linspace(0.1, 0.3, n_em)
    t0 = time.time()
    mp, _, op_out, _, hist = em.fit_efgp_sing_jax(
        likelihood=likelihood,
        t_grid=t_grid,
        output_params=output_params,
        init_params=init_params,
        latent_dim=D,
        lengthscale=1.5,
        variance=1.0,
        sigma=cfg["sigma"],
        sigma_drift_sq=cfg["sigma"] ** 2,
        eps_grid=1e-2,
        S_marginal=2,
        n_em_iters=n_em,
        n_estep_iters=10,
        rho_sched=rho,
        learn_emissions=True,
        update_R=False,
        learn_kernel=True,
        n_mstep_iters=10,
        mstep_lr=0.01,
        n_hutchinson_mstep=16,
        kernel_warmup_iters=8,
        verbose=False,
    )
    return dict(
        mp=mp,
        op=op_out,
        hist=hist,
        wall=time.time() - t0,
        ls=float(hist.lengthscale[-1]),
        var=float(hist.variance[-1]),
    )


def fit_sparse(cfg, likelihood, output_params, init_params, t_grid, *, n_em):
    rho = jnp.logspace(-2, -1, n_em)
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sd = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp0 = dict(length_scales=jnp.full((D,), 1.5), output_scale=jnp.asarray(1.0))
    t0 = time.time()
    mp, _, gp_post, dp, _, op_out, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sd,
        likelihood=likelihood,
        t_grid=t_grid,
        drift_params=sp0,
        init_params=init_params,
        output_params=output_params,
        sigma=cfg["sigma"],
        rho_sched=rho,
        n_iters=n_em,
        n_iters_e=10,
        n_iters_m=10,
        perform_m_step=True,
        learn_output_params=True,
        learning_rate=0.01 * jnp.ones(n_em),
        print_interval=999,
        drift_params_history=[],
    )
    return dict(
        mp=mp,
        op=op_out,
        sd=sd,
        gp_post=gp_post,
        dp=dp,
        wall=time.time() - t0,
        ls=float(jnp.mean(dp["length_scales"])),
        var=float(dp["output_scale"]) ** 2,
    )


def eval_efgp_qf(mp, ls, var, t_grid, sigma, grid_inf, a):
    t = mp["m"][0].shape[0]
    x_template = jnp.linspace(-3.0, 3.0, max(t, 16))[:, None] * jnp.ones((1, D))
    x_min = np.asarray(x_template.min(axis=0))
    x_max = np.asarray(x_template.max(axis=0))
    x_extent = float((x_max - x_min).max())
    xcen = jnp.asarray(0.5 * (x_min + x_max), dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var, dtype=jnp.float32)),
        K_per_dim=10,
        X_extent=x_extent,
        xcen=xcen,
        d=D,
        eps=1e-2,
    )
    m = jnp.asarray(mp["m"][0])
    s = jnp.asarray(np.asarray(mp["S"][0]))
    ss = jnp.asarray(np.asarray(mp["SS"][0]))
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        m,
        s,
        ss,
        del_t,
        grid,
        jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2,
        S_marginal=2,
        D_lat=D,
        D_out=D,
    )
    ef, _, _ = jpd.drift_moments_jax(
        mu_r,
        grid,
        jnp.asarray(grid_inf, dtype=jnp.float32),
        D_lat=D,
        D_out=D,
    )
    return np.asarray(ef) @ a.T


def eval_sparse_qf(sd, mp, ls, var, t_grid, sigma, grid_inf, a):
    dp = dict(
        length_scales=jnp.full((D,), ls),
        output_scale=jnp.asarray(math.sqrt(var)),
    )
    gp_post = sd.update_dynamics_params(
        jr.PRNGKey(0),
        t_grid,
        mp,
        jnp.ones((1, t_grid.shape[0]), dtype=bool),
        dp,
        jnp.zeros((1, t_grid.shape[0], 1)),
        jnp.zeros((D, 1)),
        sigma,
    )
    ef = sd.get_posterior_f_mean(gp_post, dp, jnp.asarray(grid_inf, dtype=jnp.float32))
    return np.asarray(ef) @ a.T


def section_linear_convergence():
    print("\n" + "=" * 78)
    print("1. Linear row convergence under the notebook protocol")
    print("=" * 78)
    xs, ys, likelihood, output_params, init_params, t_grid = make_problem(LINEAR)
    xs_np = np.asarray(xs)
    for n_em in [25, 50, 100]:
        e = fit_efgp(LINEAR, likelihood, output_params, init_params, t_grid, n_em=n_em)
        s = fit_sparse(LINEAR, likelihood, output_params, init_params, t_grid, n_em=n_em)
        print(
            f"n_em={n_em:3d}  "
            f"EFGP: wall={e['wall']:5.1f}s  ell={e['ls']:.3f}  sigma2={e['var']:.3f}  "
            f"lat={latent_rmse(e['mp']['m'][0], xs_np):.4f}"
        )
        print(
            f"         "
            f"SparseGP: wall={s['wall']:5.1f}s  ell={s['ls']:.3f}  sigma2={s['var']:.3f}  "
            f"lat={latent_rmse(s['mp']['m'][0], xs_np):.4f}"
        )


def section_qf_isolation():
    print("\n" + "=" * 78)
    print("2. Linear row q(x) / q(f) / hyper decomposition at n_em=25")
    print("=" * 78)
    xs, ys, likelihood, output_params, init_params, t_grid = make_problem(LINEAR)
    xs_np = np.asarray(xs)
    grid_true = make_truth_grid(xs_np)
    f_true = linear_truth_field(grid_true)

    e = fit_efgp(LINEAR, likelihood, output_params, init_params, t_grid, n_em=25)
    s = fit_sparse(LINEAR, likelihood, output_params, init_params, t_grid, n_em=25)

    for tag, res in [("EFGP-solution", e), ("Sparse-solution", s)]:
        a, _, grid_inf = make_eval_grid(res["mp"]["m"][0], xs_np, grid_true)
        f_efgp = eval_efgp_qf(res["mp"], res["ls"], res["var"], t_grid, LINEAR["sigma"], grid_inf, a)
        f_sparse = eval_sparse_qf(s["sd"], res["mp"], res["ls"], res["var"], t_grid, LINEAR["sigma"], grid_inf, a)
        rmse_efgp = float(np.sqrt(np.mean((f_efgp - f_true) ** 2)))
        rmse_sparse = float(np.sqrt(np.mean((f_sparse - f_true) ** 2)))
        cross = float(np.sqrt(np.mean((f_efgp - f_sparse) ** 2)))
        print(f"{tag:16s}  EFGP-qf={rmse_efgp:.4f}  Sparse-qf={rmse_sparse:.4f}  cross={cross:.4f}")

    a_e, _, grid_e = make_eval_grid(e["mp"]["m"][0], xs_np, grid_true)
    a_s, _, grid_s = make_eval_grid(s["mp"]["m"][0], xs_np, grid_true)
    combos = [
        ("E_qx + E_hyp", e["mp"], e["ls"], e["var"], grid_e, a_e),
        ("E_qx + S_hyp", e["mp"], s["ls"], s["var"], grid_e, a_e),
        ("S_qx + E_hyp", s["mp"], e["ls"], e["var"], grid_s, a_s),
        ("S_qx + S_hyp", s["mp"], s["ls"], s["var"], grid_s, a_s),
    ]
    print("\nSparseGP q(f) held fixed while swapping q(x) and hypers:")
    for label, mp, ls, var, grid_inf, a in combos:
        f_pred = eval_sparse_qf(s["sd"], mp, ls, var, t_grid, LINEAR["sigma"], grid_inf, a)
        rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))
        print(f"  {label:14s}  drift_rmse={rmse:.4f}")


def section_sparse_pathologies():
    print("\n" + "=" * 78)
    print("3. SparseGP nonlinear rows under the notebook protocol")
    print("=" * 78)
    for cfg in [DUFFING, ANHARMONIC]:
        xs, ys, likelihood, output_params, init_params, t_grid = make_problem(cfg)
        s = fit_sparse(cfg, likelihood, output_params, init_params, t_grid, n_em=25)
        ls = s["ls"]
        var = s["var"]
        print(
            f"{cfg['name']:12s}  ell={ls}  sigma2={var}  "
            f"lat={latent_rmse(s['mp']['m'][0], np.asarray(xs)):.4f}  "
            f"mean(R)={float(np.mean(np.asarray(s['op']['R']))):.4f}"
        )


def main():
    print("JAX devices:", jax.devices())
    section_linear_convergence()
    section_qf_isolation()
    section_sparse_pathologies()


if __name__ == "__main__":
    main()
