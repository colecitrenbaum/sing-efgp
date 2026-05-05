"""
bench_on_trajectory.py

Compare bounding-box vs on-trajectory drift error metrics on two
canonical cells:
  1. Duffing T=2000 ls_init=0.7  — misspecified data ("SP wins on bbox")
  2. GP-drift T=2000 ls_init=0.7 — well-specified data ("EFGP wins on bbox")

For each, run EFGP + SP M=25 + SP M=64, compute:
  - bbox: drift_pc on a 14×14 grid over the trajectory's bounding box
            (with 0.4 padding) — what the existing benches report
  - traj: drift_pc evaluated at every x_t in the smoother's mean path
            — measures accuracy where data actually exists

If the bounding-box "SP wins on Duffing" claim was extrapolation-driven,
the traj metric should reverse it.

Run:
  ~/myenv/bin/python demos/bench_on_trajectory.py
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import sing.efgp_em as efgp_em

import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
T = 2000
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
VAR_INIT = 1.0
LS_INIT = 0.7

LS_TRUE_GP = 1.0
VAR_TRUE_GP = 1.5

# Reproducible drift seed for the GP-drift case
DRIFT_SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Drift definitions
# ──────────────────────────────────────────────────────────────────────
_DUFFING = lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]])


def _rbf_K(X1, X2, ls, var):
    sq = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return var * jnp.exp(-0.5 * sq / (ls ** 2))


def sample_gp_drift_fn(seed=DRIFT_SEED, lim=3.0, n_per=30):
    """Sample f ~ GP(0, RBF(LS_TRUE_GP, VAR_TRUE_GP)) on a grid, return
    (drift_fn, X_grid, alpha)."""
    a = jnp.linspace(-lim, lim, n_per)
    GX, GY = jnp.meshgrid(a, a, indexing='ij')
    X_grid = jnp.stack([GX.ravel(), GY.ravel()], axis=-1)
    M = X_grid.shape[0]
    K = _rbf_K(X_grid, X_grid, LS_TRUE_GP, VAR_TRUE_GP) + 1e-6 * jnp.eye(M)
    L = jnp.linalg.cholesky(K)
    z = jr.normal(jr.PRNGKey(seed), (M, D))
    f_grid = L @ z
    alpha = jnp.linalg.solve(K, f_grid)

    def f_true(x, t):
        Kxg = _rbf_K(x[None, :], X_grid, LS_TRUE_GP, VAR_TRUE_GP)[0]
        return Kxg @ alpha
    return f_true, X_grid, alpha


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────
class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data_duffing():
    sigma = 0.2
    t_max = 75.0  # 15 * (2000/400)
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(13), x0=jnp.array([1.2, 0.0]),
                      f=_DUFFING, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(13)
    C = jnp.asarray(rng.standard_normal((8, D)) * 0.5)
    ys = simulate_gaussian_obs(jr.PRNGKey(14), xs,
                                dict(C=C, d=jnp.zeros(8),
                                     R=jnp.full((8,), 0.05)))
    op = dict(C=C, d=jnp.zeros(8), R=jnp.full((8,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, sigma, _DUFFING, 'Duffing'


def make_data_gpdrift():
    drift_fn, _, _ = sample_gp_drift_fn()
    sigma = 0.2
    t_max = 40.0  # 8 * (2000/400)
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(137), x0=jnp.array([1.0, 0.0]),
                      f=drift_fn, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(99)
    C = jnp.asarray(rng.standard_normal((8, D)) * 0.5)
    ys = simulate_gaussian_obs(jr.PRNGKey(100), xs,
                                dict(C=C, d=jnp.zeros(8),
                                     R=jnp.full((8,), 0.05)))
    op = dict(C=C, d=jnp.zeros(8), R=jnp.full((8,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, sigma, drift_fn, 'GP-drift'


# ──────────────────────────────────────────────────────────────────────
# Fits (mirror the duffx64 helpers)
# ──────────────────────────────────────────────────────────────────────
def fit_efgp(lik, op, ip, t_grid, sigma, ls_init):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=N_M_INNER, mstep_lr=MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    return dict(mp=mp, ls=float(hist.lengthscale[-1]),
                var=float(hist.variance[-1]),
                wall=time.perf_counter() - t0)


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((N_EM,), MSTEP_LR),
        print_interval=999)
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2,
                wall=time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────
# Drift evaluators
# ──────────────────────────────────────────────────────────────────────
def eval_efgp_drift(mp, ls, var, t_grid, sigma, points):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(jnp.asarray(ls), jnp.asarray(var),
                                X_template, eps=1e-6)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(points), D_lat=D, D_out=D)
    return np.asarray(Ef)


def eval_sp_drift(s, points):
    return np.asarray(s['sd'].get_posterior_f_mean(
        s['gp_post'], s['dp'], jnp.asarray(points)))


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def procrustes_align(m_inf, m_true):
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


def metric_rel_mse(f_pred, f_true):
    mse = float(np.mean((f_pred - f_true) ** 2))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))
    return mse / max(var_f, 1e-12), var_f


def metrics_bbox(mp, xs_np, f_eval_fn, drift_fn):
    """Drift rmse on 14×14 grid in trajectory's bounding box (raw + pc)."""
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in grid_pts])
    f_pred_raw = f_eval_fn(grid_pts)
    rel_mse_raw, _ = metric_rel_mse(f_pred_raw, f_true)
    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    grid_inf = (grid_pts - b) @ np.linalg.inv(A).T
    f_pred_pc = f_eval_fn(grid_inf) @ A.T
    rel_mse_pc, var_f = metric_rel_mse(f_pred_pc, f_true)
    return dict(raw=rel_mse_raw, pc=rel_mse_pc, var_f=var_f,
                A=A, b=b, n_pts=len(grid_pts))


def metrics_on_trajectory(mp, xs_np, f_eval_fn, drift_fn):
    """Drift rmse evaluated at the smoother's trajectory points x_t,
    against the true drift f(x_t).  Uses every-other point to avoid
    double-counting fine-scale autocorrelation."""
    # Subsample (T=2000 → 1000 points) — keeps the metric well-defined
    # without overweighting nearby x_t pairs.
    m_inf = np.asarray(mp['m'][0])[::2]
    xs_sub = xs_np[::2]
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in xs_sub])     # truth at the actual x_t
    # Raw: evaluate the inferred drift at the inferred latent path.
    f_pred_raw = f_eval_fn(m_inf)
    rel_mse_raw, _ = metric_rel_mse(f_pred_raw, f_true)
    # Procrustes-aligned: map inferred path through (A, b) to truth frame
    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    m_in_truth = m_inf @ A.T + b
    # Evaluate at the inferred path location, bring back to truth frame.
    f_pred_pc = f_eval_fn(m_inf) @ A.T
    rel_mse_pc, var_f = metric_rel_mse(f_pred_pc, f_true)
    return dict(raw=rel_mse_raw, pc=rel_mse_pc, var_f=var_f,
                n_pts=len(xs_sub))


# ──────────────────────────────────────────────────────────────────────
# Run one (data, method) cell
# ──────────────────────────────────────────────────────────────────────
def run_method(method, lik, op, ip, t_grid, sigma, drift_fn, xs_np):
    if method == 'EFGP':
        e = fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
        f_eval = lambda pts: eval_efgp_drift(e['mp'], e['ls'], e['var'],
                                              t_grid, sigma, pts)
        ls = e['ls']; var = e['var']; wall = e['wall']; mp = e['mp']
    else:
        n_per = int(round(math.sqrt(int(method.split('=')[1]))))
        s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, LS_INIT)
        f_eval = lambda pts, s_=s: eval_sp_drift(s_, pts)
        ls = s['ls']; var = s['var']; wall = s['wall']; mp = s['mp']

    bbox = metrics_bbox(mp, xs_np, f_eval, drift_fn)
    traj = metrics_on_trajectory(mp, xs_np, f_eval, drift_fn)
    return dict(method=method, ls=ls, var=var, wall=wall,
                bbox=bbox, traj=traj)


def main():
    print(f"bench_on_trajectory: T={T} ls_init={LS_INIT} x64="
          f"{jax.config.read('jax_enable_x64')}")

    runs = {}
    for case_name, make_data in [
            ('Duffing (misspec)', make_data_duffing),
            ('GP-drift (well-spec)', make_data_gpdrift)]:
        print(f"\n{'='*72}")
        print(f"  {case_name}")
        print(f"{'='*72}", flush=True)
        xs, lik, op, ip, t_grid, sigma, drift_fn, _ = make_data()
        xs_np = np.asarray(xs)
        print(f"  trajectory range: x[0]∈[{xs_np[:,0].min():.2f}, "
              f"{xs_np[:,0].max():.2f}], x[1]∈[{xs_np[:,1].min():.2f}, "
              f"{xs_np[:,1].max():.2f}]", flush=True)

        cells = []
        for method in ['EFGP', 'SP M=25', 'SP M=64']:
            print(f"  fitting {method}...", flush=True)
            r = run_method(method, lik, op, ip, t_grid, sigma,
                            drift_fn, xs_np)
            cells.append(r)
            print(f"    ℓ={r['ls']:.3f}, σ²={r['var']:.3f}, "
                  f"wall={r['wall']:.1f}s", flush=True)
            print(f"    bbox  drift_pc={r['bbox']['pc']:.4f}  "
                  f"(grid {r['bbox']['n_pts']} pts, var_f={r['bbox']['var_f']:.3f})",
                  flush=True)
            print(f"    traj  drift_pc={r['traj']['pc']:.4f}  "
                  f"(traj {r['traj']['n_pts']} pts, var_f={r['traj']['var_f']:.3f})",
                  flush=True)
        runs[case_name] = cells

    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"  {'case':<24s}  {'method':>9s}  {'ℓ_final':>8s}  "
          f"{'σ²':>7s}  {'bbox_pc':>8s}  {'traj_pc':>8s}  "
          f"{'Δ(traj-bbox)':>13s}")
    for case_name, cells in runs.items():
        for r in cells:
            d = r['traj']['pc'] - r['bbox']['pc']
            print(f"  {case_name:<24s}  {r['method']:>9s}  "
                  f"{r['ls']:>8.3f}  {r['var']:>7.3f}  "
                  f"{r['bbox']['pc']:>8.4f}  {r['traj']['pc']:>8.4f}  "
                  f"{d:>+13.4f}")

    # Verdict on Duffing
    duffing = {r['method']: r for r in runs['Duffing (misspec)']}
    bbox_winner = min(duffing.values(), key=lambda r: r['bbox']['pc'])['method']
    traj_winner = min(duffing.values(), key=lambda r: r['traj']['pc'])['method']
    print(f"\n  Duffing bbox winner: {bbox_winner}")
    print(f"  Duffing traj winner: {traj_winner}")
    print(f"  → {'CHANGED' if bbox_winner != traj_winner else 'same'} ranking "
          f"under on-trajectory metric")
    gp = {r['method']: r for r in runs['GP-drift (well-spec)']}
    bbox_winner = min(gp.values(), key=lambda r: r['bbox']['pc'])['method']
    traj_winner = min(gp.values(), key=lambda r: r['traj']['pc'])['method']
    print(f"  GP-drift bbox winner: {bbox_winner}")
    print(f"  GP-drift traj winner: {traj_winner}")
    print(f"  → {'CHANGED' if bbox_winner != traj_winner else 'same'} ranking")


if __name__ == '__main__':
    main()
