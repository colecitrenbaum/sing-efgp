"""
bench_long_lsinit.py

Test: do EFGP and SparseGP swap on drift-recovery rmse if we initialise ℓ
near the GT MLE rather than at ℓ=0.7?  Hypothesis: SparseGP's edge on
drift rmse (esp. Anharmonic T=2000) comes from its undershooting ℓ
(more wiggly drift) when starting from ℓ=0.7.  If we start near MLE,
EFGP — which converges *to* MLE — should beat SparseGP on drift rmse
because SparseGP still drifts toward smaller ℓ during EM.

Setup (matched to bench_overnight_sweep.py):
  T = 2000, n_em = 50, lr = 0.01, n_iters_m = 4
  benchmarks = [linear, duffing, anharmonic]
  methods = [EFGP, SP M=25, SP M=64]   (skip M=144 — NaNs)
  ls_init values = [0.7 (baseline, in existing npz), 3.0, 5.0]

Output: /tmp/lsinit_<short>_T2000_lsi<X>.npz per cell + summary table.

Run:
  ~/myenv/bin/python demos/bench_long_lsinit.py
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

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import sing.efgp_em as efgp_em

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


D = 2
T = 2000
N_EM = 50
MSTEP_LR = 0.01
N_M_INNER = 4
M_LIST = [25, 64]                # skip 144 (NaN-prone)
LS_INIT_LIST = [3.0, 5.0]        # baseline ls_init=0.7 already in /tmp
VAR_INIT = 1.0


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

BENCHMARKS = [
    dict(name='Damped rotation (linear)', short='linear',
          drift_fn=lambda x, t: _A_rot @ x,
          x0=jnp.array([2.0, 0.0]),
          T_base=400, t_max_base=8.0, sigma=0.3, N_obs=8, seed=7),
    dict(name='Duffing double-well', short='duffing',
          drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
          x0=jnp.array([1.2, 0.0]),
          T_base=400, t_max_base=15.0, sigma=0.2, N_obs=8, seed=13),
    dict(name='Anharmonic oscillator', short='anharmonic',
          drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3]),
          x0=jnp.array([1.5, 0.0]),
          T_base=400, t_max_base=10.0, sigma=0.3, N_obs=8, seed=21),
]


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg, T):
    t_max = cfg['t_max_base'] * (T / cfg['T_base'])
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=t_max,
                      n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def procrustes_align(m_inf, m_true):
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


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
    wall = time.perf_counter() - t0
    return dict(mp=mp,
                ls=float(hist.lengthscale[-1]),
                var=float(hist.variance[-1]),
                wall=wall)


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((N_EM,), MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2,
                wall=wall)


def eval_efgp_drift(mp, ls, var, t_grid, sigma, grid_pts):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(
        jnp.asarray(ls, dtype=jnp.float32),
        jnp.asarray(var, dtype=jnp.float32),
        X_template, eps=1e-6)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    return np.asarray(Ef)


def eval_sp_drift(s, grid_pts):
    return np.asarray(s['sd'].get_posterior_f_mean(
        s['gp_post'], s['dp'],
        jnp.asarray(grid_pts, dtype=jnp.float32)))


def compute_drift_metrics(mp, xs_np, f_eval_fn, drift_fn):
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    f_true = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.))
                        for p in grid_pts])
    A, b = procrustes_align(np.asarray(mp['m'][0]), xs_np)
    grid_inf = (grid_pts - b) @ np.linalg.inv(A).T
    f_pred = f_eval_fn(grid_inf) @ A.T

    mse = float(np.mean((f_pred - f_true) ** 2))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))
    rel_mse = mse / max(var_f, 1e-12)
    return dict(rel_mse=rel_mse)


def latent_recovery_rmse(mp, xs_np):
    """Procrustes-aligned latent recovery RMSE: ‖A m̂ + b − x‖² / dim."""
    m_inf = np.asarray(mp['m'][0])
    A, b = procrustes_align(m_inf, xs_np)
    m_aligned = m_inf @ A.T + b
    return float(np.sqrt(np.mean((m_aligned - xs_np) ** 2)))


def run_one(cfg, ls_init):
    short = cfg['short']
    print(f"\n=== {cfg['name']} | ls_init={ls_init} ===", flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(cfg, T)
    sigma = cfg['sigma']
    xs_np = np.asarray(xs)

    rows = []

    # EFGP
    print(f"  EFGP fit (ls_init={ls_init})...", flush=True)
    e = fit_efgp(lik, op, ip, t_grid, sigma, ls_init)
    e_drift = compute_drift_metrics(
        e['mp'], xs_np,
        f_eval_fn=lambda g: eval_efgp_drift(e['mp'], e['ls'], e['var'],
                                              t_grid, sigma, g),
        drift_fn=cfg['drift_fn'])
    e_lat = latent_recovery_rmse(e['mp'], xs_np)
    print(f"    ℓ={e['ls']:.3f}, σ²={e['var']:.3f}, "
          f"drift_rmse²/var={e_drift['rel_mse']:.3f}, "
          f"latent_rmse={e_lat:.4f}, wall={e['wall']:.1f}s",
          flush=True)
    rows.append(('EFGP', None, e['ls'], e['var'], e_drift['rel_mse'],
                 e_lat, e['wall']))

    # SparseGP per M
    for M in M_LIST:
        n_per_dim = int(round(math.sqrt(M)))
        print(f"  SP M={M} ({n_per_dim}x{n_per_dim}) fit...", flush=True)
        try:
            s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per_dim, ls_init)
            s_drift = compute_drift_metrics(
                s['mp'], xs_np,
                f_eval_fn=lambda g, s_=s: eval_sp_drift(s_, g),
                drift_fn=cfg['drift_fn'])
            s_lat = latent_recovery_rmse(s['mp'], xs_np)
            ls_str = f"{s['ls']:.3f}" if math.isfinite(s['ls']) else 'NaN'
            var_str = f"{s['var']:.3f}" if math.isfinite(s['var']) else 'NaN'
            rmse_str = (f"{s_drift['rel_mse']:.3f}"
                        if math.isfinite(s_drift['rel_mse']) else 'NaN')
            lat_str = f"{s_lat:.4f}" if math.isfinite(s_lat) else 'NaN'
            print(f"    ℓ={ls_str}, σ²={var_str}, "
                  f"drift_rmse²/var={rmse_str}, "
                  f"latent_rmse={lat_str}, wall={s['wall']:.1f}s",
                  flush=True)
            rows.append((f'SP M={M}', M, s['ls'], s['var'],
                         s_drift['rel_mse'], s_lat, s['wall']))
        except Exception as ex:
            print(f"    EXCEPTION: {type(ex).__name__}: {ex}", flush=True)
            rows.append((f'SP M={M}', M, np.nan, np.nan, np.nan,
                         np.nan, np.nan))

    return short, rows


def main():
    print(f"bench_long_lsinit: T={T} nem={N_EM} ls_init={LS_INIT_LIST}",
          flush=True)
    all_results = {}  # (short, ls_init) -> rows
    for ls_init in LS_INIT_LIST:
        for cfg in BENCHMARKS:
            short, rows = run_one(cfg, ls_init)
            all_results[(short, ls_init)] = rows

    print(f"\n{'='*100}")
    print(f"Final summary table:")
    print(f"{'='*100}")
    print(f"  {'bench':<11s}  {'ls_init':>7s}  {'method':>9s}  "
          f"{'ℓ_final':>8s}  {'σ²_final':>8s}  {'drift':>7s}  "
          f"{'latent':>7s}  {'wall':>7s}")
    for ls_init in LS_INIT_LIST:
        for cfg in BENCHMARKS:
            short = cfg['short']
            rows = all_results.get((short, ls_init), [])
            for r in rows:
                method, M, ls, var, drift, lat, wall = r
                ls_s = f"{ls:.3f}" if math.isfinite(ls) else 'NaN'
                var_s = f"{var:.3f}" if math.isfinite(var) else 'NaN'
                drift_s = f"{drift:.3f}" if math.isfinite(drift) else 'NaN'
                lat_s = f"{lat:.4f}" if math.isfinite(lat) else 'NaN'
                wall_s = f"{wall:.0f}s" if math.isfinite(wall) else 'NaN'
                print(f"  {short:<11s}  {ls_init:>7.1f}  {method:>9s}  "
                      f"{ls_s:>8s}  {var_s:>8s}  {drift_s:>7s}  "
                      f"{lat_s:>7s}  {wall_s:>7s}")


if __name__ == '__main__':
    main()
