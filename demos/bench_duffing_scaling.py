"""
bench_duffing_scaling.py

Single-cell runner for the EFGP-SING vs SparseGP-SING *scaling law* on the
Duffing double-well SDE (wall time & drift NRMSE vs sequence length T).

Why Duffing (not the GP-sampled drift): the canonical GP-drift sample is
NON-confining -- from x0 it diverges and slams into the clip(+/-3) wall (~96% of
steps), so `f_true` extrapolates to huge values at the wall while the clipped
velocity ->0, poisoning any drift metric and breaking the T-scaling premise
(more time = more time stuck at the wall). See demos/diag_drift_problem.py.
Duffing f(x)=[x1, x0-x0^3-0.5 x1] is confining: a single trajectory explores a
bounded 2D region (0% escape) for arbitrarily long T -- a clean sequence-length
scaling axis. The drift is known analytically, so the metric is exact. Both
methods face the same (mild) RBF misspecification -> fair.

One invocation = one (method, M, T) cell -> one .npz in --out-dir.

Metric (primary): NRMSE of the drift AT the true trajectory states x_t
(on-support), = sqrt( mean_t||f_pred(x_t)-f_true(x_t)||^2 / Var_t(f_true) ).
Emissions are pinned to oracle so the latent frame is fixed (Procrustes ~ I).

All fit hyperparameters are the canonical head-to-head settings, identical
between methods; SparseGP uses the ISOTROPIC kernel (single shared l) to match
EFGP. Reuses the fit + metric machinery from bench_gpdrift_scaling verbatim;
only the dynamics/true-drift change.

Run (GPU node):
  python -u demos/bench_duffing_scaling.py --T 1000 --method efgp \
      --ls-init 0.7 --seed 0 --out-dir demos/_bench_duffing_scaling_out
  python -u demos/bench_duffing_scaling.py --T 1000 --method sp --M 49 ...
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse fit/metric machinery (module import enables jax_enable_x64 first).
import demos.bench_gpdrift_scaling as run    # _fit_efgp_with_hist, _drift_metrics_at_states, _s_diag
import demos.bench_gpdrift_x64 as base       # efgp_em, eval_sp_drift, latent_recovery_rmse
import demos.bench_gpdrift_inducing_sweep_iso_x64 as iso  # fit_sparsegp_iso (isotropic)
import demos.bench_duffing_lsinit_x64 as duf  # Duffing CFG, simulate helpers, GLik

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

D = base.D
DUFFING_DRIFT = duf.CFG['drift_fn']      # lambda x,t: [x1, x0 - x0^3 - 0.5 x1]


def duffing_f_true(pts):
    """Vectorized Duffing drift at points pts (n, D) -> (n, D)."""
    p = np.asarray(pts)
    x0, x1 = p[:, 0], p[:, 1]
    return np.stack([x1, x0 - x0 ** 3 - 0.5 * x1], axis=-1)


def make_data(T, seed):
    """Duffing SDE, single trajectory, oracle-fixed emissions. t_max scales as
    15*(T/400) with dt held ~0.0375 (canonical Duffing regime)."""
    cfg = duf.CFG
    t_max = cfg['t_max_base'] * (T / cfg['T_base'])
    sigma = cfg['sigma']
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = duf.simulate_sde(jr.PRNGKey(cfg['seed'] + seed), x0=cfg['x0'],
                          f=DUFFING_DRIFT, t_max=t_max, n_timesteps=T,
                          sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)         # safety net; Duffing never reaches it
    rng = np.random.default_rng(cfg['seed'] + seed)
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = duf.simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + seed + 1), xs,
                                   out_true)
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda a: a[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = duf.GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, sigma


def run_cell(T, method, M, ls_init, seed, eps_grid=1e-3):
    xs, lik, op, ip, t_grid, sigma = make_data(T, seed)
    xs_np = np.asarray(xs)
    dt = float(np.asarray(t_grid[1] - t_grid[0]))

    if method == 'efgp':
        st = run._fit_efgp_with_hist(lik, op, ip, t_grid, sigma, ls_init,
                                     eps_grid=eps_grid)
        f_eval_fn = lambda g: base.efgp_em.posterior_drift_mean(st['hist'], g)
        M_out = 0
    elif method == 'sp':
        n_per = int(round(math.sqrt(M)))
        st = iso.fit_sparsegp_iso(lik, op, ip, t_grid, sigma, n_per, ls_init,
                                  xs_np)
        f_eval_fn = lambda g: base.eval_sp_drift(st, g)
        M_out = M
    else:
        raise ValueError(f"unknown method {method!r}")

    dstate = run._drift_metrics_at_states(st['mp'], xs_np, duffing_f_true,
                                          f_eval_fn)
    lat = base.latent_recovery_rmse(st['mp'], xs_np)

    return dict(
        status='ok', err='',
        dynamics='duffing',
        T=T, method=method, M=M_out, ls_init=ls_init, seed=seed, dt=dt,
        eps_grid=eps_grid,
        wall=st['wall'],
        ls_final=st['ls'], var_final=st['var'],
        ls_traj=st['ls_traj'], var_traj=st['var_traj'],
        # primary on-support metric
        drift_nrmse=dstate['nrmse'], drift_nrmse_raw=dstate['nrmse_raw'],
        drift_rel_mse=dstate['rel_mse'], drift_rel_mse_raw=dstate['rel_mse_raw'],
        var_f=dstate['var_f'],
        # persisted arrays for offline re-scoring
        eval_pts=dstate['eval_pts'], f_true_states=dstate['f_true'],
        f_pred_states_pc=dstate['f_pred_pc'], f_pred_states_raw=dstate['f_pred_raw'],
        procrustes_A=dstate['A'], procrustes_b=dstate['b'],
        lat_pc=lat['pc'], lat_raw=lat['raw'],
        m_inf=np.asarray(st['mp']['m'][0]),
        S_diag=run._s_diag(st['mp']),
        xs_true=xs_np,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, required=True)
    ap.add_argument('--method', choices=['efgp', 'sp'], required=True)
    ap.add_argument('--M', type=int, default=0)
    ap.add_argument('--ls-init', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps-grid', type=float, default=1e-3)
    ap.add_argument('--out-dir', default='demos/_bench_duffing_scaling_out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}{args.M if args.method == 'sp' else ''}"
    eps_tag = "" if (args.method != 'efgp' or args.eps_grid == 1e-3) \
        else f"_eps{args.eps_grid:g}"
    out_path = out_dir / f"cell_T{args.T}_{tag}{eps_tag}_seed{args.seed}.npz"

    print(f"[duffscale] T={args.T} method={args.method} M={args.M} "
          f"ls_init={args.ls_init} seed={args.seed} eps={args.eps_grid} x64="
          f"{jax.config.read('jax_enable_x64')} dev={jax.devices()}", flush=True)

    t0 = time.perf_counter()
    try:
        res = run_cell(args.T, args.method, args.M, args.ls_init, args.seed,
                       eps_grid=args.eps_grid)
        print(f"[duffscale] OK wall={res['wall']:.1f}s "
              f"drift_nrmse={res['drift_nrmse']:.4f} "
              f"lat_pc={res['lat_pc']:.4f} "
              f"l={res['ls_final']:.3f} var={res['var_final']:.3f}", flush=True)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[duffscale] FAILED after {time.perf_counter()-t0:.1f}s:\n{tb}",
              flush=True)
        res = dict(status='failed', err=f"{type(e).__name__}: {e}\n{tb}",
                   dynamics='duffing', T=args.T, method=args.method,
                   M=args.M if args.method == 'sp' else 0,
                   ls_init=args.ls_init, seed=args.seed)

    np.savez(out_path, **res)
    print(f"[duffscale] wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
