"""
bench_duffing_kscaling.py

Companion to bench_duffing_scaling.py: SAME Duffing setup and SAME canonical
head-to-head settings, but the swept axis is the NUMBER OF TRIALS K
(diverse ICs), each of fixed length T=1000, instead of the sequence length.

K in {1, 10, 100}; methods = EFGP, SparseGP M=49, M=100 (isotropic).
Metric = NRMSE of the drift at all trials' visited states (on-support).
Everything else identical to the T-sweep: Duffing f(x)=[x1, x0-x0^3-0.5 x1],
sigma=0.2, oracle-fixed emissions, learn-kernel canonical hypers (n_em=50,
n_estep=10, n_mstep=4, mstep_lr=0.01, rho 0.05->0.7, warmup=8, eps_grid=1e-3),
ls_init=0.7. Reuses the fit + metric machinery verbatim; only the data (now K
diverse-IC trials) and the multi-trial aggregation differ.

Why this complements the T-sweep: EFGP wall tracks TOTAL K*T (indifferent to
the K/T split), while SparseGP's SING smoother is sequential in T_per_trial
(=1000 here, fixed) and vmaps over K -> the K-split keeps its scans shallow, so
SparseGP scales far better along K than along T. See CLAUDE.md "Wall-time
scaling: K trials x T vs single trial x K*T".

Run (GPU node):
  python -u demos/bench_duffing_kscaling.py --K 10 --method efgp ...
  python -u demos/bench_duffing_kscaling.py --K 10 --method sp --M 49 ...
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

import demos.bench_gpdrift_scaling as run    # _fit_efgp_with_hist, _drift_metrics_at_states, _s_diag
import demos.bench_gpdrift_x64 as base       # efgp_em, eval_sp_drift, latent_recovery_rmse, procrustes
import demos.bench_gpdrift_inducing_sweep_iso_x64 as iso  # IsotropicRBF
import demos.bench_duffing_lsinit_x64 as duf  # Duffing CFG, simulate helpers, GLik

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from sing.sde import SparseGP
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em

D = base.D
T_FIXED = 1000
DUFFING_DRIFT = duf.CFG['drift_fn']


def duffing_f_true(pts):
    p = np.asarray(pts)
    x0, x1 = p[:, 0], p[:, 1]
    return np.stack([x1, x0 - x0 ** 3 - 0.5 * x1], axis=-1)


def make_data_K(K, T, seed):
    """K diverse-IC Duffing trajectories (K,T,D) with shared oracle emissions.
    Per-trial seeds base+7000+k so the first K trials are identical across K
    values (subset property). t_max=15*(T/400), dt~0.0375 (== single-trial)."""
    cfg = duf.CFG
    t_max = cfg['t_max_base'] * (T / cfg['T_base'])
    sigma = cfg['sigma']
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    # diverse ICs spread over the double-well region (both wells at x0~+-1)
    ic_rng = np.random.default_rng(cfg['seed'] + seed + 12345)
    x0_K = jnp.asarray(ic_rng.normal(0.0, 1.0, size=(K, D)))
    xs_K = jnp.stack([
        jnp.clip(duf.simulate_sde(jr.PRNGKey(cfg['seed'] + seed + 7000 + k),
                                  x0=x0_K[k], f=DUFFING_DRIFT, t_max=t_max,
                                  n_timesteps=T, sigma=sigma_fn), -3.0, 3.0)
        for k in range(K)], axis=0)                      # (K, T, D)

    # emissions identical to the single-trial runner (same rng seed/stream)
    C_true = jnp.asarray(
        np.random.default_rng(cfg['seed'] + seed).standard_normal(
            (cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys_K = jnp.stack([
        duf.simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + seed + 8000 + k),
                                  xs_K[k], out_true)
        for k in range(K)], axis=0)                      # (K, T, N_obs)
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K, 1, 1)))
    t_grid = jnp.linspace(0., t_max, T)
    lik = duf.GLik(ys_K, jnp.ones((K, T), dtype=bool))
    return xs_K, lik, op, ip, t_grid, sigma


def _fit_sparsegp_iso_multitrial(lik, op, ip, t_grid, sigma, num_per_dim,
                                 ls_init, xs_K, learn_kernel=True, var_init=None):
    """Multi-trial isotropic SparseGP fit (data-aware zs over (K,T) bbox).
    Same canonical settings as iso.fit_sparsegp_iso; returns the same st dict
    shape so base.eval_sp_drift works unchanged."""
    N_EM = base.N_EM
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    xs_np = np.asarray(xs_K)
    lo = xs_np.min(axis=(0, 1)) - 0.4                    # (D,) multi-trial bbox
    hi = xs_np.max(axis=(0, 1)) + 0.4
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim) for d in range(D)]
    zs = jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'), axis=-1).reshape(-1, D)
    sparse = SparseGP(zs=zs, kernel=iso.IsotropicRBF(latent_dim=D),
                      expectation=quad)
    _var0 = base.VAR_INIT if var_init is None else float(var_init)
    drift_params0 = dict(length_scale=jnp.asarray(float(ls_init)),
                         output_scale=jnp.asarray(math.sqrt(_var0)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=base.N_M_INNER, perform_m_step=bool(learn_kernel),
        learn_output_params=False,
        learning_rate=jnp.full((N_EM,), base.MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] + [float(h['length_scale']) for h in history])
    var_traj = np.array([base.VAR_INIT] +
                        [float(h['output_scale']) ** 2 for h in history])
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls_traj=ls_traj, var_traj=var_traj,
                ls=float(dp['length_scale']),
                var=float(dp['output_scale']) ** 2, wall=wall)


def run_cell(K, T, method, M, ls_init, seed, eps_grid=1e-3, k_min_ls=None,
             learn_kernel=True, var_init=None, restore_qf_variance='none'):
    xs_K, lik, op, ip, t_grid, sigma = make_data_K(K, T, seed)
    xs_np = np.asarray(xs_K)                              # (K, T, D)
    dt = float(np.asarray(t_grid[1] - t_grid[0]))

    if method == 'efgp':
        # DATA-AWARE spectral-grid box (trajectory bbox) so the grid doesn't
        # waste modes on the empty mu0+-3 union box in the diverse-IC regime
        # (see K-sweep diagnosis). Matches SparseGP's data-aware inducing bbox.
        # k_min_ls forces the mode lattice to resolve down to that lengthscale
        # (l* aliasing threshold) so the M-step can't run l below the grid.
        xt = jnp.asarray(run.data_aware_template(xs_np))
        st = run._fit_efgp_with_hist(lik, op, ip, t_grid, sigma, ls_init,
                                     eps_grid=eps_grid, x_template=xt,
                                     k_min_lengthscale=k_min_ls,
                                     learn_kernel=learn_kernel,
                                     variance=var_init,
                                     restore_qf_variance=restore_qf_variance)
        f_eval_fn = lambda g: base.efgp_em.posterior_drift_mean(st['hist'], g)
        M_out = 0
    elif method == 'sp':
        n_per = int(round(math.sqrt(M)))
        st = _fit_sparsegp_iso_multitrial(lik, op, ip, t_grid, sigma, n_per,
                                          ls_init, xs_K,
                                          learn_kernel=learn_kernel,
                                          var_init=var_init)
        f_eval_fn = lambda g: base.eval_sp_drift(st, g)
        M_out = M
    else:
        raise ValueError(f"unknown method {method!r}")

    # Flatten across trials -> reuse the single-trajectory states metric &
    # latent recovery (drift is shared; oracle emissions -> one global frame).
    m_flat = np.asarray(st['mp']['m']).reshape(-1, D)
    xs_flat = xs_np.reshape(-1, D)
    mp_flat = {'m': m_flat[None]}
    dstate = run._drift_metrics_at_states(mp_flat, xs_flat, duffing_f_true,
                                          f_eval_fn)
    lat = base.latent_recovery_rmse(mp_flat, xs_flat)

    return dict(
        status='ok', err='',
        dynamics='duffing', sweep='K',
        K=K, T=T, method=method, M=M_out, ls_init=ls_init, seed=seed, dt=dt,
        eps_grid=eps_grid,
        wall=st['wall'],
        ls_final=st['ls'], var_final=st['var'],
        ls_traj=st['ls_traj'], var_traj=st['var_traj'],
        drift_nrmse=dstate['nrmse'], drift_nrmse_raw=dstate['nrmse_raw'],
        drift_rel_mse=dstate['rel_mse'], drift_rel_mse_raw=dstate['rel_mse_raw'],
        var_f=dstate['var_f'],
        eval_pts=dstate['eval_pts'], f_true_states=dstate['f_true'],
        f_pred_states_pc=dstate['f_pred_pc'], f_pred_states_raw=dstate['f_pred_raw'],
        procrustes_A=dstate['A'], procrustes_b=dstate['b'],
        lat_pc=lat['pc'], lat_raw=lat['raw'],
        # trial 0 for the posterior sanity figure
        m_inf=np.asarray(st['mp']['m'][0]),
        S_diag=run._s_diag(st['mp']),
        xs_true=xs_np[0],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--T', type=int, default=T_FIXED)
    ap.add_argument('--method', choices=['efgp', 'sp'], required=True)
    ap.add_argument('--M', type=int, default=0)
    ap.add_argument('--ls-init', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps-grid', type=float, default=1e-3)
    ap.add_argument('--k-min-ls', type=float, default=None,
                    help='EFGP: force spectral grid to resolve down to this '
                         'lengthscale (more Fourier modes). None -> auto from ls_init')
    ap.add_argument('--no-learn-kernel', action='store_true',
                    help='EFGP: freeze kernel hypers at ls_init/var_init (M-step off)')
    ap.add_argument('--var-init', type=float, default=None,
                    help='EFGP: initial/fixed kernel variance (default VAR_INIT=1.0)')
    ap.add_argument('--restore-qf-var', default='none',
                    choices=['none', 'hutch', 'hutch_hetS'],
                    help="EFGP: restore dropped E[V] drift-uncertainty feedback")
    ap.add_argument('--out-dir', default='demos/_bench_duffing_kscaling_out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}{args.M if args.method == 'sp' else ''}"
    eps_tag = "" if (args.method != 'efgp' or args.eps_grid == 1e-3) \
        else f"_eps{args.eps_grid:g}"
    kmin_tag = "" if (args.method != 'efgp' or args.k_min_ls is None) \
        else f"_kmin{args.k_min_ls:g}"
    out_path = out_dir / f"cell_K{args.K}_{tag}{eps_tag}{kmin_tag}_seed{args.seed}.npz"

    print(f"[kscale] K={args.K} T={args.T} method={args.method} M={args.M} "
          f"ls_init={args.ls_init} seed={args.seed} eps={args.eps_grid} x64="
          f"{jax.config.read('jax_enable_x64')} dev={jax.devices()}", flush=True)

    t0 = time.perf_counter()
    try:
        res = run_cell(args.K, args.T, args.method, args.M, args.ls_init,
                       args.seed, eps_grid=args.eps_grid, k_min_ls=args.k_min_ls,
                       learn_kernel=(not args.no_learn_kernel),
                       var_init=args.var_init,
                       restore_qf_variance=args.restore_qf_var)
        print(f"[kscale] OK wall={res['wall']:.1f}s "
              f"drift_nrmse={res['drift_nrmse']:.4f} lat_pc={res['lat_pc']:.4f} "
              f"l={res['ls_final']:.3f} var={res['var_final']:.3f}", flush=True)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[kscale] FAILED after {time.perf_counter()-t0:.1f}s:\n{tb}",
              flush=True)
        res = dict(status='failed', err=f"{type(e).__name__}: {e}\n{tb}",
                   dynamics='duffing', sweep='K', K=args.K, T=args.T,
                   method=args.method, M=args.M if args.method == 'sp' else 0,
                   ls_init=args.ls_init, seed=args.seed)

    np.savez(out_path, **res)
    print(f"[kscale] wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
