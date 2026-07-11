"""
bench_gpdrift_scaling.py

Single-cell runner for the EFGP-SING vs SparseGP-SING synthetic GP-drift
*scaling law* (wall time & drift error vs sequence length T).

One invocation = one (method, M, T, seed) cell. Writes one .npz to --out-dir.
Submit the 9-cell grid via demos/submit_gpdrift_scaling.sh; assemble figures
with demos/plot_gpdrift_scaling.py and demos/plot_gpdrift_posteriors.py.

All fit hyperparameters (n_em=50, n_estep=10, n_mstep=4, mstep_lr=0.01,
rho ramp 0.05->0.7, kernel learning on, oracle-fixed emissions, eps_grid=1e-3,
gmix E-step) are inherited *verbatim* from demos/bench_gpdrift_x64.py via
`import ... as base`, so EFGP and SparseGP are apples-to-apples and this file
only varies T. The scaling regime is t_max = 8*(T/400) with dt held at 0.02
(canonical make_data).

Run (on a GPU node):
  python -u demos/bench_gpdrift_scaling.py --T 1000 --method efgp \
      --ls-init 0.7 --seed 0 --out-dir demos/_bench_gpdrift_scaling_out
  python -u demos/bench_gpdrift_scaling.py --T 1000 --method sp --M 49 ...
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

# Importing the canonical bench enables jax_enable_x64 at its module top
# (before any jax computation) and gives us every stateless helper + the
# exact fit calls. main() is guarded by __main__, so nothing runs on import.
import demos.bench_gpdrift_x64 as base  # noqa: E402
# Isotropic-kernel SparseGP (single shared lengthscale, NOT ARD) — the
# apples-to-apples match to EFGP's isotropic ℓ. Reuses the exact IsotropicRBF +
# fit_sparsegp_iso from the established isotropic inducing sweep.
import demos.bench_gpdrift_inducing_sweep_iso_x64 as iso  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402


def make_data(T, seed):
    """T-parametrized copy of base.make_data.

    Same GP-sampled drift, SDE, and oracle emissions as bench_gpdrift_x64, but
    with T and a seed offset as arguments. Keeps t_max = 8*(T/400) so dt = 0.02
    is constant across T (more data = longer observation of the ergodic SDE).
    Returns (xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha).
    """
    D = base.D
    drift_key = jr.PRNGKey(base.DRIFT_SEED + seed)
    X_grid, alpha, _ = base.sample_gp_drift(drift_key, base.LS_TRUE, base.VAR_TRUE)
    f_true = base.make_drift_fn(X_grid, alpha, base.LS_TRUE, base.VAR_TRUE)

    t_max = 8.0 * (T / 400.0)
    sigma_fn = lambda x, t: base.SIGMA_SDE * jnp.eye(D)
    x0 = jnp.array([1.0, 0.0])
    xs = base.simulate_sde(jr.PRNGKey(base.SDE_SEED + seed), x0=x0, f=f_true,
                           t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)

    rng = np.random.default_rng(base.EMISSION_SEED + seed)
    C_true = jnp.asarray(rng.standard_normal((base.N_OBS, D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(base.N_OBS),
                    R=jnp.full((base.N_OBS,), 0.05))
    ys = base.simulate_gaussian_obs(jr.PRNGKey(base.EMISSION_SEED + seed + 1),
                                    xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(base.N_OBS),
              R=jnp.full((base.N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda a: a[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = base.GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, lik, op, ip, t_grid, base.SIGMA_SDE, f_true, X_grid, alpha


def _fit_efgp_with_hist(lik, op, ip, t_grid, sigma, ls_init, eps_grid=1e-3,
                        x_template=None, k_min_lengthscale=None,
                        learn_kernel=True, variance=None,
                        restore_qf_variance='none'):
    """Same canonical fit call as base.fit_efgp, but also returns the EM
    history so drift can be evaluated from the EM's ACTUAL converged q(f) via
    efgp_em.posterior_drift_mean (base.eval_efgp_drift recomputes mu_r through
    a now-stale compute_mu_r_jax signature).

    eps_grid is exposed so we can test whether the spectral-grid truncation is
    fine enough for this problem (too coarse -> band-limited drift -> inflated
    drift error / biased sigma^2).

    x_template: optional (P, D) array whose min/max set the spectral-grid box.
    Pass a DATA-AWARE box (trajectory bbox) so the grid doesn't squander modes
    on the empty mu0+-3 default box -- matches how SparseGP places its inducing
    points, and is essential in the diverse-IC regime where the default box
    balloons (see K-sweep diagnosis). None -> library default (mu0 +- halo)."""
    N_EM = base.N_EM
    var0 = base.VAR_INIT if variance is None else float(variance)
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=base.D,
        lengthscale=ls_init, variance=var0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=eps_grid, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=learn_kernel, n_mstep_iters=base.N_M_INNER,
        mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        X_template=x_template,
        K_min_lengthscale=k_min_lengthscale,   # force enough modes: grid
        # resolves down to this l (l* aliasing threshold). None -> auto from ls_init.
        restore_qf_variance=restore_qf_variance,
        verbose=False)
    wall = time.perf_counter() - t0
    # when learn_kernel=False hist.lengthscale/variance may be empty -> use inits
    ls_hist = list(hist.lengthscale) if len(hist.lengthscale) else [ls_init]
    var_hist = list(hist.variance) if len(hist.variance) else [var0]
    return dict(mp=mp, hist=hist,
                ls_traj=np.array([ls_init] + ls_hist),
                var_traj=np.array([var0] + var_hist),
                ls=float(ls_hist[-1]),
                var=float(var_hist[-1]),
                wall=wall)


def data_aware_template(xs, pad=0.4, n=16):
    """(n, D) template whose bbox = data bbox +- pad -> a DATA-AWARE spectral
    grid box for EFGP (mirrors SparseGP's data-aware inducing bbox). xs may be
    (T,D) or (K,T,D)."""
    X = np.asarray(xs)
    X = X.reshape(-1, X.shape[-1])
    lo = X.min(0) - pad
    hi = X.max(0) + pad
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return center[None, :] + np.linspace(-1., 1., n)[:, None] * half[None, :]


def _s_diag(mp):
    """Diagonal marginal variances (T, D) from marginal_params['S'] (K,T,D,D)."""
    S = np.asarray(mp['S'][0])                       # (T, D, D)
    return np.einsum('tii->ti', S)                   # (T, D)


STATES_MAX = 2000   # cap #eval points for the trajectory-state drift metric


def _f_true_batch(pts, X_grid, alpha):
    """Vectorized true drift f_true(x)=K(x,X_grid)@alpha at points pts (n,D)."""
    K = base._rbf_K(jnp.asarray(pts), jnp.asarray(X_grid),
                    base.LS_TRUE, base.VAR_TRUE)
    return np.asarray(K @ jnp.asarray(alpha))        # (n, D)


def _drift_metrics_at_states(mp, xs_np, f_true_fn, f_eval_fn):
    """Drift error evaluated AT the trajectory states x_t (subsampled), i.e.
    where the SDE actually has support -- unlike the padded bbox grid, this is
    not dominated by unvisited-extrapolation cells.

    Primary metric = NRMSE = sqrt(MSE / Var(f_true)); rel_mse = NRMSE^2 =
    MSE/Var kept for continuity. Procrustes-aligns the inferred-frame drift to
    the true frame (identity under oracle emissions, kept for safety). Persists
    the evaluated arrays so any metric variant can be recomputed offline.

    f_true_fn : callable (n,D)->(n,D), analytic/known true drift.
    """
    T = xs_np.shape[0]
    step = max(1, T // STATES_MAX)
    pts = xs_np[::step]                              # (n, D) true states
    f_true = np.asarray(f_true_fn(pts))
    var_f = float(np.mean((f_true - f_true.mean(0, keepdims=True)) ** 2))

    f_pred_raw = np.asarray(f_eval_fn(pts))
    rel_raw = float(np.mean((f_pred_raw - f_true) ** 2)) / max(var_f, 1e-12)

    A, b = base.procrustes_align(np.asarray(mp['m'][0]), xs_np)
    pts_inf = (pts - b) @ np.linalg.inv(A).T          # true-frame -> inferred-frame
    f_pred_pc = np.asarray(f_eval_fn(pts_inf)) @ A.T   # map drift vectors back
    rel_pc = float(np.mean((f_pred_pc - f_true) ** 2)) / max(var_f, 1e-12)
    return dict(rel_mse=rel_pc, rel_mse_raw=rel_raw,
                nrmse=float(np.sqrt(rel_pc)), nrmse_raw=float(np.sqrt(rel_raw)),
                var_f=var_f, A=A, b=b,
                eval_pts=pts, f_true=f_true, f_pred_pc=f_pred_pc,
                f_pred_raw=f_pred_raw)


def run_cell(T, method, M, ls_init, seed, eps_grid=1e-3):
    """Fit one method on one T; return a flat dict of everything to save."""
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = make_data(T, seed)
    xs_np = np.asarray(xs)
    dt = float(np.asarray(t_grid[1] - t_grid[0]))

    if method == 'efgp':
        st = _fit_efgp_with_hist(lik, op, ip, t_grid, sigma, ls_init,
                                 eps_grid=eps_grid)
        f_eval_fn = lambda g: base.efgp_em.posterior_drift_mean(st['hist'], g)
        M_out = 0
    elif method == 'sp':
        n_per = int(round(math.sqrt(M)))
        # Isotropic SparseGP (single shared ℓ) — matches EFGP; avoids the ARD
        # lengthscale-vector wandering documented in the iso sweep.
        st = iso.fit_sparsegp_iso(lik, op, ip, t_grid, sigma, n_per, ls_init,
                                  xs_np)
        f_eval_fn = lambda g: base.eval_sp_drift(st, g)
        M_out = M
    else:
        raise ValueError(f"unknown method {method!r}")

    # PRIMARY drift metric: NRMSE at the trajectory states x_t (on-support).
    dstate = _drift_metrics_at_states(
        st['mp'], xs_np, lambda p: _f_true_batch(p, X_grid, alpha), f_eval_fn)
    # Reference: the canonical padded-bbox-grid metric (extrapolation-heavy).
    dgrid = base.compute_drift_metrics(st['mp'], xs_np, f_eval_fn=f_eval_fn,
                                       drift_fn=drift_fn)
    lat = base.latent_recovery_rmse(st['mp'], xs_np)

    return dict(
        status='ok', err='',
        T=T, method=method, M=M_out, ls_init=ls_init, seed=seed, dt=dt,
        eps_grid=eps_grid,
        wall=st['wall'],
        ls_final=st['ls'], var_final=st['var'],
        ls_traj=st['ls_traj'], var_traj=st['var_traj'],
        # primary = on-support (states) metric
        drift_rel_mse=dstate['rel_mse'], drift_rel_mse_raw=dstate['rel_mse_raw'],
        var_f=dstate['var_f'],
        # reference = bbox-grid metric (kept for comparison)
        drift_rel_mse_grid=dgrid['rel_mse'],
        drift_rel_mse_grid_raw=dgrid['rel_mse_raw'],
        # persisted evaluated arrays -> re-score any metric variant offline
        eval_pts=dstate['eval_pts'], f_true_states=dstate['f_true'],
        f_pred_states_pc=dstate['f_pred_pc'], f_pred_states_raw=dstate['f_pred_raw'],
        procrustes_A=dstate['A'], procrustes_b=dstate['b'],
        lat_pc=lat['pc'], lat_raw=lat['raw'],
        m_inf=np.asarray(st['mp']['m'][0]),
        S_diag=_s_diag(st['mp']),
        xs_true=xs_np,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, required=True)
    ap.add_argument('--method', choices=['efgp', 'sp'], required=True)
    ap.add_argument('--M', type=int, default=0,
                    help='inducing count for sp (e.g. 49, 100); ignored for efgp')
    ap.add_argument('--ls-init', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps-grid', type=float, default=1e-3,
                    help='EFGP spectral-grid truncation tol (finer=more modes)')
    ap.add_argument('--out-dir', default='demos/_bench_gpdrift_scaling_out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}{args.M if args.method == 'sp' else ''}"
    # keep canonical (default-eps) filenames stable; suffix only non-default eps
    eps_tag = "" if (args.method != 'efgp' or args.eps_grid == 1e-3) \
        else f"_eps{args.eps_grid:g}"
    out_path = out_dir / f"cell_T{args.T}_{tag}{eps_tag}_seed{args.seed}.npz"

    print(f"[scaling] T={args.T} method={args.method} M={args.M} "
          f"ls_init={args.ls_init} seed={args.seed} eps_grid={args.eps_grid} x64="
          f"{jax.config.read('jax_enable_x64')} dev={jax.devices()}", flush=True)

    t0 = time.perf_counter()
    try:
        res = run_cell(args.T, args.method, args.M, args.ls_init, args.seed,
                       eps_grid=args.eps_grid)
        print(f"[scaling] OK wall={res['wall']:.1f}s "
              f"drift_states={res['drift_rel_mse']:.4f} "
              f"drift_grid={res['drift_rel_mse_grid']:.4f} "
              f"lat_pc={res['lat_pc']:.4f} "
              f"l={res['ls_final']:.3f} var={res['var_final']:.3f}", flush=True)
    except Exception as e:  # noqa: BLE001 - record failure, don't lose the cell
        tb = traceback.format_exc()
        print(f"[scaling] FAILED after {time.perf_counter()-t0:.1f}s:\n{tb}",
              flush=True)
        res = dict(status='failed', err=f"{type(e).__name__}: {e}\n{tb}",
                   T=args.T, method=args.method,
                   M=args.M if args.method == 'sp' else 0,
                   ls_init=args.ls_init, seed=args.seed)

    np.savez(out_path, **res)
    print(f"[scaling] wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
