"""
diag_drift_metric.py  --  is drift rel-MSE ~2 a metric artifact or real?

For a given T/seed, on CPU:
  1. Rebuild the true drift field f_true (deterministic from seeds).
  2. Rebuild the eval grid exactly as compute_drift_metrics does (14x14 over the
     trajectory bbox + 0.4 pad).
  3. Compute the ORACLE drift = dense GP regression of pseudo-velocities
     v_t=(x_{t+1}-x_t)/dt on the TRUE states x_t with the TRUE kernel
     (l_true, var_true) + noise var_true_sde^2/dt, predicted on the grid.
     This is the best any GP method could do from this data.
  4. Report rel-MSE of the oracle on (a) the bbox grid and (b) at the trajectory
     points x_t themselves, plus grid coverage stats.

If oracle rel-MSE on the grid is also ~1-2 -> the bbox-grid metric is
extrapolation-dominated (the trajectory doesn't cover the padded bbox), and the
method numbers are fine *relatively* but the absolute metric is misleading.
If oracle rel-MSE << 1 -> the methods are underperforming (real problem).

  JAX_PLATFORMS=cpu python demos/diag_drift_metric.py --T 1000 --seed 0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import demos.bench_gpdrift_scaling as run   # make_data(T, seed), base
base = run.base


def oracle_drift_pred(xs_np, dt, X_te, ls, var, sigma_sde):
    """Posterior mean of a dense RBF-GP regressing v_t on x_t, at X_te."""
    Xin = xs_np[:-1]                                  # (n, D)
    V = (xs_np[1:] - xs_np[:-1]) / dt                 # (n, D)
    n = Xin.shape[0]
    def rbf(A, B):
        sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return var * np.exp(-0.5 * sq / ls ** 2)
    K = rbf(Xin, Xin) + (sigma_sde ** 2 / dt) * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, V))   # (n, D)
    Kte = rbf(X_te, Xin)                                   # (m, n)
    return Kte @ alpha                                     # (m, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = run.make_data(
        args.T, args.seed)
    xs_np = np.asarray(xs)
    dt = float(np.asarray(t_grid[1] - t_grid[0]))
    import jax.numpy as jnp

    # eval grid exactly like compute_drift_metrics
    lo = xs_np.min(0) - 0.4; hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14); g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid = np.stack([GX.ravel(), GY.ravel()], -1)          # (196, D)

    f_true_grid = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.)) for p in grid])
    var_f = float(np.mean((f_true_grid - f_true_grid.mean(0, keepdims=True)) ** 2))

    # oracle on grid
    f_or_grid = oracle_drift_pred(xs_np, dt, grid, base.LS_TRUE, base.VAR_TRUE,
                                  base.SIGMA_SDE)
    rel_grid = np.mean((f_or_grid - f_true_grid) ** 2) / var_f

    # oracle at trajectory points (subsample for speed at large T)
    step = max(1, xs_np.shape[0] // 2000)
    Xtr = xs_np[::step]
    f_true_tr = np.array([np.asarray(drift_fn(jnp.asarray(p), 0.)) for p in Xtr])
    var_f_tr = float(np.mean((f_true_tr - f_true_tr.mean(0, keepdims=True)) ** 2))
    f_or_tr = oracle_drift_pred(xs_np, dt, Xtr, base.LS_TRUE, base.VAR_TRUE,
                                base.SIGMA_SDE)
    rel_tr = np.mean((f_or_tr - f_true_tr) ** 2) / var_f_tr

    # coverage: min distance from each grid cell to the trajectory
    d2 = ((grid[:, None, :] - xs_np[None, ::step, :]) ** 2).sum(-1)
    mind = np.sqrt(d2.min(1))
    frac_far = float(np.mean(mind > base.LS_TRUE))    # cells > 1 lengthscale away
    # MSE split near vs far
    se = ((f_or_grid - f_true_grid) ** 2).sum(1)
    near = mind <= base.LS_TRUE
    mse_near = float(se[near].mean()) if near.any() else float('nan')
    mse_far = float(se[~near].mean()) if (~near).any() else float('nan')

    print(f"T={args.T} seed={args.seed} dt={dt:.4f}  var_f(grid)={var_f:.4f}")
    print(f"  ORACLE rel-MSE on bbox grid (196 pts) : {rel_grid:.4f}")
    print(f"  ORACLE rel-MSE at trajectory points   : {rel_tr:.4f}")
    print(f"  grid cells > 1 lengthscale from traj  : {frac_far*100:.1f}%")
    print(f"  per-cell SE  near(<=l): {mse_near:.4f}   far(>l): {mse_far:.4f}")
    print(f"  traj bbox: x0[{xs_np[:,0].min():.2f},{xs_np[:,0].max():.2f}] "
          f"x1[{xs_np[:,1].min():.2f},{xs_np[:,1].max():.2f}]")


if __name__ == '__main__':
    main()
