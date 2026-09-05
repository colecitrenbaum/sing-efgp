"""
diag_qx_calibration.py

Is SparseGP's inferred q(x) posterior variance S inflated / mis-calibrated
relative to EFGP's, and are its mean trajectories over-smoothed?  This decides
whether the lengthscale inflation reflects (a) honest latent-uncertainty
propagation or (b) a mis-calibrated posterior.

For EFGP and SparseGP (M=25, 256) on the same well-specified GP-drift data,
report:
  - posterior std  = sqrt(mean_t diag(S_i))          (claimed uncertainty)
  - actual error   = raw RMSE(m - x_true)            (realised error)
  - calibration    = posterior_std / actual_error    (≈1 calibrated; ≫1 too wide)
  - velocity std   = std(m_{i+1}-m_i)  vs true       (mean-trajectory smoothness)
  - exact-evidence argmax ℓ on THIS method's means   (what the pseudo-velocities imply)

Run under Slurm (demos/diag_qx_calibration.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep

LS_INIT = 0.7
M_LIST = [25, 256]
LOG_LS = np.linspace(-2.0, 1.5, 21)
LOG_VAR = np.linspace(-2.5, 2.0, 21)


def report(name, mp, xs_np, sigma, t_grid):
    m = np.asarray(mp['m'][0])          # (T, D)
    S = np.asarray(mp['S'][0])          # (T, D, D)
    post_var = np.mean([np.trace(S[t]) / S.shape[-1] for t in range(S.shape[0])])
    post_std = math.sqrt(post_var)
    raw_err = float(np.sqrt(np.mean((m - xs_np) ** 2)))
    calib = post_std / max(raw_err, 1e-12)
    vel = np.diff(m, axis=0)
    vel_std = float(vel.std())
    vel_true = float(np.diff(xs_np, axis=0).std())
    # exact-evidence argmax on this method's means
    L = bench.gt_landscape(jnp.asarray(m), sigma, t_grid, LOG_LS, LOG_VAR)
    k = np.unravel_index(np.nanargmin(L), L.shape)
    ex_ls = math.exp(float(LOG_LS[k[0]]))

    print(f"\n  === {name} ===")
    print(f"    posterior std   = {post_std:.4f}   (claimed latent uncertainty)")
    print(f"    actual RMSE     = {raw_err:.4f}   (realised latent error)")
    print(f"    calibration     = {calib:.2f}     (≈1 calibrated; ≫1 over-wide S)")
    print(f"    velocity std    = {vel_std:.4f}   vs true {vel_true:.4f}  "
          f"(ratio {vel_std / vel_true:.3f}; <1 ⇒ means over-smoothed)")
    print(f"    exact-evid argmax ℓ on THESE means = {ex_ls:.3f}  "
          f"(pseudo-velocities imply this ℓ)", flush=True)
    return dict(post_std=post_std, raw_err=raw_err, calib=calib,
                vel_std=vel_std, vel_true=vel_true, ex_ls=ex_ls)


def main():
    print(f"diag_qx_calibration: M={M_LIST}  devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)
    print(f"  true velocity std = {float(np.diff(xs_np, axis=0).std()):.4f}",
          flush=True)

    print("  EFGP fit...", flush=True)
    e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
    print(f"    EFGP recovered ℓ={e['ls']:.3f}", flush=True)
    report("EFGP", e['mp'], xs_np, sigma, t_grid)

    for M in M_LIST:
        n_per = int(round(math.sqrt(M)))
        print(f"\n  SparseGP M={M} fit...", flush=True)
        s = bench.fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, LS_INIT,
                               xs_np)
        print(f"    SparseGP recovered ℓ={s['ls']:.3f}", flush=True)
        report(f"SparseGP M={M}", s['mp'], xs_np, sigma, t_grid)

    print("\n  READING: if SparseGP calibration ≫ EFGP's (and ≫1), S is "
          "over-wide (mis-calibrated) ⇒ ℓ inflation is an artifact.  If "
          "calibration ≈ EFGP's but velocity-std is lower, the means are "
          "genuinely smoother ⇒ ℓ inflation is 'honest' but comes from an "
          "over-smoothed posterior mean, not S per se.", flush=True)


if __name__ == '__main__':
    main()
