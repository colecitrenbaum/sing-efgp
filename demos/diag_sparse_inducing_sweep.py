"""Does SparseGP's sigma_f^2 climb toward EFGP's as inducing points increase?

Tests the "sparse variational GP under-estimates signal variance" explanation
for the EFGP (sigma_f^2=0.29) vs SparseGP (sigma_f^2=0.16) gap.  Fits SparseGP
on the neural data at increasing inducing-point counts (8x8, 12x12, 16x16) and
records the converged (ell, sigma_f^2).  Prediction: sigma_f^2 rises toward
EFGP's ~0.29 as the inducing set grows (and the sparse-region drift gap shrinks).

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_sparse_inducing_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca

OUT = b.OUT_DIR / "sparse_inducing_sweep.png"
EFGP_VAR = 0.285      # EFGP converged sigma_f^2 (100-iter run)
EFGP_LS = 3.55        # EFGP converged ell
NUM_PER_DIM = [8, 12, 16]     # -> 64, 144, 256 inducing points


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, _, _ = b.build_inputs(n_timesteps)
    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0

    rows = []
    for npd in NUM_PER_DIM:
        n_ind = npd * npd
        print(f"\n=== SparseGP with {npd}x{npd} = {n_ind} inducing points ===",
              flush=True)
        (mp, fn, dp, gp_post, B, elbos,
         ls_hist, var_hist, wall) = b.fit_sparsegp(
            ys, inputs, output_params, x0, t_grid, lo, hi, ls_init,
            num_per_dim=npd)
        ls_f, var_f = ls_hist[-1], var_hist[-1]
        rows.append((n_ind, ls_f, var_f, float(elbos[-1]), wall))
        print(f"  -> {n_ind} inducing: ell={ls_f:.3f}  sigma_f^2={var_f:.3f}  "
              f"ELBO={float(elbos[-1]):.1f}  wall={wall:.0f}s", flush=True)

    print("\n  === summary ===")
    print(f"  {'n_ind':>6} {'ell':>8} {'sigma_f^2':>10} {'ELBO':>12} {'wall_s':>8}")
    for n_ind, ls_f, var_f, elbo, wall in rows:
        print(f"  {n_ind:>6} {ls_f:>8.3f} {var_f:>10.3f} {elbo:>12.1f} {wall:>8.0f}")
    print(f"  (EFGP reference: ell~{EFGP_LS}, sigma_f^2~{EFGP_VAR})")
    np.savez(b.OUT_DIR / "sparse_inducing_sweep.npz",
             n_ind=np.array([r[0] for r in rows]),
             ls=np.array([r[1] for r in rows]),
             var=np.array([r[2] for r in rows]),
             elbo=np.array([r[3] for r in rows]),
             wall=np.array([r[4] for r in rows]),
             efgp_var=EFGP_VAR, efgp_ls=EFGP_LS)

    n_ind = np.array([r[0] for r in rows])
    var_f = np.array([r[2] for r in rows])
    ls_f = np.array([r[1] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(n_ind, var_f, '-o', color='tab:red', label='SparseGP')
    ax[0].axhline(EFGP_VAR, ls='--', color='tab:blue', label=f'EFGP ~{EFGP_VAR}')
    ax[0].set_xlabel("# inducing points"); ax[0].set_ylabel("sigma_f^2")
    ax[0].set_title("signal variance vs inducing count"); ax[0].legend()
    ax[1].plot(n_ind, ls_f, '-o', color='tab:red', label='SparseGP')
    ax[1].axhline(EFGP_LS, ls='--', color='tab:blue', label=f'EFGP ~{EFGP_LS}')
    ax[1].set_xlabel("# inducing points"); ax[1].set_ylabel("ell")
    ax[1].set_title("lengthscale vs inducing count"); ax[1].legend()
    fig.suptitle("SparseGP hypers vs inducing-point count (does sigma_f^2 -> EFGP as rank grows?)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
