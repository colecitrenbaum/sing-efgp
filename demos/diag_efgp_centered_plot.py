"""Confirm the gmix centering bug visually: off-center grid vs centered grid.

The bug shifts the EFGP drift field by grid.xcen.  The bench builds
X_template = linspace(lo.min, hi.max) -> xcen ~ [1.88, 1.88] (off origin), so
the attractor is displaced.  If we instead build a SYMMETRIC X_template
(xcen ~ 0), the bug's xcen-shift vanishes and the slow region should land on
the data.

Fits EFGP twice (same data, inputs on) with:
  (A) off-center grid  (bench default)
  (B) centered  grid   (symmetric X_template, xcen ~ 0)
and plots both attractor panels side by side.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_centered_plot.py
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

OUT = b.OUT_DIR / "efgp_centered_vs_offcenter.png"


def fit_and_eval(ys, inputs, output_params, x0, t_grid, X_template, ls_init,
                 lo, hi):
    mp, hist, _ = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                             X_template, ls_init)
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    m = np.asarray(mp['m'][0])
    fn = lambda X: b.efgp_drift_field(mp, ls, var, t_grid, X_template, X)
    xcen = float(0.5 * (np.asarray(X_template).min() + np.asarray(X_template).max()))
    return m, hist.input_effect, fn, ls, var, xcen


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, onset1, onset2 = b.build_inputs(n_timesteps)
    onsets = (onset1, onset2)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0

    # (A) off-center grid (bench default): xcen ~ [1.88, 1.88]
    X_off = (jnp.linspace(float(lo.min()), float(hi.max()), max(n_timesteps, 64))[:, None]
             * jnp.ones((1, b.D)))
    # (B) centered grid: symmetric about 0 covering the data -> xcen ~ 0
    R = float(np.max(np.abs([lo, hi]))) + 1.0
    X_cen = (jnp.linspace(-R, R, max(n_timesteps, 64))[:, None]
             * jnp.ones((1, b.D)))

    print("  fitting EFGP with OFF-CENTER grid ...")
    mA, BA, fA, lsA, vA, xcA = fit_and_eval(ys, inputs, output_params, x0,
                                            t_grid, X_off, ls_init, lo, hi)
    print(f"    xcen={xcA:.2f}  ell={lsA:.2f} var={vA:.3f}")
    print("  fitting EFGP with CENTERED grid ...")
    mB, BB, fB, lsB, vB, xcB = fit_and_eval(ys, inputs, output_params, x0,
                                            t_grid, X_cen, ls_init, lo, hi)
    print(f"    xcen={xcB:.2f}  ell={lsB:.2f} var={vB:.3f}")

    # shared slow-point eps from off-center field median speed
    gx = np.linspace(lo[0], hi[0], 30); gy = np.linspace(lo[1], hi[1], 30)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    eps = max(0.1 * float(np.median(np.linalg.norm(fB(pts), axis=-1))), 1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    im0 = b.attractor_panel(axes[0], fA, mA, BA, onsets, lo, hi, eps,
                            title=f"OFF-CENTER grid (xcen={xcA:.2f})\n"
                                  f"buggy: attractor shifted by ~xcen")
    im1 = b.attractor_panel(axes[1], fB, mB, BB, onsets, lo, hi, eps,
                            title=f"CENTERED grid (xcen={xcB:.2f})\n"
                                  f"shift removed: attractor on data?")
    fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04, label='slow-point proxy')
    fig.suptitle("EFGP neural attractor: gmix centering bug shifts field by xcen",
                 fontsize=11)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
