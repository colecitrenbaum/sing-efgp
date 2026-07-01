"""Standalone EFGP-SING (RBF) 2D dynamics plot on the neural data, in the
style of ``neural_data_gpslds.ipynb`` cell 37, for direct visual comparison.

Re-fits ONLY the EFGP path (reusing bench_neural_efgp_vs_sparsegp) and writes
a single-panel line-attractor figure: posterior drift quiver (black), inferred
latent trajectory (blue), speed-based slow-point map (purple), and learned
input-effect arrows (orange) at the two intruder onsets.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/plot_efgp_neural_dynamics.py
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

OUT = b.OUT_DIR / "efgp_line_attractor.png"


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps, n_neurons = norm_neural.shape
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, onset1, onset2 = b.build_inputs(n_timesteps)
    onsets = (onset1, onset2)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0
    hi = xs_pca.max(0) + 1.0
    extent = float(np.max(hi - lo))
    ls_init = extent / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    print("  fitting EFGP-SING (RBF, inputs + learned B) for the plot...")
    mp, hist, wall = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                               X_template, ls_init)
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    B = hist.input_effect
    m = np.asarray(mp['m'][0])
    efgp_fn = lambda X: b.efgp_drift_field(mp, ls, var, t_grid, X_template, X)

    # Slow-point speed scale: 10% of median drift speed on a coarse grid.
    gx = np.linspace(lo[0], hi[0], 30); gy = np.linspace(lo[1], hi[1], 30)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    sp_scale = float(np.median(np.linalg.norm(
        efgp_fn(np.stack([GX.ravel(), GY.ravel()], -1)), axis=-1)))
    eps_slow = max(0.1 * sp_scale, 1e-3)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = b.attractor_panel(ax, efgp_fn, m, B, onsets, lo, hi, eps_slow,
                           title=f"EFGP-SING fit (RBF)\nell={ls:.2f}, "
                                 f"var={var:.2f}, {wall:.0f}s")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("slow-point proxy  (||f|| ~ 0)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {OUT}")
    print(f"  ell={ls:.3f}  var={var:.3f}  B=\n{np.asarray(B).round(3)}")


if __name__ == "__main__":
    main()
