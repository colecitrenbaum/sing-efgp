"""Translation-equivariance test for the EFGP drift pipeline.

The posterior drift must be translation-equivariant: with the SAME q(x),
shifting every latent position by a constant c must shift the field by c and
leave its VALUES unchanged, i.e.

    f_shifted(x + c) == f_original(x)   for all x.

EFGP represents f in a Fourier basis centered at grid.xcen.  If the centering
phase used to BUILD mu_r (compute_mu_r_gmix_jax) disagrees with the one used
to EVALUATE the drift (drift_moments_jax / nufft2), the field will NOT be
translation-equivariant -- invisible when xcen ~ 0 (all synthetic benches)
but a real spatial shift when the data is off-origin (xcen=[1.88,1.88] here).

This fits EFGP once, then rebuilds q(f) from the SAME q(x) shifted by c and
compares the fields.  max|f_shift(x+c) - f(x)| ~ 0  => equivariant (no bug).
A large discrepancy localizes a centering/phase bug.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_translation.py
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

import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, _, _ = b.build_inputs(n_timesteps)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    a, bb = float(lo.min()), float(hi.max())
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(a, bb, max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    print("  fitting EFGP once (inputs on) ...")
    mp, hist, _ = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                             X_template, ls_init)
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    xcen = (a + bb) / 2.0
    print(f"  ell={ls:.3f} var={var:.3f}  grid xcen~[{xcen:.3f},{xcen:.3f}]")

    # Evaluation grid over the data bbox.
    n = 40
    gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    G = np.stack([GX.ravel(), GY.ravel()], -1)

    # (1) Original field.
    f1 = b.efgp_drift_field(mp, ls, var, t_grid, X_template, G)

    # (2) Shift EVERYTHING by c so the grid recenters at the origin.
    c = -np.array([xcen, xcen])
    mp_s = {**mp, 'm': mp['m'] + jnp.asarray(c)}
    X_template_s = X_template + jnp.asarray(c)
    f2 = b.efgp_drift_field(mp_s, ls, var, t_grid, X_template_s, G + c)

    # Equivariance: f2(x + c) should equal f1(x).
    d = f2 - f1
    print(f"\n  shift c = {np.round(c,3)} (recenters grid to origin)")
    print(f"  max |f_shift(x+c) - f(x)| = {np.abs(d).max():.4f}")
    print(f"  mean (f_shift - f)        = {np.round(d.mean(0),4)}  "
          f"(||{np.linalg.norm(d.mean(0)):.4f}||)")
    print(f"  median ||f||              = {np.median(np.linalg.norm(f1,axis=-1)):.3f}")
    print(f"  mean f over grid (orig)   = {np.round(f1.mean(0),3)}")
    print(f"  mean f over grid (shift)  = {np.round(f2.mean(0),3)}")
    if np.abs(d).max() < 1e-2 * max(np.median(np.linalg.norm(f1, axis=-1)), 1e-3):
        print("\n  => EQUIVARIANT: no centering/phase bug; offset is methodological.")
    else:
        print("\n  => NOT equivariant: centering/phase bug in the EFGP drift pipeline.")


if __name__ == "__main__":
    main()
