"""Pin down the EFGP 'drift offset' to the plotting reconstruction wiring.

Fits EFGP once on the neural data, then evaluates the posterior drift two ways:

  (A) EM-ACTUAL  : history.final_grid + history.final_mu_r via
                   em.posterior_drift_mean  -> the q(f) the EM converged to.
  (B) RECONSTRUCT: bench.efgp_drift_field   -> rebuilds q(f) on a freshly
                   made spectral grid (different M) -> the old plot path.

For a recurrent trajectory the mean pseudo-velocity ~ 0, so a correctly wired
drift must have mean f over the data ~ 0 and a small autonomous residual
(u - Bv - f).  If (A) satisfies that but (B) shows a systematic mean drift,
the 'offset' was a reconstruction-grid artifact in the plot, NOT the fit.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_wiring.py
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

import sing.efgp_em as em
import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca

OUT = b.OUT_DIR / "efgp_wiring_diag.png"


def _report(name, m, fvals, u_auto):
    fmean = fvals.mean(0)
    fmag = np.linalg.norm(fvals, axis=-1)
    resid = (u_auto - fvals[:-1]).mean(0)
    print(f"  [{name}]  mean f = {np.round(fmean,3)}  (||{np.linalg.norm(fmean):.3f}||)"
          f"   median||f(m)|| = {np.median(fmag):.3f}"
          f"   mean autonomous residual = {np.round(resid,3)} "
          f"(||{np.linalg.norm(resid):.3f}||)")
    return fmean


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, onset1, onset2 = b.build_inputs(n_timesteps)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    print("  fitting EFGP (inputs on) ...")
    mp, hist, wall = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                X_template, ls_init)
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    m = np.asarray(mp['m'][0])
    print(f"  ell={ls:.3f} var={var:.3f} wall={wall:.0f}s  "
          f"EM grid M={hist.final_grid.M}, mtot={hist.final_grid.mtot_per_dim}")

    dt = float(t_grid[1] - t_grid[0])
    Bv = (np.asarray(hist.input_effect) @ np.asarray(inputs[0]).T).T
    u_auto = (m[1:] - m[:-1]) / dt - Bv[:-1]
    print(f"  data net velocity (mean u) = {np.round(((m[-1]-m[0])/(len(m)*dt)),3)} "
          f"-> ~0 means recurrent\n")

    # (A) EM-actual drift
    f_A = em.posterior_drift_mean(hist, m)
    fmean_A = _report("EM-actual (final_grid M=%d)" % hist.final_grid.M, m, f_A, u_auto)

    # (B) old reconstruction path
    f_B = b.efgp_drift_field(mp, ls, var, t_grid, X_template, m)
    fmean_B = _report("reconstruction (efgp_drift_field)", m, f_B, u_auto)

    print(f"\n  => offset injected by reconstruction = mean f_B - mean f_A = "
          f"{np.round(fmean_B - fmean_A,3)}")

    # ---- Plot: speed valley under each path + latents ----
    def speed(ffn, n=80):
        gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        F = ffn(np.stack([GX.ravel(), GY.ravel()], -1))
        return GX, GY, np.linalg.norm(F, axis=-1).reshape(n, n)

    fA = lambda X: em.posterior_drift_mean(hist, X)
    fB = lambda X: b.efgp_drift_field(mp, ls, var, t_grid, X_template, X)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True, sharey=True)
    for ax, ffn, name in zip(axes, [fA, fB],
                             ["(A) EM-actual q(f)  [fixed wiring]",
                              "(B) reconstruction  [old plot path]"]):
        GX, GY, sp = speed(ffn)
        pc = ax.pcolormesh(GX, GY, np.log10(sp + 1e-6), cmap='viridis', shading='auto')
        amin = np.unravel_index(np.argmin(sp), sp.shape)
        ax.scatter(GX[amin], GY[amin], marker='*', s=220, c='red',
                   edgecolor='white', zorder=6, label='min ||f||')
        ax.scatter(m[:, 0], m[:, 1], c=np.linspace(0, 1, len(m)), cmap='autumn',
                   s=6, alpha=0.8, zorder=5)
        ax.scatter(*m.mean(0), marker='X', s=140, c='cyan', edgecolor='black',
                   zorder=7, label='data centroid')
        ax.set_title(name); ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.legend(loc='upper left', fontsize=8)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, label='log10 ||f||')
    fig.suptitle("EFGP drift: EM-actual q(f) vs reconstruction (autumn=time)",
                 fontsize=11)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
