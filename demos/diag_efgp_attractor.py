"""Diagnose the EFGP 'attractor off the data' shift on the neural fit.

Hypothesis: EFGP's spectral (Fourier/NUFFT) drift is PERIODIC over the grid
support box.  The bench set that box to hug the data with no padding, so the
latents touch the boundary and the drift wraps/aliases there -> the slow
region is displaced from the data.  SparseGP (inducing points) has no such
periodicity, so it isn't shifted.

Test: fit EFGP once, then rebuild the posterior drift field from the SAME
converged q(x) on (a) the tight bench grid and (b) a grid padded several ell
beyond the data.  If the slow region snaps onto the data under padding, the
shift is an edge/periodicity artifact (fix = pad the support box, and re-fit
with padding so q(x) is consistent).

Also reports ||f(m_t)|| at the actual latents under each grid.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_attractor.py
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

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca

OUT = b.OUT_DIR / "efgp_attractor_diag.png"


def drift_field(mp, ls, var, t_grid, X_template, X_eval):
    """Posterior EFGP drift mean at X_eval, q(f) rebuilt on the grid that
    spectral_grid_se makes from X_template (so padding X_template = padding
    the periodic support box)."""
    ms = jnp.asarray(mp['m']); Ss = jnp.asarray(mp['S']); SSs = jnp.asarray(mp['SS'])
    trial_mask = jnp.ones(ms.shape[:2], dtype=bool)
    del_t = t_grid[1:] - t_grid[:-1]
    grid = jp.spectral_grid_se(ls, var, X_template, eps=1e-2)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=b.SIGMA ** 2, D_lat=b.D, D_out=b.D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval)[None], D_lat=b.D, D_out=b.D)
    return np.asarray(Ef[0]), (float(grid.xcen[0]), float(grid.xcen[1])), grid.M


def speed_grid(field_fn, lo, hi, n=80):
    gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    F = field_fn(np.stack([GX.ravel(), GY.ravel()], -1))
    sp = np.linalg.norm(F, axis=-1).reshape(n, n)
    return GX, GY, sp


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, onset1, onset2 = b.build_inputs(n_timesteps)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0
    hi = xs_pca.max(0) + 1.0
    extent = float(np.max(hi - lo))
    ls_init = extent / 8.0
    # Same tight template the bench used.
    X_tmpl_tight = (jnp.linspace(lo.min(), hi.max(), max(n_timesteps, 64))[:, None]
                    * jnp.ones((1, b.D)))

    print("  fitting EFGP (bench settings) for the diagnostic ...")
    mp, hist, wall = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                X_tmpl_tight, ls_init)
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    m = np.asarray(mp['m'][0])
    print(f"  ell={ls:.3f} var={var:.3f} wall={wall:.0f}s")

    # Padded support box: extend ~3*ell beyond the data on every side.
    pad = 3.0 * ls
    lo_p = lo - pad; hi_p = hi + pad
    X_tmpl_pad = (jnp.linspace(float(lo_p.min()), float(hi_p.max()),
                               max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    f_tight = lambda X: drift_field(mp, ls, var, t_grid, X_tmpl_tight, X)[0]
    f_pad = lambda X: drift_field(mp, ls, var, t_grid, X_tmpl_pad, X)[0]

    _, cen_t, M_t = drift_field(mp, ls, var, t_grid, X_tmpl_tight, m[:1])
    _, cen_p, M_p = drift_field(mp, ls, var, t_grid, X_tmpl_pad, m[:1])
    print(f"  tight grid: xcen={np.round(cen_t,2)}  M={M_t}")
    print(f"  padded grid: xcen={np.round(cen_p,2)}  M={M_p}")

    # ||f(m_t)|| at the actual latents under each grid.
    f_at_m_tight = np.linalg.norm(f_tight(m), axis=-1)
    f_at_m_pad = np.linalg.norm(f_pad(m), axis=-1)
    # Pseudo-velocity for a self-consistency reference (autonomous, B*v removed).
    Bv = (np.asarray(hist.input_effect) @ np.asarray(inputs[0]).T).T   # (T, D)
    u = (m[1:] - m[:-1]) / float(t_grid[1] - t_grid[0])               # (T-1, D)
    u_auto = u - Bv[:-1]
    resid_tight = u_auto - f_tight(m)[:-1]
    resid_pad = u_auto - f_pad(m)[:-1]
    print(f"\n  ||f(m_t)|| at latents   tight: median={np.median(f_at_m_tight):.3f} "
          f"mean={f_at_m_tight.mean():.3f}")
    print(f"  ||f(m_t)|| at latents   padded: median={np.median(f_at_m_pad):.3f} "
          f"mean={f_at_m_pad.mean():.3f}")
    print(f"  mean autonomous residual (u-Bv-f)  tight={np.round(resid_tight.mean(0),3)} "
          f"||.||={np.linalg.norm(resid_tight.mean(0)):.3f}")
    print(f"  mean autonomous residual (u-Bv-f)  padded={np.round(resid_pad.mean(0),3)} "
          f"||.||={np.linalg.norm(resid_pad.mean(0)):.3f}")

    # ---- Plot: speed valleys (continuous, log) + time-colored latents ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True, sharey=True)
    tt = np.linspace(0, 1, len(m))
    for ax, ffn, name in zip(axes, [f_tight, f_pad],
                             [f"tight grid (bench)\nxcen={np.round(cen_t,1)}",
                              f"padded grid (+3 ell)\nxcen={np.round(cen_p,1)}"]):
        GX, GY, sp = speed_grid(ffn, lo, hi, n=80)
        pc = ax.pcolormesh(GX, GY, np.log10(sp + 1e-6), cmap='viridis',
                           shading='auto')
        # mark the global drift minimum (the inferred fixed point / attractor)
        amin = np.unravel_index(np.argmin(sp), sp.shape)
        ax.scatter(GX[amin], GY[amin], marker='*', s=220, c='red',
                   edgecolor='white', zorder=6, label='min ||f||')
        sc = ax.scatter(m[:, 0], m[:, 1], c=tt, cmap='autumn', s=6, alpha=0.8,
                        zorder=5)
        ax.scatter(*m.mean(0), marker='X', s=140, c='cyan', edgecolor='black',
                   zorder=7, label='data centroid')
        ax.set_title(name); ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.legend(loc='upper left', fontsize=8)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, label='log10 ||f||')
    fig.suptitle("EFGP drift speed valley vs latents (autumn=time):  "
                 "does padding the periodic grid move the slow region onto the data?",
                 fontsize=11)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
