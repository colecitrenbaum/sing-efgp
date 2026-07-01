"""Why do EFGP and SparseGP drifts agree in dense data but differ where data
is sparse?  Separate the candidate causes by holding q(x) fixed.

Since the inferred latents are ~identical between the two methods, we take ONE
EFGP fit's q(x) and reconstruct BOTH drift posteriors from it:
  - EFGP:     random-feature / Fourier-mode basis (compute_mu_r_gmix)
  - SparseGP: inducing-point basis (SparseGP.update_dynamics_params)
each at (a) its OWN learned hypers and (b) MATCHED hypers.

This isolates:
  * hyperparameter effect  = EFGP@own vs EFGP@matched  (basis fixed)
  * basis/approx effect     = EFGP@matched vs SparseGP@matched (hypers fixed, q(x) fixed)
  * inference error         = would show as disagreement in DENSE data at
                              matched hypers (should be ~0 after the centering fix)

Difference maps are overlaid with data density and inducing-point locations,
and rms differences are reported split by dense vs sparse regions.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_vs_sparse_posterior.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature

OUT = b.OUT_DIR / "efgp_vs_sparse_posterior.png"

# Learned hypers from the fixed-centering bench run.
LS_E, VAR_E = 3.525, 0.286     # EFGP
LS_S, VAR_S = 3.243, 0.162     # SparseGP


def sparse_drift_builder(mp, lo, hi):
    """Return f(X; ls, var) for a SparseGP drift built from the given q(x)."""
    K_, T_, _ = mp['m'].shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    quad = GaussHermiteQuadrature(D=b.D, n_quad=5)
    zs = b._data_aware_zs(8, lo, hi)                    # same 8x8 grid as bench
    fn = SparseGP(zs=zs, kernel=RBF(latent_dim=b.D), expectation=quad)
    t_grid = jnp.arange(T_) * (b.DT * b.SUBSAMPLE_T)
    inp = jnp.zeros((K_, T_, 1)); Bz = jnp.zeros((b.D, 1))

    def build(ls, var):
        dp = dict(length_scales=jnp.full((b.D,), float(ls)),
                  output_scale=jnp.asarray(math.sqrt(var)))
        gp = fn.update_dynamics_params(jr.PRNGKey(0), t_grid, mp, trial_mask,
                                       dp, inp, Bz, b.SIGMA)
        fmean = lambda X: np.asarray(fn.get_posterior_f_mean(gp, dp, jnp.asarray(X)))
        fvar = lambda X: np.asarray(fn.get_posterior_f_var(gp, dp, jnp.asarray(X)))
        return fmean, fvar
    return build, np.asarray(zs)


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
    X_template = (jnp.linspace(float(lo.min()), float(hi.max()), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    print("  fitting EFGP once to get shared q(x) ...")
    mp, hist, _ = b.fit_efgp(ys, inputs, output_params, x0, t_grid, X_template, ls_init)
    m = np.asarray(mp['m'][0])
    print(f"  q(x) obtained; EFGP hypers ell={hist.lengthscale[-1]:.3f} var={hist.variance[-1]:.3f}")

    # Evaluation grid + data-density (sparse vs dense) mask.
    n = 46
    gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    P = np.stack([GX.ravel(), GY.ravel()], -1)
    Hd, _, _ = np.histogram2d(m[:, 0], m[:, 1], bins=[gx, gy])
    dens = np.zeros((n, n)); dens[:-1, :-1] = Hd            # per-cell counts
    dens_pts = dens.ravel()
    sparse_mask = dens_pts < 1.0                            # cells with ~no latents
    dense_mask = dens_pts >= 3.0

    # EFGP drift from the shared q(x) at own / matched hypers.
    fE = lambda ls, var: b.efgp_drift_field(mp, ls, var, t_grid, X_template, P)
    fE_own = fE(LS_E, VAR_E)
    fE_match = fE(LS_S, VAR_S)

    # SparseGP drift from the SAME q(x) at own / matched hypers.
    build_sp, zs = sparse_drift_builder(mp, lo, hi)
    fmean_own, fvar_own = build_sp(LS_S, VAR_S)
    fS_own = fmean_own(P)
    fmean_m, _ = build_sp(LS_E, VAR_E)   # SparseGP at EFGP hypers (unused for now)

    def rms(a, msk):
        d = np.linalg.norm(a, axis=-1)
        return float(np.sqrt(np.mean(d[msk] ** 2))) if msk.any() else float('nan')

    d_own = fE_own - fS_own                    # full difference (each own hypers)
    d_basis = fE_match - fS_own                # matched hypers -> basis + q(x)-only
    scale = float(np.median(np.linalg.norm(fS_own, axis=-1)) + 1e-9)
    print("\n  rms||f_EFGP - f_Sparse|| / median||f||  (relative):")
    print(f"    OWN hypers      dense={rms(d_own, dense_mask)/scale:.3f}  "
          f"sparse={rms(d_own, sparse_mask)/scale:.3f}")
    print(f"    MATCHED hypers  dense={rms(d_basis, dense_mask)/scale:.3f}  "
          f"sparse={rms(d_basis, sparse_mask)/scale:.3f}")
    print(f"    hyper-only (EFGP own vs matched) dense="
          f"{rms(fE_own - fE_match, dense_mask)/scale:.3f}  "
          f"sparse={rms(fE_own - fE_match, sparse_mask)/scale:.3f}")

    # ---- Figure: difference maps with density + inducing points ----
    def panel(ax, diff, title):
        dn = np.linalg.norm(diff, axis=-1).reshape(n, n)
        im = ax.pcolormesh(GX, GY, dn, cmap='magma', shading='auto')
        ax.contour(GX, GY, dens.reshape(n, n), levels=[1, 5, 15],
                   colors='cyan', linewidths=0.6, alpha=0.7)
        ax.scatter(zs[:, 0], zs[:, 1], s=10, c='lime', marker='x',
                   label='inducing pts')
        ax.plot(m[:, 0], m[:, 1], 'w-', lw=0.3, alpha=0.4)
        ax.set_title(title); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        return im

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    im0 = panel(axes[0], d_own, "|f_EFGP - f_Sparse|  (each OWN hypers)")
    im1 = panel(axes[1], d_basis, "|f_EFGP - f_Sparse|  (MATCHED hypers -> basis)")
    im2 = panel(axes[2], fE_own - fE_match, "EFGP own vs matched hypers (hyper-only)")
    for im, ax in zip([im0, im1, im2], axes):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].legend(loc='upper left', fontsize=7)
    fig.suptitle("EFGP vs SparseGP drift from the SAME q(x): where and why they differ "
                 "(cyan=data density, green x=inducing pts)", fontsize=11)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
