"""Why do EFGP and SparseGP posterior drift fields point DIFFERENTLY in
low-data-density regions?  Structural (random-feature vs inducing-point basis)
or hyperparameter (ell, sigma^2) difference?

Controlled 2x2: hold q(x) FIXED (use EFGP's converged latents mp_e) and the same
inputs + learned B, then evaluate BOTH drift representations at BOTH hyper
settings:

              theta_E (EFGP hypers)      theta_S (SparseGP-iso hypers)
  EFGP        efgp_drift_field(mp_e,.)   efgp_drift_field(mp_e,.)
  SparseGP    update_dyn(mp_e,.)+f_mean  update_dyn(mp_e,.)+f_mean

Only the drift BASIS (EFGP random features vs isotropic inducing-point GP) and
the hypers vary; q(x), inputs, B, sigma, and the eval grid are identical.  The
drift DIRECTION angle is scale-invariant, so it isolates shape from magnitude.

Decomposition (median angular disagreement, split by data density):
  structural@theta = angle(EFGP@theta,  SP@theta)     <- basis difference
  hyper_EFGP       = angle(EFGP@theta_E, EFGP@theta_S) <- hyper effect, EFGP
  hyper_SP         = angle(SP@theta_E,   SP@theta_S)   <- hyper effect, SP
  observed         = angle(EFGP@theta_E, SP@theta_S)   <- what the plots show

If structural >> hyper (esp. in low density) -> it's the basis.  If hyper
explains most of `observed` -> it's the hypers.

Run on swl1 (demos/probe_drift_structural_vs_hyper.sbatch); one EFGP fit + a few
forward evals, ~4 min.
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

import demos.bench_neural_efgp_vs_sparsegp as bench
from demos.bench_neural_inducing_sweep_iso_x64 import IsotropicRBF
from sing.sde import SparseGP
from sing.expectation import GaussHermiteQuadrature

D = bench.D
NUM_PER_DIM = 12                        # 144 inducing (matches the sweep asymptote)
OUT_DIR = _SING / "demos" / "_probe_drift_out"
OUT_DIR.mkdir(exist_ok=True)
ISO_NPZ = _SING / "demos" / "_bench_neural_inducing_sweep_iso_out" / "sweep.npz"


def _angle_deg(U, V, mag_floor):
    """Per-cell angle (deg) between drift vectors U,V; NaN where either is tiny."""
    nu = np.linalg.norm(U, axis=-1); nv = np.linalg.norm(V, axis=-1)
    cos = (U * V).sum(-1) / np.maximum(nu * nv, 1e-30)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[(nu < mag_floor) | (nv < mag_floor)] = np.nan
    return ang


def main():
    print(f"=== probe: EFGP vs SparseGP drift direction, structural vs hyper ===")
    print(f"    jax {jax.__version__}  devices={jax.devices()}", flush=True)

    norm = bench.load_neural_data()[::bench.SUBSAMPLE_T]
    n_t, n_n = norm.shape
    t_grid = jnp.arange(n_t) * (bench.DT * bench.SUBSAMPLE_T)
    ys = jnp.asarray(norm[None])
    inputs, o1, o2 = bench.build_inputs(n_t)
    output_params, x0 = bench.initialize_params_pca(D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_t, 64))[:, None]
                  * jnp.ones((1, D)))

    # --- EFGP fit once: gives the SHARED q(x)=mp_e, theta_E, and B_e ---
    print("  [EFGP] fitting (shared q(x))...", flush=True)
    mp_e, hist_e, wall_e = bench.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                          X_template, ls_init)
    ls_E, var_E = float(hist_e.lengthscale[-1]), float(hist_e.variance[-1])
    B_e = jnp.asarray(hist_e.input_effect)
    m_e = np.asarray(mp_e['m'][0])
    print(f"    theta_E: ell={ls_E:.3f} var={var_E:.3f}", flush=True)

    # theta_S from the isotropic sweep asymptote (M=144).
    if ISO_NPZ.exists():
        d = np.load(ISO_NPZ)
        ls_S = float(d['ls_sparsegp'][-1]); var_S = float(d['var_sparsegp'][-1])
    else:
        ls_S, var_S = 3.826, 0.232
    print(f"    theta_S: ell={ls_S:.3f} var={var_S:.3f}", flush=True)

    # --- SparseGP object (isotropic) sharing EFGP's q(x) ---
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = bench._data_aware_zs(NUM_PER_DIM, lo, hi)
    fn = SparseGP(zs=zs, kernel=IsotropicRBF(latent_dim=D), expectation=quad)
    zs_np = np.asarray(zs)
    trial_mask = jnp.ones((1, n_t), dtype=bool)

    def efgp_field(ls, var, X):
        return bench.efgp_drift_field(mp_e, ls, var, t_grid, X_template, X)

    def sp_field(ls, var, X):
        dp = dict(length_scale=jnp.asarray(float(ls)),
                  output_scale=jnp.asarray(math.sqrt(float(var))))
        gp_post = fn.update_dynamics_params(jr.PRNGKey(0), t_grid, mp_e,
                                            trial_mask, dp, inputs, B_e, bench.SIGMA)
        return np.asarray(fn.get_posterior_f_mean(gp_post, dp, jnp.asarray(X)))

    # --- dense grid for angle stats + density classification ---
    ng = 40
    gx = np.linspace(lo[0], hi[0], ng); gy = np.linspace(lo[1], hi[1], ng)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    P = np.stack([GX.ravel(), GY.ravel()], -1)                 # (ng*ng, 2)

    print("  evaluating 4 fields on the grid...", flush=True)
    F = {
        ('EFGP', 'E'): np.asarray(efgp_field(ls_E, var_E, P)),
        ('EFGP', 'S'): np.asarray(efgp_field(ls_S, var_S, P)),
        ('SP',   'E'): sp_field(ls_E, var_E, P),
        ('SP',   'S'): sp_field(ls_S, var_S, P),
    }

    # data density: distance from each grid cell to nearest latent point.
    dmin = np.sqrt(((P[:, None, :] - m_e[None, :, :]) ** 2).sum(-1)).min(1)  # (ng*ng,)
    mag_floor = 0.02 * float(np.median([np.linalg.norm(F[k], axis=-1).max()
                                        for k in F]))
    HI_D = dmin < 1.0          # high density: within ~1 unit of the trajectory
    LO_D = dmin > 3.0          # low density: far from any latent

    def med_angle(a, b, mask):
        ang = _angle_deg(F[a], F[b], mag_floor)
        sel = ang[mask & np.isfinite(ang)]
        return float(np.median(sel)) if sel.size else float('nan')

    pairs = {
        'structural@theta_E': (('EFGP', 'E'), ('SP', 'E')),
        'structural@theta_S': (('EFGP', 'S'), ('SP', 'S')),
        'hyper_EFGP (E vs S)': (('EFGP', 'E'), ('EFGP', 'S')),
        'hyper_SP   (E vs S)': (('SP', 'E'), ('SP', 'S')),
        'OBSERVED (EFGP@E vs SP@S)': (('EFGP', 'E'), ('SP', 'S')),
    }
    print("\n  === median drift-direction disagreement (degrees) ===")
    print(f"    {'comparison':<28s} {'all':>7s} {'high-dens':>10s} {'low-dens':>9s}")
    stats = {}
    for name, (a, b) in pairs.items():
        alld = med_angle(a, b, np.ones_like(HI_D))
        hid = med_angle(a, b, HI_D)
        lod = med_angle(a, b, LO_D)
        stats[name] = (alld, hid, lod)
        print(f"    {name:<28s} {alld:>7.1f} {hid:>10.1f} {lod:>9.1f}", flush=True)

    np.savez(OUT_DIR / "probe.npz",
             ls_E=ls_E, var_E=var_E, ls_S=ls_S, var_S=var_S,
             P=P, dmin=dmin, m_e=m_e, zs=zs_np, lo=lo, hi=hi,
             **{f"F_{k[0]}_{k[1]}": v for k, v in F.items()},
             stats_names=np.array(list(stats.keys())),
             stats_vals=np.array(list(stats.values())))

    # --- 2x2 quiver figure on a coarser grid, density background ---
    nq = 24
    qx = np.linspace(lo[0], hi[0], nq); qy = np.linspace(lo[1], hi[1], nq)
    QX, QY = np.meshgrid(qx, qy, indexing='ij')
    Q = np.stack([QX.ravel(), QY.ravel()], -1)
    dmin_q = np.sqrt(((Q[:, None, :] - m_e[None, :, :]) ** 2).sum(-1)).min(1)
    dens_q = np.exp(-0.5 * (dmin_q / 1.5) ** 2).reshape(nq, nq)
    Fq = {k: (np.asarray(efgp_field(*(  # noqa
             (ls_E, var_E) if k[1] == 'E' else (ls_S, var_S)), Q))
             if k[0] == 'EFGP' else
             sp_field(*((ls_E, var_E) if k[1] == 'E' else (ls_S, var_S)), Q))
          for k in F}

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    layout = [[('EFGP', 'E'), ('EFGP', 'S')], [('SP', 'E'), ('SP', 'S')]]
    ttl = {('EFGP', 'E'): f"EFGP @ theta_E (ell={ls_E:.2f}, var={var_E:.2f})  [its own]",
           ('EFGP', 'S'): f"EFGP @ theta_S (ell={ls_S:.2f}, var={var_S:.2f})",
           ('SP', 'E'): f"SparseGP @ theta_E (ell={ls_E:.2f}, var={var_E:.2f})",
           ('SP', 'S'): f"SparseGP @ theta_S (ell={ls_S:.2f}, var={var_S:.2f})  [its own]"}
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]; key = layout[i][j]
            ax.imshow(dens_q.T, origin='lower', extent=[lo[0], hi[0], lo[1], hi[1]],
                      cmap='Oranges', vmin=0, vmax=1, alpha=0.5, aspect='auto')
            v = Fq[key].reshape(nq, nq, 2)
            ax.quiver(QX, QY, v[..., 0], v[..., 1], angles='xy', color='black',
                      alpha=0.85)
            ax.plot(m_e[:, 0], m_e[:, 1], color='tab:blue', lw=0.6, alpha=0.5)
            ax.plot(zs_np[:, 0], zs_np[:, 1], 'r.', ms=3, alpha=0.5)
            ax.set_title(ttl[key], fontsize=10)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    fig.suptitle("Drift field: basis (rows) x hypers (cols), SHARED q(x)\n"
                 "orange=data density  blue=latents  red=inducing pts", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drift_2x2.png", dpi=130, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/{{drift_2x2.png, probe.npz}}", flush=True)


if __name__ == "__main__":
    main()
