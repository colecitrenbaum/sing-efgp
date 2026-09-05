"""
efgp_long_3p0.py

From the ls_init=3.0 start, EFGP landed at ℓ=0.575 (BELOW the oracle MLE
ℓ=0.779) after 50 EM iters — an overshoot past the optimum.  Question: does it
turn back UP toward the optimum if run longer?

Runs ONLY EFGP from ls_init=3.0 with N_EM=150: the first 50 iters use the exact
same ρ ramp linspace(0.05,0.7,50) as the sweep (so they reproduce the original
run), then ρ is held at 0.7 for 100 more iters.  Captures the full ℓ/σ²
trajectory and overlays it on the GT oracle log-marginal contours.

Out: demos/_bench_gpdrift_inducing_sweep_iso_out/efgp_long_3p0.png  + .npz
Run under Slurm (efgp_long_3p0.sbatch), NOT the login node.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)

import math, sys, time
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import demos.bench_gpdrift_x64 as bench
import sing.efgp_em as efgp_em

LS_INIT = 3.0
N_RAMP = bench.N_EM          # 50 — same ramp as the sweep
N_HOLD = 100                 # extra iters held at ρ=0.7
N_EM = N_RAMP + N_HOLD
OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print(f"efgp_long_3p0: N_EM={N_EM} (ramp {N_RAMP} + hold {N_HOLD}) "
          f"ls_init={LS_INIT}  devices={jax.devices()}", flush=True)

    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)

    LOG_LS = np.linspace(bench.LOG_LS_RANGE[0], bench.LOG_LS_RANGE[1], bench.N_GRID)
    LOG_VAR = np.linspace(bench.LOG_VAR_RANGE[0], bench.LOG_VAR_RANGE[1], bench.N_GRID)
    t0 = time.perf_counter()
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # First N_RAMP iters identical to the sweep, then hold ρ=0.7.
    rho_sched = jnp.concatenate([jnp.linspace(0.05, 0.7, N_RAMP),
                                 jnp.full((N_HOLD,), 0.7)])
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=bench.D,
        lengthscale=LS_INIT, variance=bench.VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=bench.N_M_INNER, mstep_lr=bench.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8, verbose=False)
    wall = time.perf_counter() - t0

    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([bench.VAR_INIT] + list(hist.variance))
    print(f"  done in {wall:.1f}s  final ℓ={ls_traj[-1]:.4f}, σ²={var_traj[-1]:.4f}",
          flush=True)

    # trajectory print: iter, ℓ, σ² at a few checkpoints
    print("  iter    ℓ        σ²", flush=True)
    for it in [0, 10, 25, 50, 75, 100, 125, N_EM]:
        print(f"  {it:>4d}  {ls_traj[it]:.4f}   {var_traj[it]:.4f}", flush=True)

    np.savez(OUT_DIR / 'efgp_long_3p0.npz',
             ls_traj=ls_traj, var_traj=var_traj, wall=wall,
             gb_ll=gb_ll, gb_lv=gb_lv, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR,
             L_gt=L_gt, n_ramp=N_RAMP, n_hold=N_HOLD,
             ls_true=bench.LS_TRUE, var_true=bench.VAR_TRUE)

    # ---- plot: trajectory over GT contours ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    lo, hi = float(finite.min()), float(finite.max())
    levels = list(np.logspace(np.log10(max(lo, hi / 1000)), np.log10(hi), 10))
    ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                cmap='viridis_r', extend='max')
    cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                    colors='k', linewidths=0.4)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

    lp = np.log(ls_traj); vp = np.log(var_traj)
    # colour by iter so the eye tracks direction of travel
    sc = ax.scatter(lp, vp, c=np.arange(len(lp)), cmap='autumn_r', s=18,
                    zorder=6)
    ax.plot(lp, vp, '-', color='0.3', lw=1.0, zorder=5)
    ax.scatter([lp[N_RAMP]], [vp[N_RAMP]], marker='D', s=90, facecolor='none',
               edgecolor='red', lw=2, zorder=8,
               label=f'iter {N_RAMP} (end of original run) '
                     f'[ℓ={ls_traj[N_RAMP]:.2f}, σ²={var_traj[N_RAMP]:.2f}]')
    ax.scatter([lp[-1]], [vp[-1]], marker='o', s=110, color='red',
               edgecolor='k', zorder=9,
               label=f'iter {N_EM} (final) [ℓ={ls_traj[-1]:.2f}, σ²={var_traj[-1]:.2f}]')
    ax.scatter([math.log(LS_INIT)], [math.log(bench.VAR_INIT)], marker='+',
               s=200, color='black', zorder=9, label='init')
    ax.scatter([gb_ll], [gb_lv], marker='*', s=260, color='gold', edgecolor='k',
               zorder=10, label=f'GT MLE [ℓ={math.exp(gb_ll):.2f}, σ²={math.exp(gb_lv):.2f}]')
    ax.scatter([math.log(bench.LS_TRUE)], [math.log(bench.VAR_TRUE)], marker='X',
               s=240, color='magenta', edgecolor='k', zorder=11,
               label=f'θ_true [ℓ={bench.LS_TRUE}, σ²={bench.VAR_TRUE}]')
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label('EM iter')
    ax.set_xlim(LOG_LS.min(), LOG_LS.max()); ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
    ax.set_title(f"EFGP from ls_init={LS_INIT}, run {N_EM} EM iters "
                 f"(ρ ramp→0.7 by {N_RAMP}, then held)\n"
                 f"does it turn back toward the gold GT MLE after overshooting?")
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'efgp_long_3p0.png', dpi=130)
    print(f"  saved {OUT_DIR / 'efgp_long_3p0.png'}", flush=True)


if __name__ == '__main__':
    main()
