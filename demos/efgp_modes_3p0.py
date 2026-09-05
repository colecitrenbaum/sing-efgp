"""
efgp_modes_3p0.py

Hypothesis test: EFGP's ℓ-collapse from ls_init=3.0 is a SPECTRAL-GRID
RESOLUTION artifact, not a fundamental failure.

`fit_efgp_sing_jax`'s DEFAULT grid policy auto-sizes the Fourier mode count
K_per_dim from the INITIAL lengthscale (efgp_em.py:697-699).  For an SE kernel
the spectral density is concentrated at low frequency when ℓ is large, so:
  init ℓ=3.0  → FEW modes   (too few to resolve ℓ once EM drives it to ~0.78)
  init ℓ=0.7  → MANY modes  (plenty — recovers the MLE, as the sweep showed)
Once ℓ shrinks below what the mode count can resolve, spectral_grid_se_fixed_K
is forced to h > 1/L (spatial aliasing; efgp_jax_primitives.py:259-267), the
collapsed-NLL is mis-estimated, and ℓ runs away to 0 / σ² blows up.

This script runs EFGP from ls_init=3.0 TWO ways, everything else identical to the
sweep (150 iters: ρ ramp→0.7 by 50, then held):
  A) DEFAULT grid           — K from init 3.0  (few modes)  → expect collapse
  B) K_min_lengthscale=0.5  — K sized for a SHORT ℓ (many modes) → expect it to
                              recover the MLE (ℓ≈0.78), matching the 0.7 init.
Prints the mode counts (grid.M) for init 3.0-default, init 0.7-default, and the
K_min=0.5 lattice, then overlays both trajectories on the GT oracle contours.

Out: demos/_bench_gpdrift_inducing_sweep_iso_out/efgp_modes_3p0.png + .npz
Run under Slurm (efgp_modes_3p0.sbatch), NOT the login node.
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
import sing.efgp_jax_primitives as jp

LS_INIT = 3.0
K_MIN_LS = 0.5               # size mode lattice for a short ℓ (below MLE 0.78)
N_RAMP = bench.N_EM          # 50 — same ramp as the sweep
N_HOLD = 100
N_EM = N_RAMP + N_HOLD
OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)


def _run(lik, op, ip, t_grid, sigma, k_min_lengthscale):
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
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        K_min_lengthscale=k_min_lengthscale, verbose=False)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([bench.VAR_INIT] + list(hist.variance))
    M = int(hist.final_grid.M) if hist.final_grid is not None else -1
    return dict(ls_traj=ls_traj, var_traj=var_traj, wall=wall, M=M)


def main():
    print(f"efgp_modes_3p0: N_EM={N_EM} ls_init={LS_INIT} K_min_ls={K_MIN_LS}  "
          f"devices={jax.devices()}", flush=True)

    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)

    # --- mode-count diagnostic: build the default tailored grids at init 3.0
    #     and 0.7 (what the sweep used), and the K_min=0.5 lattice. ---
    V0 = jnp.asarray(ip['V0']); mu0 = jnp.asarray(ip['mu0'])
    diag_std0 = jnp.sqrt(jnp.diagonal(V0, axis1=-2, axis2=-1))
    half = jnp.maximum(jnp.full_like(diag_std0, 3.0), 4.0 * diag_std0)
    per_min = (mu0 - half).min(0); per_max = (mu0 + half).max(0)
    center = 0.5 * (per_min + per_max); half_span = 0.5 * (per_max - per_min)
    Xtmpl = center[None] + jnp.linspace(-1., 1., max(bench.T, 16))[:, None] * half_span[None]
    X_extent = float((np.asarray(Xtmpl.max(0)) - np.asarray(Xtmpl.min(0))).max())

    g3 = jp.spectral_grid_se(3.0, bench.VAR_INIT, Xtmpl, eps=1e-3)
    g07 = jp.spectral_grid_se(0.7, bench.VAR_INIT, Xtmpl, eps=1e-3)
    Kmin = jp.choose_K_for_min_lengthscale(K_MIN_LS, bench.VAR_INIT, X_extent,
                                           eps=1e-3, d=bench.D)
    Mmin = (2 * Kmin + 1) ** bench.D
    print(f"  MODE COUNTS (grid.M = total Fourier modes):", flush=True)
    print(f"    init ℓ=3.0 default : M={g3.M:5d}  (mtot/dim={g3.mtot_per_dim})",
          flush=True)
    print(f"    init ℓ=0.7 default : M={g07.M:5d}  (mtot/dim={g07.mtot_per_dim})",
          flush=True)
    print(f"    K_min_ls={K_MIN_LS}      : M={Mmin:5d}  (K_per_dim={Kmin})",
          flush=True)

    LOG_LS = np.linspace(bench.LOG_LS_RANGE[0], bench.LOG_LS_RANGE[1], bench.N_GRID)
    LOG_VAR = np.linspace(bench.LOG_VAR_RANGE[0], bench.LOG_VAR_RANGE[1], bench.N_GRID)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}", flush=True)

    print("  [A] DEFAULT grid (K from init 3.0, few modes)...", flush=True)
    A = _run(lik, op, ip, t_grid, sigma, k_min_lengthscale=None)
    print(f"      grid M={A['M']}  final ℓ={A['ls_traj'][-1]:.4f}, "
          f"σ²={A['var_traj'][-1]:.4f}  wall={A['wall']:.1f}s", flush=True)

    print(f"  [B] K_min_lengthscale={K_MIN_LS} (many modes)...", flush=True)
    B = _run(lik, op, ip, t_grid, sigma, k_min_lengthscale=K_MIN_LS)
    print(f"      grid M={B['M']}  final ℓ={B['ls_traj'][-1]:.4f}, "
          f"σ²={B['var_traj'][-1]:.4f}  wall={B['wall']:.1f}s", flush=True)

    for tag, R in [('A default', A), ('B K_min', B)]:
        print(f"  traj {tag}: " +
              "  ".join(f"it{it}:ℓ={R['ls_traj'][it]:.3f}"
                        for it in [0, 25, 50, 75, 100, 150]), flush=True)

    np.savez(OUT_DIR / 'efgp_modes_3p0.npz',
             A_ls=A['ls_traj'], A_var=A['var_traj'], A_M=A['M'],
             B_ls=B['ls_traj'], B_var=B['var_traj'], B_M=B['M'],
             M_init3=g3.M, M_init07=g07.M, M_kmin=Mmin,
             gb_ll=gb_ll, gb_lv=gb_lv, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt)

    # ---- plot ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7))
    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    lo, hi = float(finite.min()), float(finite.max())
    levels = list(np.logspace(np.log10(max(lo, hi / 1000)), np.log10(hi), 10))
    ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                cmap='viridis_r', extend='max')
    cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                    colors='k', linewidths=0.4)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

    for R, col, lab in [(A, 'tab:red',  f"A: default grid (M={A['M']}, few modes) "
                                        f"→ ℓ={A['ls_traj'][-1]:.2f} COLLAPSE"),
                        (B, 'tab:cyan', f"B: K_min_ls={K_MIN_LS} (M={B['M']}, many modes) "
                                        f"→ ℓ={B['ls_traj'][-1]:.2f} RECOVER")]:
        lp = np.log(R['ls_traj']); vp = np.log(R['var_traj'])
        ax.plot(lp, vp, '-o', color=col, ms=3, lw=1.6, label=lab)
        ax.scatter([lp[-1]], [vp[-1]], marker='o', s=110, color=col,
                   edgecolor='k', zorder=9)
    ax.scatter([math.log(LS_INIT)], [math.log(bench.VAR_INIT)], marker='+',
               s=200, color='black', zorder=9, label='init (ℓ=3.0)')
    ax.scatter([gb_ll], [gb_lv], marker='*', s=280, color='gold', edgecolor='k',
               zorder=10, label=f'GT MLE [ℓ={math.exp(gb_ll):.2f}, σ²={math.exp(gb_lv):.2f}]')
    ax.scatter([math.log(bench.LS_TRUE)], [math.log(bench.VAR_TRUE)], marker='X',
               s=240, color='magenta', edgecolor='k', zorder=11,
               label=f'θ_true [ℓ={bench.LS_TRUE}, σ²={bench.VAR_TRUE}]')
    ax.set_xlim(LOG_LS.min(), LOG_LS.max()); ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
    ax.set_title("EFGP from ls_init=3.0: ℓ-collapse is a FOURIER-MODE-COUNT artifact\n"
                 "default sizes K from the big init (too few modes); K_min_ls "
                 "sizes K for a short ℓ → recovers the MLE")
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'efgp_modes_3p0.png', dpi=130)
    print(f"  saved {OUT_DIR / 'efgp_modes_3p0.png'}", flush=True)


if __name__ == '__main__':
    main()
