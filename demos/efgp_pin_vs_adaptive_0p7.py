"""
efgp_pin_vs_adaptive_0p7.py

CONFIRMATORY TEST for the EFGP ℓ-trajectory choppiness mechanism.

Claim: the period-2 saw-tooth in EFGP's ℓ path (ls_init=0.7, gmix, deterministic
— NOT Hutchinson/MC noise) is a limit cycle driven by the ADAPTIVE-H spectral
grid re-tailoring its spacing h(θ) to the current ℓ EVERY outer EM iter, so the
collapsed M-step objective landscape shifts under its own updates.  SparseGP's
fixed inducing basis doesn't move → it descends smoothly.

Controlled knock-out: run EFGP from ls_init=0.7, N_EM=50, ρ=linspace(0.05,0.7,50)
(identical to the isotropic sweep) TWO ways:
  A) DEFAULT adaptive-h grid   — grid re-tailors h each iter   → expect saw-tooth
  B) pin_grid=True             — grid built ONCE at init, frozen → expect smooth
At ls_init=0.7 both use the SAME 289-mode lattice (K auto = init-0.7 grid), so the
ONLY difference is whether h moves.  If B damps the saw-tooth, the moving grid is
confirmed as the mechanism.

Prints choppiness metrics (std of Δlogℓ, #direction-reversals) for A vs B, plots
ℓ-vs-iter (left) and trajectories over GT contours (right).

Out: demos/_bench_gpdrift_inducing_sweep_iso_out/efgp_pin_vs_adaptive_0p7.png + .npz
Run under Slurm (efgp_pin_vs_adaptive_0p7.sbatch), NOT the login node.
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

LS_INIT = 0.7
N_EM = bench.N_EM            # 50 — same as the sweep
OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)


def _run(lik, op, ip, t_grid, sigma, pin_grid):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
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
        pin_grid=pin_grid, verbose=False)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([bench.VAR_INIT] + list(hist.variance))
    M = int(hist.final_grid.M) if hist.final_grid is not None else -1
    return dict(ls_traj=ls_traj, var_traj=var_traj, wall=wall, M=M)


def _chop(x):
    """(std of successive log-differences, # of direction reversals)."""
    lx = np.log(x); dd = np.diff(lx)
    rev = int(np.sum(np.diff(np.sign(dd)) != 0))
    return float(dd.std()), rev


def main():
    print(f"efgp_pin_vs_adaptive_0p7: N_EM={N_EM} ls_init={LS_INIT}  "
          f"devices={jax.devices()}", flush=True)

    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()

    LOG_LS = np.linspace(bench.LOG_LS_RANGE[0], bench.LOG_LS_RANGE[1], bench.N_GRID)
    LOG_VAR = np.linspace(bench.LOG_VAR_RANGE[0], bench.LOG_VAR_RANGE[1], bench.N_GRID)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}", flush=True)

    print("  [A] DEFAULT adaptive-h grid (h re-tailors each iter)...", flush=True)
    A = _run(lik, op, ip, t_grid, sigma, pin_grid=False)
    print(f"      M={A['M']}  final ℓ={A['ls_traj'][-1]:.4f}, "
          f"σ²={A['var_traj'][-1]:.4f}  wall={A['wall']:.1f}s", flush=True)

    print("  [B] pin_grid=True (grid frozen at init)...", flush=True)
    B = _run(lik, op, ip, t_grid, sigma, pin_grid=True)
    print(f"      M={B['M']}  final ℓ={B['ls_traj'][-1]:.4f}, "
          f"σ²={B['var_traj'][-1]:.4f}  wall={B['wall']:.1f}s", flush=True)

    # choppiness on the post-warmup segment (iters >= 8, where ℓ actually moves)
    w = 8
    Als_c = _chop(A['ls_traj'][w:]); Avar_c = _chop(A['var_traj'][w:])
    Bls_c = _chop(B['ls_traj'][w:]); Bvar_c = _chop(B['var_traj'][w:])
    print(f"\n  CHOPPINESS (post-warmup iters {w}..{N_EM}; std Δlogℓ, #reversals):",
          flush=True)
    print(f"    A adaptive-h  ℓ: std={Als_c[0]:.4f} rev={Als_c[1]:2d}   "
          f"σ²: std={Avar_c[0]:.4f} rev={Avar_c[1]:2d}", flush=True)
    print(f"    B pinned      ℓ: std={Bls_c[0]:.4f} rev={Bls_c[1]:2d}   "
          f"σ²: std={Bvar_c[0]:.4f} rev={Bvar_c[1]:2d}", flush=True)
    print(f"    → ℓ-noise ratio A/B = {Als_c[0]/max(Bls_c[0],1e-9):.2f}×,  "
          f"reversals {Als_c[1]}→{Bls_c[1]}", flush=True)
    print(f"  A ls_traj: {np.array2string(A['ls_traj'], precision=3, max_line_width=200)}",
          flush=True)
    print(f"  B ls_traj: {np.array2string(B['ls_traj'], precision=3, max_line_width=200)}",
          flush=True)

    np.savez(OUT_DIR / 'efgp_pin_vs_adaptive_0p7.npz',
             A_ls=A['ls_traj'], A_var=A['var_traj'], A_M=A['M'],
             B_ls=B['ls_traj'], B_var=B['var_traj'], B_M=B['M'],
             gb_ll=gb_ll, gb_lv=gb_lv, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
             Als_chop=Als_c, Bls_chop=Bls_c, warmup=w)

    # ---- plot ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    it = np.arange(N_EM + 1)
    axL.plot(it, A['ls_traj'], '-o', color='tab:red', ms=3,
             label=f"A: adaptive-h (default) — ℓ std(Δlogℓ)={Als_c[0]:.3f}, "
                   f"{Als_c[1]} reversals")
    axL.plot(it, B['ls_traj'], '-o', color='tab:cyan', ms=3,
             label=f"B: pin_grid — ℓ std(Δlogℓ)={Bls_c[0]:.3f}, "
                   f"{Bls_c[1]} reversals")
    axL.axhline(math.exp(gb_ll), color='gold', lw=2, label=f'GT MLE ℓ={math.exp(gb_ll):.2f}')
    axL.axvline(w, color='0.7', ls=':', lw=1, label=f'warmup end (iter {w})')
    axL.set_xlabel('EM iter'); axL.set_ylabel('lengthscale ℓ')
    axL.set_title('ℓ vs iter — pinning the grid damps the saw-tooth?')
    axL.legend(fontsize=8, loc='lower right')

    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    lo, hi = float(finite.min()), float(finite.max())
    levels = list(np.logspace(np.log10(max(lo, hi / 1000)), np.log10(hi), 10))
    axR.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                 cmap='viridis_r', extend='max')
    cs = axR.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                     colors='k', linewidths=0.4)
    axR.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
    for R, col, lab in [(A, 'tab:red', 'A adaptive-h'), (B, 'tab:cyan', 'B pinned')]:
        axR.plot(np.log(R['ls_traj']), np.log(R['var_traj']), '-o', color=col,
                 ms=3, lw=1.5, label=lab)
    axR.scatter([math.log(LS_INIT)], [math.log(bench.VAR_INIT)], marker='+',
                s=200, color='black', zorder=9, label='init')
    axR.scatter([gb_ll], [gb_lv], marker='*', s=260, color='gold', edgecolor='k',
                zorder=10, label='GT MLE')
    axR.set_xlim(LOG_LS.min(), LOG_LS.max()); axR.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    axR.set_xlabel('log ℓ'); axR.set_ylabel('log σ²')
    axR.set_title('trajectories over GT contours')
    axR.legend(fontsize=8, loc='lower right')

    fig.suptitle('EFGP ls_init=0.7: adaptive-h vs pinned spectral grid '
                 '(both M=289 modes, deterministic gmix)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / 'efgp_pin_vs_adaptive_0p7.png', dpi=130)
    print(f"  saved {OUT_DIR / 'efgp_pin_vs_adaptive_0p7.png'}", flush=True)


if __name__ == '__main__':
    main()
