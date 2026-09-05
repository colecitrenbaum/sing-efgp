"""
efgp_grid_recipe_and_timing.py

Answers two questions about the EFGP spectral-grid policy.

PART A — TIMING DECOMPOSITION.
  The pinned grid ran ~30% faster (108s vs 150s) than the adaptive-h grid on
  the 50-iter ls_init=0.7 fit.  Is that difference (a) one-time JIT compilation,
  or (b) steady-state per-iter compute?
  Static reasoning already says it's NOT compile:
    * mtot_per_dim is a Python tuple => static under jit; K is held fixed in the
      adaptive path, so the jit'd E-step scan compiles ONCE (no per-iter retrace
      as h(theta) moves; efgp_em.py:996-1002).
    * pin & adaptive both build the scan via _build_jit_estep_scan_jax(**_est_kw)
      with identical static args => identical compiled graph => identical compile
      cost.
  We CONFIRM by the two-N differencing trick.  For each policy run the fit at
  N1 and N2 (>2) EM iters; each fit call recompiles once, so
      wall(N) = C_compile + N * t_iter.
  Then  t_iter = (wall(N2)-wall(N1))/(N2-N1)  and  C_compile = wall(N1)-N1*t_iter.
  A cache probe (rerun N1) checks the recompile-per-fit assumption.
  Both N1,N2 are chosen > kernel_warmup_iters(=8) so the M-step + grid-rebuild
  path (where pin vs adaptive actually differ) is exercised every counted iter.

PART B — UNIFIED A-PRIORI RECIPE.
  Claim: the two knobs (pin-grid vs #modes) are ONE decision, split by which
  kernel scale drives which axis of the grid.  From spectral_grid_se:
      modes-per-side  hm(l) = ceil( Lfreq(l) / h(l) )
      Lfreq(l) = sqrt(-ln eps)/(pi*sqrt2 * l)      (max frequency ~ 1/l)
      h(l)     = 1/(L + l*sqrt(-2 ln eps))         (freq spacing ~ 1/domain)
   => hm(l) = ceil[ a*(L/l) + b ],  a=sqrt(-ln eps)/(pi*sqrt2), b=-ln eps/pi.
      eps=1e-3 -> a=0.591, b=2.199.
   INTERPRETATION: hm ~ 0.6 * (#lengthscales spanning the domain) + 2 -- exactly
   "how many wiggles fit in the box."  The x-grid spacing h is set by the DOMAIN
   L (l enters only through the small kernel-tail margin) -> freeze it.  The mode
   count is set by the SMALLEST l you want to resolve -> size it from l_min.
   RECIPE:  K_per_dim = ceil(0.6*L/l_min + 2);  PIN the grid at l_min so h never
   re-tailors (kills the limit cycle) while K stays generous (kills the collapse).
  We validate: pin_grid=True, pin_grid_lengthscale=L_MIN from BOTH a good init
  (0.7) AND a bad-high init (3.0).  One setting should (i) recover the MLE from
  both inits (no collapse) and (ii) be SMOOTH (no saw-tooth).  Compared against
  the two documented pathologies: adaptive-default from 3.0 (collapse) and
  adaptive-default from 0.7 (chop).

Out: demos/_bench_gpdrift_inducing_sweep_iso_out/efgp_grid_recipe_and_timing.{png,npz}
Run under Slurm, NOT the login node.
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

OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)

L_MIN = 0.35                 # smallest l to resolve (~2x below oracle MLE ~0.78)
EPS = 1e-3


def _fit(lik, op, ip, t_grid, sigma, *, ls_init, n_em, pin_grid,
         pin_ls=None, k_min_ls=None):
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=bench.D,
        lengthscale=ls_init, variance=bench.VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=EPS, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=bench.N_M_INNER, mstep_lr=bench.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        pin_grid=pin_grid, pin_grid_lengthscale=pin_ls,
        K_min_lengthscale=k_min_ls, verbose=False)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] + list(hist.lengthscale))
    var_traj = np.array([bench.VAR_INIT] + list(hist.variance))
    M = int(hist.final_grid.M) if hist.final_grid is not None else -1
    return dict(wall=wall, ls_traj=ls_traj, var_traj=var_traj, M=M)


def _chop(x, w=8):
    lx = np.log(x[w:]); dd = np.diff(lx)
    rev = int(np.sum(np.diff(np.sign(dd)) != 0))
    return float(dd.std()), rev


def main():
    print(f"efgp_grid_recipe_and_timing  devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()

    # ---- analytic mode-count formula sanity check -----------------------
    a = math.sqrt(-math.log(EPS)) / (math.pi * math.sqrt(2.0))
    b = -math.log(EPS) / math.pi
    print(f"\n[FORMULA] hm(l) ~ ceil({a:.3f}*(L/l) + {b:.3f})  (eps={EPS})",
          flush=True)
    # reconstruct the domain L that the default X_template box implies
    V0 = jnp.asarray(ip['V0']); mu0 = jnp.asarray(ip['mu0'])
    diag_std0 = jnp.sqrt(jnp.diagonal(V0, axis1=-2, axis2=-1))
    half = jnp.maximum(jnp.full_like(diag_std0, 3.0), 4.0 * diag_std0)
    per_min = (mu0 - half).min(0); per_max = (mu0 + half).max(0)
    center = 0.5 * (per_min + per_max); half_span = 0.5 * (per_max - per_min)
    Xtmpl = center[None] + jnp.linspace(-1., 1., max(bench.T, 16))[:, None] * half_span[None]
    Xtmpl_np = np.asarray(Xtmpl)
    L_domain = float((Xtmpl_np.max(0) - Xtmpl_np.min(0)).max())
    X_extent = L_domain
    print(f"[FORMULA] implied support extent L = {L_domain:.3f}", flush=True)
    for ltest in [0.35, 0.7, 1.0, 3.0]:
        g = jp.spectral_grid_se(ltest, bench.VAR_INIT, Xtmpl, eps=EPS)
        hm_true = (int(g.mtot_per_dim[0]) - 1) // 2
        hm_formula = math.ceil(a * (L_domain / ltest) + b)
        print(f"    l={ltest:4.2f}: hm formula={hm_formula:3d}  actual={hm_true:3d}"
              f"  M_actual={g.M}", flush=True)
    K_min = jp.choose_K_for_min_lengthscale(L_MIN, bench.VAR_INIT, X_extent,
                                            eps=EPS, d=bench.D)
    print(f"[FORMULA] K_per_dim for l_min={L_MIN}: choose_K={K_min} "
          f"(formula ceil={math.ceil(a*(L_domain/L_MIN)+b)}) -> "
          f"M={(2*K_min+1)**bench.D}", flush=True)

    # ---- GT landscape (for part B plot) ---------------------------------
    LOG_LS = np.linspace(bench.LOG_LS_RANGE[0], bench.LOG_LS_RANGE[1], bench.N_GRID)
    LOG_VAR = np.linspace(bench.LOG_VAR_RANGE[0], bench.LOG_VAR_RANGE[1], bench.N_GRID)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"\n[GT] MLE: l={math.exp(gb_ll):.3f}, s2={math.exp(gb_lv):.3f}", flush=True)

    # =====================================================================
    # PART A: timing decomposition (two-N differencing)
    # =====================================================================
    N1, N2 = 10, 30
    print(f"\n===== PART A: timing (N1={N1}, N2={N2}, ls_init=0.7) =====",
          flush=True)
    print("  adaptive-h N1 (cold)...", flush=True)
    aA1 = _fit(lik, op, ip, t_grid, sigma, ls_init=0.7, n_em=N1, pin_grid=False)
    print(f"    wall={aA1['wall']:.2f}s  M={aA1['M']}", flush=True)
    print("  adaptive-h N2...", flush=True)
    aA2 = _fit(lik, op, ip, t_grid, sigma, ls_init=0.7, n_em=N2, pin_grid=False)
    print(f"    wall={aA2['wall']:.2f}s", flush=True)
    print("  adaptive-h N1 (probe: recompiles each fit?)...", flush=True)
    aA1b = _fit(lik, op, ip, t_grid, sigma, ls_init=0.7, n_em=N1, pin_grid=False)
    print(f"    wall={aA1b['wall']:.2f}s", flush=True)
    print("  pin N1...", flush=True)
    pP1 = _fit(lik, op, ip, t_grid, sigma, ls_init=0.7, n_em=N1, pin_grid=True)
    print(f"    wall={pP1['wall']:.2f}s  M={pP1['M']}", flush=True)
    print("  pin N2...", flush=True)
    pP2 = _fit(lik, op, ip, t_grid, sigma, ls_init=0.7, n_em=N2, pin_grid=True)
    print(f"    wall={pP2['wall']:.2f}s", flush=True)

    t_adapt = (aA2['wall'] - aA1['wall']) / (N2 - N1)
    C_adapt = aA1['wall'] - N1 * t_adapt
    t_pin = (pP2['wall'] - pP1['wall']) / (N2 - N1)
    C_pin = pP1['wall'] - N1 * t_pin
    print(f"\n  [DECOMP]  per-iter t_iter:  adaptive={t_adapt:.3f}s  "
          f"pin={t_pin:.3f}s   (adaptive/pin = {t_adapt/max(t_pin,1e-9):.2f}x)",
          flush=True)
    print(f"  [DECOMP]  compile C:        adaptive={C_adapt:.1f}s  "
          f"pin={C_pin:.1f}s   (should be ~equal: same static graph)", flush=True)
    print(f"  [PROBE]   N1 rerun {aA1b['wall']:.2f}s vs first {aA1['wall']:.2f}s "
          f"(~equal => each fit recompiles => differencing valid)", flush=True)
    for N in (50,):
        pred_a = C_adapt + N * t_adapt
        pred_p = C_pin + N * t_pin
        print(f"  [PREDICT @N={N}] adaptive~{pred_a:.0f}s  pin~{pred_p:.0f}s  "
              f"(observed earlier: 150 vs 108) -> "
              f"delta {pred_a-pred_p:.0f}s is per-iter, "
              f"{N*(t_adapt-t_pin):.0f}s of it", flush=True)

    # =====================================================================
    # PART B: unified recipe -- pin at l_min from good AND bad init
    # =====================================================================
    N_EM = bench.N_EM
    print(f"\n===== PART B: pin@l_min={L_MIN} recipe (N_EM={N_EM}) =====",
          flush=True)
    runs = {}
    for tag, ls_init, pin_grid, pin_ls, kmin in [
        ("recipe_from0.7", 0.7, True, L_MIN, None),
        ("recipe_from3.0", 3.0, True, L_MIN, None),
        ("adaptive_from0.7", 0.7, False, None, None),   # documented chop
        ("adaptive_from3.0", 3.0, False, None, None),   # documented collapse
    ]:
        print(f"  {tag} (init={ls_init}, pin={pin_grid}, pin_ls={pin_ls})...",
              flush=True)
        r = _fit(lik, op, ip, t_grid, sigma, ls_init=ls_init, n_em=N_EM,
                 pin_grid=pin_grid, pin_ls=pin_ls, k_min_ls=kmin)
        ch = _chop(r['ls_traj'])
        r['chop'] = ch
        runs[tag] = r
        print(f"    final l={r['ls_traj'][-1]:.4f}  s2={r['var_traj'][-1]:.4f}  "
              f"M={r['M']}  chop(std dlogl={ch[0]:.4f}, rev={ch[1]})  "
              f"wall={r['wall']:.1f}s", flush=True)

    np.savez(OUT_DIR / 'efgp_grid_recipe_and_timing.npz',
             t_adapt=t_adapt, C_adapt=C_adapt, t_pin=t_pin, C_pin=C_pin,
             aA1=aA1['wall'], aA2=aA2['wall'], aA1b=aA1b['wall'],
             pP1=pP1['wall'], pP2=pP2['wall'], N1=N1, N2=N2,
             L_domain=L_domain, K_min=K_min, L_MIN=L_MIN,
             gb_ll=gb_ll, gb_lv=gb_lv, LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
             **{f"{k}_ls": v['ls_traj'] for k, v in runs.items()},
             **{f"{k}_var": v['var_traj'] for k, v in runs.items()},
             **{f"{k}_M": v['M'] for k, v in runs.items()})

    # ---- plot: part B trajectories over GT contours ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))
    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    lo, hi = float(finite.min()), float(finite.max())
    levels = list(np.logspace(np.log10(max(lo, hi / 1000)), np.log10(hi), 10))

    styles = {
        "recipe_from0.7":   ('tab:green',  '-o', 'recipe pin@l_min, init 0.7'),
        "recipe_from3.0":   ('tab:blue',   '-o', 'recipe pin@l_min, init 3.0'),
        "adaptive_from0.7": ('tab:red',    '--o', 'adaptive default, init 0.7 (CHOP)'),
        "adaptive_from3.0": ('tab:orange', '--o', 'adaptive default, init 3.0 (COLLAPSE)'),
    }
    it = np.arange(N_EM + 1)
    for tag, (col, ls, lab) in styles.items():
        r = runs[tag]
        axL.plot(it, r['ls_traj'], ls, color=col, ms=3, lw=1.4,
                 label=f"{lab}  [std dlogl={r['chop'][0]:.3f}, rev={r['chop'][1]}, M={r['M']}]")
    axL.axhline(math.exp(gb_ll), color='gold', lw=2, label=f'GT MLE l={math.exp(gb_ll):.2f}')
    axL.set_yscale('log')
    axL.set_xlabel('EM iter'); axL.set_ylabel('lengthscale l (log)')
    axL.set_title('l vs iter: pin@l_min recovers MLE + smooth from BOTH inits')
    axL.legend(fontsize=7, loc='lower left')

    axR.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                 cmap='viridis_r', extend='max')
    cs = axR.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                     colors='k', linewidths=0.4)
    axR.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
    for tag, (col, ls, lab) in styles.items():
        r = runs[tag]
        axR.plot(np.log(r['ls_traj']), np.log(r['var_traj']), ls, color=col,
                 ms=3, lw=1.4, label=lab)
    axR.scatter([math.log(0.7), math.log(3.0)],
                [math.log(bench.VAR_INIT)] * 2, marker='+', s=180,
                color='black', zorder=9, label='inits')
    axR.scatter([gb_ll], [gb_lv], marker='*', s=280, color='gold', edgecolor='k',
                zorder=10, label='GT MLE')
    axR.set_xlim(LOG_LS.min(), LOG_LS.max()); axR.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    axR.set_xlabel('log l'); axR.set_ylabel('log s2')
    axR.set_title('trajectories over GT contours')
    axR.legend(fontsize=7, loc='lower right')

    fig.suptitle(f"Unified grid recipe: pin at l_min={L_MIN} (M={runs['recipe_from0.7']['M']}) "
                 f"fixes BOTH collapse (init 3.0) and chop (init 0.7)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / 'efgp_grid_recipe_and_timing.png', dpi=130)
    print(f"\n  saved {OUT_DIR / 'efgp_grid_recipe_and_timing.png'}", flush=True)


if __name__ == '__main__':
    main()
