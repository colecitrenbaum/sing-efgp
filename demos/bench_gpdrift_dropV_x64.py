"""
bench_gpdrift_dropV_x64.py

DECISIVE test of "what critical term does EFGP drop that SparseGP keeps?".

Per efgp_estep.tex (eq. transition-frozen, §Item (iv)), the per-transition
E-step cross-entropy is
    -E[δ'Σ⁻¹δ]/(2Δ) + E[δ'Σ⁻¹ f̄] - (Δ/2) E[ f̄'Σ⁻¹f̄  +  V ] ,
    V(x) = Σ_r σ_r⁻² φ(x)* A_r⁻¹ φ(x)   (the q(f) posterior variance).
EFGP production DROPS the E[V] piece (Approximation A); SparseGP KEEPS it
(it is term1+term2 in SparseGP.ff()).

Here we toggle exactly that term via SparseGP(..., include_qf_variance=False)
— nothing else changes (same E-step, same standard ELBO M-step, same matched
schedule).  If ℓ collapses from ~1.5 toward the oracle ~0.78 when V is dropped,
then E[V] is THE critical term.

Sweep M ∈ {25, 64, 256}; for each fit SparseGP with V kept vs V dropped, and
compare recovered (ℓ,σ²) to the oracle MLE and EFGP.

Run under Slurm (demos/bench_gpdrift_dropV.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.random as jr

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em

D = bench.D
T = bench.T
LS_INIT = 0.7
M_LIST = [25, 64, 256]


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init, xs_np,
                 include_V):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = bench._data_aware_zs(num_per_dim, xs_np)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad,
                      include_qf_variance=include_V)
    dp0 = dict(length_scales=jnp.full((D,), float(ls_init)),
               output_scale=jnp.asarray(math.sqrt(bench.VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, bench.N_EM)
    history = []
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=dp0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=bench.N_EM, n_iters_e=10,
        n_iters_m=bench.N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((bench.N_EM,), bench.MSTEP_LR),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] +
                       [float(np.mean(h['length_scales'])) for h in history])
    var_traj = np.array([bench.VAR_INIT] +
                        [float(h['output_scale']) ** 2 for h in history])
    # posterior-std calibration
    Sd = np.asarray(mp['S'][0])
    post_std = math.sqrt(np.mean([np.trace(Sd[t]) / D for t in range(Sd.shape[0])]))
    raw_err = float(np.sqrt(np.mean((np.asarray(mp['m'][0]) - xs_np) ** 2)))
    return dict(ls_traj=ls_traj, var_traj=var_traj,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2, wall=wall,
                post_std=post_std, raw_err=raw_err)


def main():
    print(f"bench_gpdrift_dropV: T={T} ls_init={LS_INIT} M={M_LIST}  "
          f"devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)

    LOG_LS = np.linspace(-2.0, 1.5, 21); LOG_VAR = np.linspace(-2.5, 2.0, 21)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  oracle MLE: ℓ={math.exp(gb_ll):.3f}  σ²={math.exp(gb_lv):.3f}",
          flush=True)

    print("  EFGP reference...", flush=True)
    e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
    print(f"    EFGP ℓ={e['ls']:.3f}  σ²={e['var']:.3f}", flush=True)

    results = {}
    for M in M_LIST:
        n_per = int(round(math.sqrt(M)))
        for include_V in (True, False):
            tag = 'V-kept' if include_V else 'V-dropped'
            print(f"  SP M={M} [{tag}] fit...", flush=True)
            s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, LS_INIT,
                             xs_np, include_V)
            print(f"    ℓ={s['ls']:.3f}  σ²={s['var']:.3f}  "
                  f"post_std={s['post_std']:.4f} (err {s['raw_err']:.4f})  "
                  f"wall={s['wall']:.0f}s", flush=True)
            results[(M, tag)] = s

    # ---- plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    OUT = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_out"
    OUT.mkdir(exist_ok=True)
    L_norm = L_gt - L_gt[np.unravel_index(np.nanargmin(L_gt), L_gt.shape)]
    fin = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    levels = list(np.logspace(np.log10(max(fin.min(), fin.max() / 1e3)),
                              np.log10(fin.max()), 10))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, tag in zip(axes, ['V-kept', 'V-dropped']):
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                    cmap='viridis_r', extend='max')
        ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels, colors='k',
                   linewidths=0.3)
        for c, M in zip(cm.viridis(np.linspace(0.15, 0.9, len(M_LIST))), M_LIST):
            s = results[(M, tag)]
            ax.plot(np.log(s['ls_traj']), np.log(s['var_traj']), '-s', color=c,
                    ms=3, lw=1.4, label=f"SP M={M} [ℓ={s['ls']:.2f}]")
        ax.plot(np.log(e['ls_traj']), np.log(e['var_traj']), '-o', color='C0',
                ms=3.5, lw=2.2, zorder=8, label=f"EFGP [ℓ={e['ls']:.2f}]")
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                   edgecolor='k', zorder=10, label=f"oracle ℓ={math.exp(gb_ll):.2f}")
        ax.scatter([math.log(bench.LS_TRUE)], [math.log(bench.VAR_TRUE)],
                   marker='X', s=200, color='magenta', edgecolor='k', zorder=10)
        ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
        ax.set_title(f"SparseGP  {tag}  (E[V] {'included' if tag=='V-kept' else 'dropped, = EFGP Approx A'})")
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle("Isolating the critical dropped term: SparseGP with vs without "
                 "the q(f)-variance term E[V] in the E-step", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT / 'dropV.png', dpi=130)
    print(f"  saved {OUT / 'dropV.png'}", flush=True)

    print(f"\n{'='*78}")
    print(f"SUMMARY  (oracle ℓ={math.exp(gb_ll):.3f}; EFGP ℓ={e['ls']:.3f})")
    print(f"{'='*78}")
    print(f"  {'M':>5s}  {'variant':>10s}  {'ℓ_final':>8s}  {'σ²_final':>9s}  "
          f"{'post_std':>8s}  {'err':>7s}  {'wall':>6s}")
    for M in M_LIST:
        for tag in ['V-kept', 'V-dropped']:
            s = results[(M, tag)]
            print(f"  {M:>5d}  {tag:>10s}  {s['ls']:>8.3f}  {s['var']:>9.3f}  "
                  f"{s['post_std']:>8.4f}  {s['raw_err']:>7.4f}  {s['wall']:>5.0f}s")


if __name__ == '__main__':
    main()
