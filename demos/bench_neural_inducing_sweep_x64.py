"""Inducing-point sweep (Phase C): does SparseGP-SING converge to EFGP-SING as
the number of inducing points grows?  Neural line-attractor data, RBF drift.

EFGP is fit ONCE and treated as the accurate gold-standard reference (there is
no ground-truth (ell, sigma_f^2) on real neural data).  SparseGP is fit at a
sweep of inducing counts num_per_dim in {5,6,8,10,12} -> {25,36,64,100,144}.
5x5=25 is the deliberately-insufficient baseline; 8x8=64 is the previous bench's
setting; the sweep goes past it.

For each we record wall-clock, recovered (ell, sigma_f^2), the per-EM-iter hyper
trajectory, the posterior drift field, the inferred latents, and the learned
input-effect B.

Deliverables (in _bench_neural_inducing_sweep_out/):
  sweep.npz                  raw results
  convergence.png            ell / sigma_f^2 / wall vs #inducing (EFGP ref line)
  hyper_trajectory_plane.png (ell, sigma_f^2)-plane EM paths, all methods  <-- primary
  line_attractor_grid.png    posterior drift / slow-point panel per method

Run under Slurm (demos/bench_neural_inducing_sweep.sbatch), NOT the login node.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax.* use (CLAUDE.md)
import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# Reuse ALL fit/plot machinery from the head-to-head demo.
import demos.bench_neural_efgp_vs_sparsegp as bench

D = bench.D
NUM_PER_DIM = [5, 6, 8, 10, 12]           # -> 25, 36, 64, 100, 144 inducing
OUT_DIR = _SING / "demos" / "_bench_neural_inducing_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


def main():
    print(f"\n=== Inducing sweep: EFGP vs SparseGP (RBF), neural line-attractor ===")
    print(f"    jax {jax.__version__}  devices={jax.devices()}")
    print(f"    SUBSAMPLE_T={bench.SUBSAMPLE_T}  N_EM={bench.N_EM}  "
          f"num_per_dim={NUM_PER_DIM}", flush=True)

    # ---- data + shared init (mirrors bench.main) ----
    norm = bench.load_neural_data()[::bench.SUBSAMPLE_T]
    n_t, n_n = norm.shape
    t_grid = jnp.arange(n_t) * (bench.DT * bench.SUBSAMPLE_T)
    ys = jnp.asarray(norm[None])
    inputs, o1, o2 = bench.build_inputs(n_t)
    onsets = (o1, o2)
    output_params, x0 = bench.initialize_params_pca(D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0
    hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_t, 64))[:, None]
                  * jnp.ones((1, D)))
    print(f"    T={n_t} N={n_n}  bbox lo={lo.round(2)} hi={hi.round(2)}  "
          f"ls_init={ls_init:.3f}", flush=True)

    # ---- EFGP once (reference) ----
    print("\n  [EFGP] fitting reference ...", flush=True)
    mp_e, hist_e, wall_e = bench.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                          X_template, ls_init)
    ls_e, var_e = float(hist_e.lengthscale[-1]), float(hist_e.variance[-1])
    B_e = np.asarray(hist_e.input_effect)
    m_e = np.asarray(mp_e['m'][0])
    ls_traj_e = np.asarray(hist_e.lengthscale)
    var_traj_e = np.asarray(hist_e.variance)
    efgp_fn = lambda X: bench.efgp_drift_field(mp_e, ls_e, var_e, t_grid,
                                               X_template, X)
    print(f"    EFGP wall={wall_e:.1f}s  ell={ls_e:.3f}  var={var_e:.3f}", flush=True)

    # ---- SparseGP sweep ----
    sp = []   # list of per-inducing result dicts
    for npd in NUM_PER_DIM:
        n_ind = npd * npd
        print(f"\n  [SparseGP] num_per_dim={npd} ({n_ind} inducing) ...", flush=True)
        (mp_s, fn_s, dp_s, gp_s, B_s, elbos_s,
         ls_h, var_h, wall_s) = bench.fit_sparsegp(
            ys, inputs, output_params, x0, t_grid, lo, hi, ls_init,
            num_per_dim=npd)
        drift_fn = (lambda fn_s, dp_s, gp_s: (
            lambda X: bench.sparsegp_drift_field(fn_s, dp_s, gp_s, X)))(fn_s, dp_s, gp_s)
        sp.append(dict(
            npd=npd, n_ind=n_ind, wall=wall_s,
            ls=float(ls_h[-1]), var=float(var_h[-1]),
            ls_traj=np.asarray(ls_h), var_traj=np.asarray(var_h),
            B=np.asarray(B_s), m=np.asarray(mp_s['m'][0]), drift_fn=drift_fn))
        print(f"    SparseGP wall={wall_s:.1f}s  ell={ls_h[-1]:.3f}  "
              f"var={var_h[-1]:.3f}", flush=True)

    # ---- shared slow-point eps (from EFGP median drift speed) ----
    gx = np.linspace(lo[0], hi[0], 30); gy = np.linspace(lo[1], hi[1], 30)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    sp_scale = float(np.median(np.linalg.norm(efgp_fn(pts), axis=-1)))
    eps_slow = max(0.1 * sp_scale, 1e-3)
    print(f"\n  slow-point eps={eps_slow:.3f}", flush=True)

    # ---- save ----
    np.savez(
        OUT_DIR / "sweep.npz",
        T=n_t, N=n_n, subsample=bench.SUBSAMPLE_T, n_em=bench.N_EM,
        num_per_dim=np.asarray(NUM_PER_DIM),
        n_ind=np.asarray([s['n_ind'] for s in sp]),
        wall_efgp=wall_e, ls_efgp=ls_e, var_efgp=var_e,
        ls_traj_efgp=ls_traj_e, var_traj_efgp=var_traj_e,
        B_efgp=B_e, m_efgp=m_e, xs_pca=xs_pca,
        wall_sparsegp=np.asarray([s['wall'] for s in sp]),
        ls_sparsegp=np.asarray([s['ls'] for s in sp]),
        var_sparsegp=np.asarray([s['var'] for s in sp]),
        ls_traj_sparsegp=np.asarray([s['ls_traj'] for s in sp]),
        var_traj_sparsegp=np.asarray([s['var_traj'] for s in sp]),
        m_sparsegp=np.asarray([s['m'] for s in sp]),
        B_sparsegp=np.asarray([s['B'] for s in sp]),
        lo=lo, hi=hi, onsets=np.asarray(onsets), eps_slow=eps_slow,
        ls_init=ls_init, var_init=bench.VAR_INIT)

    n_inds = np.asarray([s['n_ind'] for s in sp])

    # ---- figure 1: convergence vs #inducing ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(n_inds, [s['ls'] for s in sp], '-s', color='tab:red', label='SparseGP')
    ax[0].axhline(ls_e, color='tab:blue', ls='--', label=f'EFGP ref ({ls_e:.2f})')
    ax[0].set_xlabel('# inducing'); ax[0].set_ylabel(r'recovered $\ell$')
    ax[0].set_title(r'lengthscale $\ell$'); ax[0].legend(fontsize=8)
    ax[1].plot(n_inds, [s['var'] for s in sp], '-s', color='tab:red', label='SparseGP')
    ax[1].axhline(var_e, color='tab:blue', ls='--', label=f'EFGP ref ({var_e:.2f})')
    ax[1].set_xlabel('# inducing'); ax[1].set_ylabel(r'recovered $\sigma_f^2$')
    ax[1].set_title(r'variance $\sigma_f^2$'); ax[1].legend(fontsize=8)
    ax[2].semilogy(n_inds, [s['wall'] for s in sp], '-s', color='tab:red', label='SparseGP')
    ax[2].axhline(wall_e, color='tab:blue', ls='--', label=f'EFGP ({wall_e:.0f}s)')
    ax[2].set_xlabel('# inducing'); ax[2].set_ylabel('wall (s)')
    ax[2].set_title('wall-clock'); ax[2].legend(fontsize=8)
    fig.suptitle('SparseGP -> EFGP convergence vs # inducing points', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- figure 2 (PRIMARY): (ell, sigma_f^2)-plane EM trajectories ----
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = cm.viridis(np.linspace(0.15, 0.9, len(sp)))
    for s, c in zip(sp, colors):
        ax.plot(s['ls_traj'], s['var_traj'], '-', color=c, lw=1.3, alpha=0.9)
        ax.plot(s['ls_traj'][0], s['var_traj'][0], 'o', color=c, ms=4)
        ax.plot(s['ls_traj'][-1], s['var_traj'][-1], '*', color=c, ms=15,
                label=f"SparseGP {s['n_ind']}")
    ax.plot(ls_traj_e, var_traj_e, '-', color='tab:blue', lw=2.2, alpha=0.95)
    ax.plot(ls_traj_e[-1], var_traj_e[-1], '*', color='tab:blue', ms=20,
            markeredgecolor='k', label='EFGP (ref)')
    ax.plot(ls_init, bench.VAR_INIT, 'kx', ms=11, mew=2.5, label='init')
    ax.set_xlabel(r'lengthscale $\ell$'); ax.set_ylabel(r'variance $\sigma_f^2$')
    ax.set_title('Hyperparameter EM trajectories in the '
                 r'$(\ell,\sigma_f^2)$ plane'
                 '\nstars = converged; SparseGP endpoints should march toward EFGP '
                 'as #inducing grows')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hyper_trajectory_plane.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- figure 3: line-attractor panel grid (EFGP + each inducing count) ----
    panels = [("EFGP (ref)", efgp_fn, m_e, B_e,
               f"EFGP\nell={ls_e:.2f} var={var_e:.2f} {wall_e:.0f}s")]
    for s in sp:
        panels.append((f"{s['n_ind']} ind", s['drift_fn'], s['m'], s['B'],
                       f"SparseGP {s['n_ind']}\nell={s['ls']:.2f} "
                       f"var={s['var']:.2f} {s['wall']:.0f}s"))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    im = None
    for k, (name, dfn, m_lat, B, title) in enumerate(panels):
        im = bench.attractor_panel(axes[k], dfn, m_lat, B, onsets, lo, hi,
                                   eps_slow, title=title)
    for k in range(len(panels), len(axes)):
        axes[k].axis('off')
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02,
                     label='slow-point proxy')
    fig.suptitle('Neural line-attractor: posterior drift field convergence '
                 'SparseGP -> EFGP\nblue=latents  black=drift  orange=input B',
                 fontsize=12)
    fig.savefig(OUT_DIR / "line_attractor_grid.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/{{sweep.npz, convergence.png, "
          f"hyper_trajectory_plane.png, line_attractor_grid.png}}")
    print("\n  === SUMMARY ===")
    print(f"  EFGP        ell={ls_e:.3f}  var={var_e:.3f}  wall={wall_e:.1f}s")
    for s in sp:
        print(f"  SparseGP {s['n_ind']:4d}  ell={s['ls']:.3f}  var={s['var']:.3f}  "
              f"wall={s['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
