"""EFGP Fourier-mode sweep: is the EFGP-vs-SparseGP hyper gap an artifact of a
too-coarse EFGP spectral (random-feature) approximation?

The neural sweep (bench_neural_inducing_sweep_x64.py) found EFGP settles at
ell=3.53, sigma_f^2=0.29 while SparseGP (as #inducing -> 144) converges to a
DIFFERENT optimum ell~3.08, sigma_f^2~0.15.  One explanation: EFGP's spectral
grid (M=121 modes at eps_grid=1e-2) is too coarse, biasing its collapsed-ML
M-step.  If so, tightening eps_grid (more Fourier modes) should pull EFGP's
recovered hypers toward SparseGP's.  If EFGP's hypers are flat in #modes, the
gap is methodological (not an approximation artifact).

Sweeps eps_grid in {1e-2, 1e-3, 1e-4} (the DEFAULT grid policy auto-picks a
larger mode lattice K for tighter eps).  Everything else identical to the
neural bench.  Reports recovered (ell, sigma_f^2), the mode count M, wall, and
the EM trajectory; overlays the SparseGP asymptote from the earlier sweep.

Run on swl1 (demos/bench_efgp_modes.sbatch).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax.* use
import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import demos.bench_neural_efgp_vs_sparsegp as bench
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
from sing.inputs import InputSignals

D = bench.D
# spectral-grid tolerance -> more Fourier modes as eps shrinks.
# On this neural bbox: eps {1e-2,1e-4,1e-6,1e-8,1e-10} -> M {121,289,529,841,1225}.
EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
OUT_DIR = _SING / "demos" / "_bench_efgp_modes_out"
OUT_DIR.mkdir(exist_ok=True)
SWEEP_NPZ = _SING / "demos" / "_bench_neural_inducing_sweep_out" / "sweep.npz"


def fit_efgp_eps(ys, inputs, output_params, x0, t_grid, X_template, ls_init, eps):
    """bench.fit_efgp but with a caller-supplied eps_grid (mode count)."""
    T = ys.shape[1]
    lik = bench.GLik(ys, jnp.ones((1, T), dtype=bool))
    ip = dict(mu0=x0, V0=jnp.eye(D)[None])
    rho = jnp.linspace(0.05, 0.7, bench.N_EM)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(output_params), init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=bench.VAR_INIT, sigma=bench.SIGMA,
        sigma_drift_sq=bench.SIGMA ** 2, eps_grid=eps,
        estep_method='gmix',
        n_em_iters=bench.N_EM, n_estep_iters=bench.N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=bench.N_MSTEP_INNER, mstep_lr=bench.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        input_signals=InputSignals(inputs),
        learn_input_effect=True, input_effect_warmup_iters=8,
        X_template=X_template,
        verbose=True,
    )
    wall = time.time() - t0
    return mp, hist, wall


def main():
    print(f"\n=== EFGP Fourier-mode sweep (neural), eps_grid={EPS_LIST} ===")
    print(f"    jax {jax.__version__}  devices={jax.devices()}", flush=True)

    norm = bench.load_neural_data()[::bench.SUBSAMPLE_T]
    n_t, n_n = norm.shape
    t_grid = jnp.arange(n_t) * (bench.DT * bench.SUBSAMPLE_T)
    ys = jnp.asarray(norm[None])
    inputs, o1, o2 = bench.build_inputs(n_t)
    output_params, x0 = bench.initialize_params_pca(D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0
    hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_t, 64))[:, None]
                  * jnp.ones((1, D)))
    print(f"    T={n_t} N={n_n}  ls_init={ls_init:.3f}", flush=True)

    # SparseGP asymptote (reference) from the earlier inducing sweep, if present.
    sp_ls = sp_var = None
    if SWEEP_NPZ.exists():
        d = np.load(SWEEP_NPZ)
        sp_ls = float(d['ls_sparsegp'][-1]); sp_var = float(d['var_sparsegp'][-1])
        print(f"    SparseGP asymptote (144 ind): ell={sp_ls:.3f} var={sp_var:.3f}",
              flush=True)

    runs = []
    for eps in EPS_LIST:
        # Mode count of the initial tailored grid at this eps (default policy
        # matches K to this) — the # of Fourier modes.
        gs = jp.spectral_grid_se(ls_init, bench.VAR_INIT, X_template, eps=eps)
        M = int(gs.M)
        print(f"\n  [EFGP] eps_grid={eps:.0e}  ~{M} Fourier modes "
              f"(mtot_per_dim={tuple(gs.mtot_per_dim)}) ...", flush=True)
        mp, hist, wall = fit_efgp_eps(ys, inputs, output_params, x0, t_grid,
                                      X_template, ls_init, eps)
        runs.append(dict(
            eps=eps, M=M, wall=wall,
            ls=float(hist.lengthscale[-1]), var=float(hist.variance[-1]),
            ls_traj=np.asarray(hist.lengthscale),
            var_traj=np.asarray(hist.variance)))
        print(f"    eps={eps:.0e}  M~{M}  ell={runs[-1]['ls']:.3f}  "
              f"var={runs[-1]['var']:.3f}  wall={wall:.1f}s", flush=True)

    np.savez(
        OUT_DIR / "efgp_modes.npz",
        eps=np.asarray([r['eps'] for r in runs]),
        M=np.asarray([r['M'] for r in runs]),
        wall=np.asarray([r['wall'] for r in runs]),
        ls=np.asarray([r['ls'] for r in runs]),
        var=np.asarray([r['var'] for r in runs]),
        ls_traj=np.asarray([r['ls_traj'] for r in runs]),
        var_traj=np.asarray([r['var_traj'] for r in runs]),
        sparsegp_ls=(np.nan if sp_ls is None else sp_ls),
        sparsegp_var=(np.nan if sp_var is None else sp_var),
        ls_init=ls_init, var_init=bench.VAR_INIT)

    Ms = [r['M'] for r in runs]

    # ---- figure 1: recovered ell / sigma_f^2 / wall vs # modes ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(Ms, [r['ls'] for r in runs], '-o', color='tab:blue', label='EFGP')
    if sp_ls is not None:
        ax[0].axhline(sp_ls, color='tab:red', ls='--',
                      label=f'SparseGP asymptote ({sp_ls:.2f})')
    ax[0].set_xscale('log'); ax[0].set_xlabel('# Fourier modes M')
    ax[0].set_ylabel(r'recovered $\ell$'); ax[0].set_title(r'lengthscale $\ell$')
    ax[0].legend(fontsize=8)
    ax[1].plot(Ms, [r['var'] for r in runs], '-o', color='tab:blue', label='EFGP')
    if sp_var is not None:
        ax[1].axhline(sp_var, color='tab:red', ls='--',
                      label=f'SparseGP asymptote ({sp_var:.2f})')
    ax[1].set_xscale('log'); ax[1].set_xlabel('# Fourier modes M')
    ax[1].set_ylabel(r'recovered $\sigma_f^2$'); ax[1].set_title(r'variance $\sigma_f^2$')
    ax[1].legend(fontsize=8)
    ax[2].loglog(Ms, [r['wall'] for r in runs], '-o', color='tab:blue')
    ax[2].set_xlabel('# Fourier modes M'); ax[2].set_ylabel('wall (s)')
    ax[2].set_title('wall-clock')
    fig.suptitle('EFGP recovered hypers vs # Fourier modes\n'
                 '(does tightening the spectral grid move EFGP toward SparseGP?)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "efgp_modes_convergence.png", dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- figure 2: (ell, sigma_f^2)-plane EM trajectories per mode count ----
    fig, axp = plt.subplots(figsize=(7.2, 6.0))
    colors = cm.plasma(np.linspace(0.1, 0.85, len(runs)))
    for r, c in zip(runs, colors):
        axp.plot(r['ls_traj'], r['var_traj'], '-', color=c, lw=1.4, alpha=0.9)
        axp.plot(r['ls_traj'][-1], r['var_traj'][-1], '*', color=c, ms=16,
                 label=f"EFGP M~{r['M']} (eps={r['eps']:.0e})")
    if sp_ls is not None:
        axp.plot(sp_ls, sp_var, 'X', color='tab:red', ms=15, mew=2,
                 label='SparseGP asymptote (144 ind)')
    axp.plot(ls_init, bench.VAR_INIT, 'kx', ms=11, mew=2.5, label='init')
    axp.set_xlabel(r'lengthscale $\ell$'); axp.set_ylabel(r'variance $\sigma_f^2$')
    axp.set_title('EFGP EM trajectories vs # Fourier modes\n'
                  'stars = converged EFGP; red X = SparseGP asymptote')
    axp.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "efgp_modes_trajectory_plane.png", dpi=140,
                bbox_inches='tight')
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/{{efgp_modes.npz, efgp_modes_convergence.png, "
          f"efgp_modes_trajectory_plane.png}}")
    print("\n  === SUMMARY ===")
    if sp_ls is not None:
        print(f"  SparseGP asymptote   ell={sp_ls:.3f}  var={sp_var:.3f}")
    for r in runs:
        print(f"  EFGP eps={r['eps']:.0e}  M~{r['M']:5d}  ell={r['ls']:.3f}  "
              f"var={r['var']:.3f}  wall={r['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
