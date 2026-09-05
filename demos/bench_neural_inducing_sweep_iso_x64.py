"""ISOTROPIC-kernel re-run of demos/bench_neural_inducing_sweep_x64.py.

The ARD neural sweep found SparseGP converges (as #inducing -> 144) to a
DIFFERENT optimum than EFGP (ell~3.08, sigma^2~0.15 vs EFGP 3.53, 0.29), and
the EFGP Fourier-mode sweep ruled out EFGP spectral looseness as the cause.
The remaining prime suspect (cf. the gpdrift study, memory
sing-inducing-sweep-isotropic-resolves-ard) is the KERNEL PARAMETRISATION:
bench.fit_sparsegp learns an **ARD** RBF (`length_scales` shape (D,), D
independent lengthscales) while EFGP uses a single **ISOTROPIC** lengthscale.
The extra ARD degrees of freedom let SparseGP wander to a different (mean-ell,
sigma^2) optimum.

This script re-runs the exact neural inducing sweep with SparseGP constrained
to ONE shared (isotropic) lengthscale via IsotropicRBF — an apples-to-apples
match to EFGP's isotropic SE kernel. Everything else (data, inputs + learned B,
PCA-fixed emissions, schedule, inducing layout, plots) is identical to the ARD
neural sweep; only the SparseGP kernel changes.

Deliverables (in demos/_bench_neural_inducing_sweep_iso_out/): same schema as
the ARD run — sweep.npz, convergence.png, hyper_trajectory_plane.png,
line_attractor_grid.png.

Run on swl1 (demos/bench_neural_inducing_sweep_iso.sbatch).
"""
from __future__ import annotations

import math
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
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# Reuse ALL data / EFGP-fit / plot machinery from the head-to-head demo.
import demos.bench_neural_efgp_vs_sparsegp as bench
from sing.likelihoods import Gaussian
from sing.inputs import InputSignals
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em

D = bench.D
NUM_PER_DIM = [5, 6, 8, 10, 12]           # -> 25, 36, 64, 100, 144 inducing
OUT_DIR = _SING / "demos" / "_bench_neural_inducing_sweep_iso_out"
OUT_DIR.mkdir(exist_ok=True)
ARD_NPZ = _SING / "demos" / "_bench_neural_inducing_sweep_out" / "sweep.npz"


class IsotropicRBF(RBF):
    """RBF with a single SHARED (isotropic) lengthscale (cf. memory
    sing-inducing-sweep-isotropic-resolves-ard / bench_gpdrift_inducing_sweep_iso).

    kernel_params carries a scalar `length_scale`; internally broadcast to a
    (D,) `length_scales` vector so every closed-form RBF expectation works
    unchanged. The optimised pytree leaf is one scalar, so the SING M-step ties
    all D dims together — the same isotropic constraint EFGP imposes.
    """

    def _expand(self, kernel_params):
        # Idempotent: base RBF.E_dKzxdx re-dispatches to self.E_Kxz with
        # already-expanded params, so pass those straight through.
        if "length_scale" not in kernel_params:
            return kernel_params
        ls = kernel_params["length_scale"] * jnp.ones(self.latent_dim)
        return {"length_scales": ls, "output_scale": kernel_params["output_scale"]}

    def K(self, x1, x2, kernel_params):
        return super().K(x1, x2, self._expand(kernel_params))

    def E_Kxx(self, expectation, key, m, S, kernel_params):
        return super().E_Kxx(expectation, key, m, S, self._expand(kernel_params))

    def E_Kxz(self, expectation, key, z, m, S, kernel_params, **kwargs):
        return super().E_Kxz(expectation, key, z, m, S,
                             self._expand(kernel_params), **kwargs)

    def E_KzxKxz(self, expectation, key, z1, z2, m, S, kernel_params, **kwargs):
        return super().E_KzxKxz(expectation, key, z1, z2, m, S,
                                self._expand(kernel_params), **kwargs)

    def E_dKzxdx(self, expectation, key, z, m, S, kernel_params, **kwargs):
        return super().E_dKzxdx(expectation, key, z, m, S,
                                self._expand(kernel_params), **kwargs)


def fit_sparsegp_iso(ys, inputs, output_params, x0, t_grid, lo, hi, ls_init,
                     num_per_dim=8):
    """bench.fit_sparsegp but with the ISOTROPIC kernel: a single scalar
    `length_scale` in drift_params (learned as one shared ell) instead of a
    (D,) ARD `length_scales` vector. Same inputs + learned-B path as the demo."""
    T = ys.shape[1]
    lik = Gaussian(ys, jnp.ones((1, T), dtype=jnp.float64))
    ip = dict(mu0=x0, V0=jnp.eye(D)[None])
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = bench._data_aware_zs(num_per_dim, lo, hi)
    print(f"  [SparseGP-iso] {zs.shape[0]} inducing pts "
          f"({num_per_dim}x{num_per_dim} over latent bbox)", flush=True)
    fn = SparseGP(zs=zs, kernel=IsotropicRBF(latent_dim=D), expectation=quad)
    # Single scalar length_scale (isotropic). output_scale = sqrt(variance).
    drift_params = dict(length_scale=jnp.asarray(float(ls_init)),
                        output_scale=jnp.asarray(math.sqrt(bench.VAR_INIT)))
    rho = jnp.linspace(0.05, 0.7, bench.N_EM)
    lr = jnp.full((bench.N_EM,), bench.MSTEP_LR)
    hist = []
    t0 = time.time()
    (mp, _, gp_post, dp, _, op, input_effect, elbos) = fit_variational_em(
        key=jr.PRNGKey(33), fn=fn, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params, init_params=ip,
        output_params=dict(output_params),
        input_signals=InputSignals(inputs),
        sigma=bench.SIGMA, rho_sched=rho,
        n_iters=bench.N_EM, n_iters_e=bench.N_ESTEP, n_iters_m=bench.N_MSTEP_INNER,
        perform_m_step=True, learn_output_params=False,
        learning_rate=lr, print_interval=5,
        drift_params_history=hist)
    wall = time.time() - t0
    ls_hist = [float(d['length_scale']) for d in hist]         # scalar, no geo-mean
    var_hist = [float(d['output_scale']) ** 2 for d in hist]
    return mp, fn, dp, gp_post, np.asarray(input_effect), elbos, ls_hist, var_hist, wall


def main():
    print(f"\n=== ISOTROPIC inducing sweep: EFGP vs SparseGP(iso RBF), neural ===")
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

    # ---- EFGP once (reference; already isotropic) ----
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

    # ---- SparseGP (isotropic) sweep ----
    sp = []
    for npd in NUM_PER_DIM:
        n_ind = npd * npd
        print(f"\n  [SparseGP-iso] num_per_dim={npd} ({n_ind} inducing) ...", flush=True)
        (mp_s, fn_s, dp_s, gp_s, B_s, elbos_s,
         ls_h, var_h, wall_s) = fit_sparsegp_iso(
            ys, inputs, output_params, x0, t_grid, lo, hi, ls_init,
            num_per_dim=npd)
        drift_fn = (lambda fn_s, dp_s, gp_s: (
            lambda X: bench.sparsegp_drift_field(fn_s, dp_s, gp_s, X)))(fn_s, dp_s, gp_s)
        sp.append(dict(
            npd=npd, n_ind=n_ind, wall=wall_s,
            ls=float(ls_h[-1]), var=float(var_h[-1]),
            ls_traj=np.asarray(ls_h), var_traj=np.asarray(var_h),
            B=np.asarray(B_s), m=np.asarray(mp_s['m'][0]), drift_fn=drift_fn))
        print(f"    SparseGP-iso wall={wall_s:.1f}s  ell={ls_h[-1]:.3f}  "
              f"var={var_h[-1]:.3f}", flush=True)

    # ---- shared slow-point eps (from EFGP median drift speed) ----
    gx = np.linspace(lo[0], hi[0], 30); gy = np.linspace(lo[1], hi[1], 30)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    sp_scale = float(np.median(np.linalg.norm(efgp_fn(pts), axis=-1)))
    eps_slow = max(0.1 * sp_scale, 1e-3)
    print(f"\n  slow-point eps={eps_slow:.3f}", flush=True)

    # ---- ARD reference (from the earlier anisotropic run), if present ----
    ard_ls = ard_var = None
    if ARD_NPZ.exists():
        da = np.load(ARD_NPZ)
        ard_ls = np.asarray(da['ls_sparsegp']); ard_var = np.asarray(da['var_sparsegp'])
        print(f"  ARD SparseGP (for contrast): ell={ard_ls}  var={ard_var}", flush=True)

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
        ard_ls_sparsegp=(np.array([]) if ard_ls is None else ard_ls),
        ard_var_sparsegp=(np.array([]) if ard_var is None else ard_var),
        lo=lo, hi=hi, onsets=np.asarray(onsets), eps_slow=eps_slow,
        ls_init=ls_init, var_init=bench.VAR_INIT)

    n_inds = np.asarray([s['n_ind'] for s in sp])

    # ---- figure 1: convergence vs #inducing (iso, with EFGP ref + ARD contrast) ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(n_inds, [s['ls'] for s in sp], '-s', color='tab:green',
               label='SparseGP (iso)')
    if ard_ls is not None and ard_ls.size == len(n_inds):
        ax[0].plot(n_inds, ard_ls, '-^', color='tab:red', alpha=0.6,
                   label='SparseGP (ARD)')
    ax[0].axhline(ls_e, color='tab:blue', ls='--', label=f'EFGP ref ({ls_e:.2f})')
    ax[0].set_xlabel('# inducing'); ax[0].set_ylabel(r'recovered $\ell$')
    ax[0].set_title(r'lengthscale $\ell$'); ax[0].legend(fontsize=8)
    ax[1].plot(n_inds, [s['var'] for s in sp], '-s', color='tab:green',
               label='SparseGP (iso)')
    if ard_var is not None and ard_var.size == len(n_inds):
        ax[1].plot(n_inds, ard_var, '-^', color='tab:red', alpha=0.6,
                   label='SparseGP (ARD)')
    ax[1].axhline(var_e, color='tab:blue', ls='--', label=f'EFGP ref ({var_e:.2f})')
    ax[1].set_xlabel('# inducing'); ax[1].set_ylabel(r'recovered $\sigma_f^2$')
    ax[1].set_title(r'variance $\sigma_f^2$'); ax[1].legend(fontsize=8)
    ax[2].semilogy(n_inds, [s['wall'] for s in sp], '-s', color='tab:green',
                   label='SparseGP (iso)')
    ax[2].axhline(wall_e, color='tab:blue', ls='--', label=f'EFGP ({wall_e:.0f}s)')
    ax[2].set_xlabel('# inducing'); ax[2].set_ylabel('wall (s)')
    ax[2].set_title('wall-clock'); ax[2].legend(fontsize=8)
    fig.suptitle('ISOTROPIC SparseGP -> EFGP convergence vs # inducing points',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- figure 2 (PRIMARY): (ell, sigma_f^2)-plane EM trajectories ----
    fig, axp = plt.subplots(figsize=(7.2, 6.0))
    colors = cm.viridis(np.linspace(0.15, 0.9, len(sp)))
    for s, c in zip(sp, colors):
        axp.plot(s['ls_traj'], s['var_traj'], '-', color=c, lw=1.3, alpha=0.9)
        axp.plot(s['ls_traj'][0], s['var_traj'][0], 'o', color=c, ms=4)
        axp.plot(s['ls_traj'][-1], s['var_traj'][-1], '*', color=c, ms=15,
                 label=f"SP-iso {s['n_ind']}")
    axp.plot(ls_traj_e, var_traj_e, '-', color='tab:blue', lw=2.2, alpha=0.95)
    axp.plot(ls_traj_e[-1], var_traj_e[-1], '*', color='tab:blue', ms=20,
             markeredgecolor='k', label='EFGP (ref)')
    if ard_ls is not None and ard_ls.size:
        axp.plot(ard_ls[-1], ard_var[-1], 'X', color='tab:red', ms=14, mew=2,
                 label='SP-ARD 144 (old)')
    axp.plot(ls_init, bench.VAR_INIT, 'kx', ms=11, mew=2.5, label='init')
    axp.set_xlabel(r'lengthscale $\ell$'); axp.set_ylabel(r'variance $\sigma_f^2$')
    axp.set_title('Isotropic SparseGP EM trajectories in the '
                  r'$(\ell,\sigma_f^2)$ plane'
                  '\nstars = converged; do iso endpoints land on EFGP?')
    axp.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hyper_trajectory_plane.png", dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- figure 3: line-attractor panel grid (EFGP + each inducing count) ----
    panels = [("EFGP (ref)", efgp_fn, m_e, B_e,
               f"EFGP\nell={ls_e:.2f} var={var_e:.2f} {wall_e:.0f}s")]
    for s in sp:
        panels.append((f"{s['n_ind']} ind", s['drift_fn'], s['m'], s['B'],
                       f"SP-iso {s['n_ind']}\nell={s['ls']:.2f} "
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
    fig.suptitle('Neural line-attractor: ISOTROPIC SparseGP -> EFGP\n'
                 'blue=latents  black=drift  orange=input B', fontsize=12)
    fig.savefig(OUT_DIR / "line_attractor_grid.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/{{sweep.npz, convergence.png, "
          f"hyper_trajectory_plane.png, line_attractor_grid.png}}")
    print("\n  === SUMMARY (ISOTROPIC SparseGP) ===")
    print(f"  EFGP         ell={ls_e:.3f}  var={var_e:.3f}  wall={wall_e:.1f}s")
    for s in sp:
        print(f"  SP-iso {s['n_ind']:4d}  ell={s['ls']:.3f}  var={s['var']:.3f}  "
              f"wall={s['wall']:.1f}s", flush=True)
    if ard_ls is not None and ard_ls.size:
        print(f"  [contrast] ARD SP 144  ell={float(ard_ls[-1]):.3f}  "
              f"var={float(ard_var[-1]):.3f}")


if __name__ == "__main__":
    main()
