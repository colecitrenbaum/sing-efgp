"""
bench_gpdrift_hybrid_mstep_x64.py

Test the debiasing recipe: keep the SparseGP E-step (inducing-point q(f),
natural-grad q(x)) but replace the M-step objective for the drift hypers with
the (near-)EXACT GP marginal likelihood of the pseudo-velocities given the
current q(x) means — instead of the variational ELBO lower bound.

Rationale (see analysis): max_θ ELBO is a biased θ-estimator because the bound
gap G(θ)=KL[q‖p(·|θ)] is θ-dependent and rewards longer ℓ.  EFGP avoids this by
marginalising f (near-)exactly (Fourier).  Here we do the dense analog: at each
M-step, maximise
    log p(v | θ, m) ,   v_i = (m_{i+1}-m_i)/Δt ,  v_i | m_i ~ N(f(m_i), (σ²/Δt)I)
    f ~ GP(0, σ_f² K_RBF(·;ℓ))
via dense Cholesky on the current q(x) means m_i.  This is exactly the oracle
objective form (bench.gt_landscape) but on the inferred means rather than the
true latents.  The E-step, q(u), inducing points are UNCHANGED.

Hooked in via fit_variational_em(..., drift_loss_fn=exact_evidence_loss).

Sweep M ∈ {25, 64, 256}; for each, run BOTH the standard (ELBO M-step) and the
hybrid (exact-evidence M-step) SparseGP, and compare recovered (ℓ,σ²) to the
oracle MLE and EFGP.

Run under Slurm (demos/bench_gpdrift_hybrid_mstep.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax.* (CLAUDE.md)

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.scipy.linalg as jsla
import jax.random as jr

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep   # fit_efgp
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em

D = bench.D
T = bench.T
LS_INIT = 0.7
M_LIST = [25, 64, 256]


def exact_evidence_loss(drift_params, marginal_params, t_grid, sigma):
    """(Near-)exact pseudo-velocity GP marginal NLL on the current q(x) means.

    This REPLACES the ELBO in the drift M-step.  Anisotropic RBF matching
    sing.kernels.RBF (output_scale is the sqrt of the variance).  Single trial.
    """
    ls = drift_params['length_scales']              # (D,)
    var = drift_params['output_scale'] ** 2         # scalar
    m = marginal_params['m'][0]                     # (T, D)
    dt = t_grid[1] - t_grid[0]
    Xin = m[:-1]                                    # (n, D)
    v = (m[1:] - m[:-1]) / dt                       # (n, D) pseudo-velocities
    n = Xin.shape[0]
    diffs = (Xin[:, None, :] - Xin[None, :, :]) / ls
    Krbf = jnp.exp(-0.5 * (diffs ** 2).sum(-1))     # (n, n)
    noise_var = sigma ** 2 / dt
    A = var * Krbf + noise_var * jnp.eye(n)
    L = jnp.linalg.cholesky(A)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    alpha = jsla.cho_solve((L, True), v)            # (n, D)
    quad = jnp.sum(v * alpha)                       # Σ_d v_dᵀ A⁻¹ v_d
    D_out = m.shape[1]
    return 0.5 * quad + 0.5 * D_out * logdet


def fit_sparsegp(lik, op, ip, t_grid, sigma, num_per_dim, ls_init, xs_np,
                 use_exact):
    """SparseGP-SING fit.  use_exact=True swaps in the exact-evidence M-step;
    everything else (E-step, q(u), inducing, matched schedule) is identical."""
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = bench._data_aware_zs(num_per_dim, xs_np)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(ls_init)),
                         output_scale=jnp.asarray(math.sqrt(bench.VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, bench.N_EM)
    history = []
    dloss = exact_evidence_loss if use_exact else None
    t0 = time.perf_counter()
    mp, _, gp_post, dp, *_ = fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=bench.N_EM, n_iters_e=10,
        n_iters_m=bench.N_M_INNER, perform_m_step=True,
        learn_output_params=False,
        learning_rate=jnp.full((bench.N_EM,), bench.MSTEP_LR),
        print_interval=999, drift_params_history=history,
        drift_loss_fn=dloss)
    wall = time.perf_counter() - t0
    ls_traj = np.array([ls_init] +
                       [float(np.mean(h['length_scales'])) for h in history])
    var_traj = np.array([bench.VAR_INIT] +
                        [float(h['output_scale']) ** 2 for h in history])
    return dict(mp=mp, sd=sparse, gp_post=gp_post, dp=dp,
                ls_traj=ls_traj, var_traj=var_traj,
                ls=float(jnp.mean(dp['length_scales'])),
                var=float(dp['output_scale']) ** 2, wall=wall)


def main():
    print(f"bench_gpdrift_hybrid_mstep: T={T} ls_init={LS_INIT} M={M_LIST}  "
          f"devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)

    LOG_LS = np.linspace(-2.0, 1.5, 21)
    LOG_VAR = np.linspace(-2.5, 2.0, 21)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"  oracle MLE: ℓ={math.exp(gb_ll):.3f}  σ²={math.exp(gb_lv):.3f}  "
          f"(true ℓ={bench.LS_TRUE}, σ²={bench.VAR_TRUE})", flush=True)

    print("  EFGP reference fit...", flush=True)
    e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
    print(f"    EFGP  ℓ={e['ls']:.3f}  σ²={e['var']:.3f}  wall={e['wall']:.0f}s",
          flush=True)

    results = {}
    for M in M_LIST:
        n_per = int(round(math.sqrt(M)))
        for use_exact in (False, True):
            tag = 'hybrid' if use_exact else 'standard'
            print(f"  SP M={M} [{tag}] fit...", flush=True)
            s = fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, LS_INIT,
                             xs_np, use_exact)
            print(f"    ℓ={s['ls']:.3f}  σ²={s['var']:.3f}  wall={s['wall']:.0f}s",
                  flush=True)
            results[(M, tag)] = s

    # ---- plot: trajectories on oracle landscape ----
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
    for ax, tag in zip(axes, ['standard', 'hybrid']):
        ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                    cmap='viridis_r', extend='max')
        ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels, colors='k',
                   linewidths=0.3)
        colors = cm.viridis(np.linspace(0.15, 0.9, len(M_LIST)))
        for c, M in zip(colors, M_LIST):
            s = results[(M, tag)]
            ax.plot(np.log(s['ls_traj']), np.log(s['var_traj']), '-s',
                    color=c, ms=3, lw=1.4,
                    label=f"SP M={M} [ℓ={s['ls']:.2f}, σ²={s['var']:.2f}]")
        ax.plot(np.log(e['ls_traj']), np.log(e['var_traj']), '-o', color='C0',
                ms=3.5, lw=2.2, zorder=8, label=f"EFGP [ℓ={e['ls']:.2f}]")
        ax.scatter([math.log(LS_INIT)], [math.log(bench.VAR_INIT)], marker='+',
                   s=200, color='k', zorder=9)
        ax.scatter([gb_ll], [gb_lv], marker='*', s=240, color='gold',
                   edgecolor='k', zorder=10,
                   label=f"oracle MLE [ℓ={math.exp(gb_ll):.2f}]")
        ax.scatter([math.log(bench.LS_TRUE)], [math.log(bench.VAR_TRUE)],
                   marker='X', s=220, color='magenta', edgecolor='k', zorder=10)
        ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
        ax.set_title(f"{tag} M-step  ({'exact evidence' if tag=='hybrid' else 'variational ELBO'})")
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle("SparseGP hyperparameter recovery: variational-ELBO vs "
                 "exact-evidence M-step (E-step unchanged)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT / 'hybrid_mstep.png', dpi=130)
    print(f"  saved {OUT / 'hybrid_mstep.png'}", flush=True)

    # ---- summary ----
    print(f"\n{'='*80}")
    print(f"SUMMARY  (oracle MLE ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}; "
          f"EFGP ℓ={e['ls']:.3f}, σ²={e['var']:.3f})")
    print(f"{'='*80}")
    print(f"  {'M':>5s}  {'M-step':>10s}  {'ℓ_final':>8s}  {'σ²_final':>9s}  {'wall':>7s}")
    for M in M_LIST:
        for tag in ['standard', 'hybrid']:
            s = results[(M, tag)]
            print(f"  {M:>5d}  {tag:>10s}  {s['ls']:>8.3f}  {s['var']:>9.3f}  "
                  f"{s['wall']:>6.0f}s")


if __name__ == '__main__':
    main()
