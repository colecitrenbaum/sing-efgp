"""
diag_mstep_landscape.py

Isolate the SparseGP lengthscale bias by comparing M-step OBJECTIVE surfaces at
a FIXED, shared q(x) — removing the E-step-feedback confound.

Only neg_CE(θ) + prior_term(θ) depend on the drift hypers θ=(ℓ,σ_f²)
(elbo = ell + entropy + neg_CE + prior).  We evaluate SparseGP's drift objective
    J_sp(θ; q(x), M) = Σ_i neg_CE_single(θ) + prior_term(θ)      [maximize]
on an (ℓ,σ²) grid at a fixed q(x), for several inducing counts M and several q(x)
variants, and locate its argmax.  Compared against the oracle pseudo-velocity
marginal-likelihood (gt_landscape, argmin at the MLE).

q(x) variants:
  efgp   : EFGP's converged posterior (realistic S_i)
  oracle : true latents with S→0 (no posterior input uncertainty)
  S-scaling sweep on the efgp q(x): S_i × {0, 0.25, 1.0}

Reads out, per (M, variant):  argmax (ℓ, σ²) of J_sp.

Logic:
  - If S→0 collapses argmax-ℓ to the oracle but realistic S does not
      ⇒ bias is the uncertain-input Ψ-statistics (E_Kxz / E_KzxKxz smearing).
  - If argmax-ℓ stays high even at S→0 and large M
      ⇒ bias is the inducing-variational bound itself (trace / KL slack),
        which EFGP (exact Fourier evidence) structurally lacks.

Run under Slurm (demos/diag_mstep_landscape.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax.* (CLAUDE.md)

import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import jax.random as jr

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep   # fit_efgp (returns hist+mp)
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.utils.sing_helpers import compute_neg_CE

D = bench.D
LS_INIT = 0.7
M_LIST = [25, 256]
S_SCALES = [0.0, 0.25, 1.0]          # multiply EFGP q(x) covariance S_i
S_FLOOR = 1e-4                        # jitter floor so S stays non-singular
N_GRID = 15
LOG_LS = np.linspace(-2.5, 2.0, N_GRID)
LOG_VAR = np.linspace(-2.5, 3.0, N_GRID)

OUT_DIR = _ROOT / "demos" / "_bench_gpdrift_inducing_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


def make_sparse(M):
    n_per = int(round(math.sqrt(M)))
    xs = _XS_NP
    zs = bench._data_aware_zs(n_per, xs)
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    return SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)


def build_objective(sparse, t_grid, init_params, inputs, input_effect,
                    trial_mask, sigma):
    """Returns jitted L(log_ls, log_var, marginal_params) = -(neg_CE + prior),
    the SparseGP drift objective as an NLL (argmin = M-step optimum)."""
    B = 1
    key0 = jr.PRNGKey(0)

    @jax.jit
    def L(log_ls, log_var, marginal_params):
        dp = dict(length_scales=jnp.full((D,), jnp.exp(log_ls)),
                  output_scale=jnp.exp(0.5 * log_var))   # RBF stores sqrt(var)
        gp_post = sparse.update_dynamics_params(
            key0, t_grid, marginal_params, trial_mask, dp, inputs,
            input_effect, sigma)
        neg_CE = jax.vmap(partial(compute_neg_CE, t_grid, sparse, gp_post, dp,
                                  input_effect=input_effect, sigma=sigma))(
            init_params, jr.split(key0, B), marginal_params, inputs,
            trial_mask).sum()
        prior = sparse.prior_term(dp, gp_post)
        return -(neg_CE + prior)     # maximize elbo ⇔ minimize this
    return L


def landscape(L, marginal_params):
    grid = np.zeros((N_GRID, N_GRID))
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            v = float(L(jnp.asarray(ll), jnp.asarray(lv), marginal_params))
            grid[i, j] = v if np.isfinite(v) else np.nan
    return grid


def argmin_lsvar(grid):
    if not np.any(np.isfinite(grid)):
        return float('nan'), float('nan')
    k = np.unravel_index(np.nanargmin(grid), grid.shape)
    return float(LOG_LS[k[0]]), float(LOG_VAR[k[1]])


def main():
    global _XS_NP
    print(f"diag_mstep_landscape: M={M_LIST} S_scales={S_SCALES} "
          f"grid={N_GRID}²  devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)
    _XS_NP = xs_np
    T = xs_np.shape[0]
    trial_mask = jnp.ones((1, T), dtype=bool)
    inputs = jnp.zeros((1, T, 1))
    input_effect = jnp.zeros((D, 1))

    # oracle reference (argmin = MLE)
    L_gt = bench.gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gt_ll, gt_lv = argmin_lsvar(L_gt)
    print(f"  oracle GT MLE: ℓ={math.exp(gt_ll):.3f}  σ²={math.exp(gt_lv):.3f}",
          flush=True)

    # EFGP converged q(x) (realistic S) — one fit, reused for all M / S-scales
    print("  EFGP fit (for its converged q(x))...", flush=True)
    e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
    print(f"    EFGP recovered ℓ={e['ls']:.3f}  σ²={e['var']:.3f}", flush=True)
    m_efgp = e['mp']['m']      # (1, T, D)
    S_efgp = e['mp']['S']      # (1, T, D, D)
    SS_efgp = e['mp']['SS']    # (1, T-1, D, D)

    floor = S_FLOOR * jnp.eye(D)
    def mp_efgp_scaled(s_scale):
        # floor keeps S non-singular (E_KzxKxz does solve(S, ·)); s_scale=0
        # then means "point inputs" without an exact-zero covariance.
        return dict(m=m_efgp, S=S_efgp * s_scale + floor, SS=SS_efgp)

    # oracle-input q(x): true latents, S→0 (floored), SS = x_{i+1} x_iᵀ
    S_orc = jnp.broadcast_to(floor, (1, T, D, D))
    SS_orc = (xs[1:, :, None] * xs[:-1, None, :])[None]    # (1, T-1, D, D)
    mp_oracle = dict(m=xs[None], S=S_orc, SS=SS_orc)

    # DEBIASING TARGET: exact pseudo-velocity GP evidence (no inducing, no
    # variational bound) evaluated on the SAME q(x) means EFGP converged to.
    # This is what the user's proposed exact-evidence M-step would maximize.
    L_exact = bench.gt_landscape(jnp.asarray(m_efgp[0]), sigma, t_grid,
                                 LOG_LS, LOG_VAR)
    ex_ll, ex_lv = argmin_lsvar(L_exact)
    print(f"  EXACT evidence on EFGP q(x) means: argmax ℓ={math.exp(ex_ll):.3f} "
          f"σ²={math.exp(ex_lv):.3f}  (this is the debiasing target)", flush=True)

    results = {}
    for M in M_LIST:
        sparse = make_sparse(M)
        L = build_objective(sparse, t_grid, ip, inputs, input_effect,
                            trial_mask, sigma)
        print(f"\n  === M={M} ===", flush=True)

        # oracle inputs (S→0)
        t0 = time.perf_counter()
        L_orc = landscape(L, mp_oracle)
        a_ll, a_lv = argmin_lsvar(L_orc)
        print(f"    [oracle inputs, S→0]  argmax ℓ={math.exp(a_ll):.3f}  "
              f"σ²={math.exp(a_lv):.3f}   ({time.perf_counter()-t0:.0f}s)",
              flush=True)
        results[(M, 'oracle')] = dict(grid=L_orc, ll=a_ll, lv=a_lv)

        # EFGP q(x) at several S scalings
        for sc in S_SCALES:
            t0 = time.perf_counter()
            L_e = landscape(L, mp_efgp_scaled(sc))
            a_ll, a_lv = argmin_lsvar(L_e)
            print(f"    [EFGP q(x), S×{sc:>4}]  argmax ℓ={math.exp(a_ll):.3f}  "
                  f"σ²={math.exp(a_lv):.3f}   ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
            results[(M, f'efgp_s{sc}')] = dict(grid=L_e, ll=a_ll, lv=a_lv)

    # ---- save ----
    save = dict(LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
                gt_ll=gt_ll, gt_lv=gt_lv, xs_np=xs_np,
                efgp_ls=e['ls'], efgp_var=e['var'])
    for (M, tag), r in results.items():
        save[f'M{M}_{tag}_grid'] = r['grid']
        save[f'M{M}_{tag}_ll'] = r['ll']
        save[f'M{M}_{tag}_lv'] = r['lv']
    np.savez(OUT_DIR / 'mstep_landscape.npz', **save)
    print(f"\n  saved {OUT_DIR / 'mstep_landscape.npz'}", flush=True)

    # ---- plot: rows = M, cols = [oracle-S0, EFGP S×0, S×0.25, S×1.0] ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cols = [('oracle', 'oracle in, S→0')] + \
           [(f'efgp_s{sc}', f'EFGP q(x), S×{sc}') for sc in S_SCALES]
    nrow, ncol = len(M_LIST), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for r_i, M in enumerate(M_LIST):
        for c_i, (tag, title) in enumerate(cols):
            ax = axes[r_i][c_i]
            g = results[(M, tag)]['grid']
            gn = g - np.nanmin(g)
            fin = gn[np.isfinite(gn) & (gn > 0)]
            if fin.size:
                lv = list(np.logspace(np.log10(max(fin.min(), fin.max() / 1e3)),
                                      np.log10(fin.max()), 10))
            else:
                lv = [1, 10, 100]
            ax.contourf(LOG_LS, LOG_VAR, gn.T, levels=[0] + lv,
                        cmap='viridis_r', extend='max')
            ax.contour(LOG_LS, LOG_VAR, gn.T, levels=lv, colors='k',
                       linewidths=0.3)
            a_ll, a_lv = results[(M, tag)]['ll'], results[(M, tag)]['lv']
            ax.scatter([a_ll], [a_lv], marker='s', s=120, color='red',
                       edgecolor='k', zorder=6,
                       label=f'SP argmax ℓ={math.exp(a_ll):.2f}')
            ax.scatter([gt_ll], [gt_lv], marker='*', s=220, color='gold',
                       edgecolor='k', zorder=7,
                       label=f'oracle MLE ℓ={math.exp(gt_ll):.2f}')
            ax.scatter([math.log(e['ls'])], [math.log(e['var'])], marker='o',
                       s=90, color='C0', edgecolor='k', zorder=7,
                       label=f'EFGP fit ℓ={e["ls"]:.2f}')
            ax.set_title(f"M={M}  |  {title}", fontsize=9)
            ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
            ax.legend(fontsize=6, loc='lower right')
    fig.suptitle("SparseGP M-step objective argmax at FIXED q(x) — "
                 "isolating the ℓ bias (E-step feedback removed)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / 'mstep_landscape.png', dpi=125)
    print(f"  saved {OUT_DIR / 'mstep_landscape.png'}", flush=True)


if __name__ == '__main__':
    main()
