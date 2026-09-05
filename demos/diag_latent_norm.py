"""
diag_latent_norm.py

Direct test of the "are EFGP and SparseGP on different latent normalizations?"
hypothesis for the well-specified GP-drift problem (same data as
bench_gpdrift_inducing_sweep_x64).

If SparseGP's inferred latents were globally scaled by a factor c relative to
EFGP / the true latents, then:
  - the Procrustes map A (inferred -> true) would have singular values ≈ 1/c
    (NOT ≈ 1),
  - raw (unaligned) latent RMSE would be large while aligned (pc) RMSE stays
    tiny,
  - std(inferred latents) / std(true latents) ≈ c,
and the recovered lengthscale ℓ would scale by c.  This script prints all of
those per method so the normalization explanation can be confirmed or ruled out.

Reuses the exact matched-hyper fits from the sweep (fit_efgp / bench.fit_sparsegp).

Run under Slurm (demos/diag_latent_norm.sbatch), NOT the login node.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax.* (CLAUDE.md)

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import demos.bench_gpdrift_x64 as bench
import demos.bench_gpdrift_inducing_sweep_x64 as sweep   # local fit_efgp / eval_efgp_drift

LS_INIT = 0.7
M_DIAG = [25, 256]


def _report(name, m_inf, xs_np, ls, var, drift_pred_fn, drift_true_fn,
            grid_pts):
    """Print latent-normalization diagnostics for one method."""
    m_inf = np.asarray(m_inf)
    # raw vs Procrustes-aligned latent RMSE
    raw = float(np.sqrt(np.mean((m_inf - xs_np) ** 2)))
    A, b = bench.procrustes_align(m_inf, xs_np)         # x_true ≈ A m_inf + b
    m_aligned = m_inf @ A.T + b
    pc = float(np.sqrt(np.mean((m_aligned - xs_np) ** 2)))
    sv = np.linalg.svd(A, compute_uv=False)             # scale factors of A
    A_off = float(np.linalg.norm(A - np.eye(A.shape[0])))
    # per-dim std ratio inferred/true (scale of latents)
    std_inf = m_inf.std(0)
    std_true = xs_np.std(0)
    scale_ratio = std_inf / std_true
    # drift magnitude ratio on the eval grid
    fp = np.asarray(drift_pred_fn(grid_pts))
    ft = np.asarray([np.asarray(drift_true_fn(np.asarray(p), 0.))
                     for p in grid_pts])
    rms_pred = float(np.sqrt(np.mean(fp ** 2)))
    rms_true = float(np.sqrt(np.mean(ft ** 2)))

    print(f"\n  === {name} ===")
    print(f"    recovered   ℓ={ls:.3f}  σ²={var:.3f}")
    print(f"    latent RMSE raw={raw:.4f}  pc={pc:.4f}  "
          f"(raw>>pc ⇒ a linear map is needed ⇒ possible scale/rotation)")
    print(f"    Procrustes A singular values (inferred→true) = "
          f"[{sv[0]:.4f}, {sv[1]:.4f}]  (≈1 ⇒ no rescale; ≈1/c ⇒ latents ×c)")
    print(f"    ||A - I|| = {A_off:.4f}")
    print(f"    std(latents) per dim: inferred={std_inf.round(3)}  "
          f"true={std_true.round(3)}  ratio={scale_ratio.round(3)}")
    print(f"    drift RMS magnitude: pred={rms_pred:.3f}  true={rms_true:.3f}  "
          f"ratio={rms_pred / max(rms_true, 1e-12):.3f}", flush=True)
    return dict(name=name, ls=ls, var=var, raw=raw, pc=pc, sv=sv, A=A,
                scale_ratio=scale_ratio)


def main():
    print(f"diag_latent_norm: ls_init={LS_INIT}  M={M_DIAG}  "
          f"devices={jax.devices()}", flush=True)
    xs, lik, op, ip, t_grid, sigma, drift_fn, X_grid, alpha = bench.make_data()
    xs_np = np.asarray(xs)

    # eval grid (same as compute_drift_metrics)
    lo = xs_np.min(0) - 0.4; hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14); g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)

    print(f"  true-latent std per dim = {xs_np.std(0).round(3)}", flush=True)

    # EFGP
    print("\n  EFGP fit...", flush=True)
    e = sweep.fit_efgp(lik, op, ip, t_grid, sigma, LS_INIT)
    _report("EFGP", e['mp']['m'][0], xs_np, e['ls'], e['var'],
            drift_pred_fn=lambda g: sweep.eval_efgp_drift(e['hist'], g),
            drift_true_fn=drift_fn, grid_pts=grid_pts)

    # SparseGP at sparse + dense
    for M in M_DIAG:
        n_per = int(round(math.sqrt(M)))
        print(f"\n  SparseGP M={M} ({n_per}×{n_per}) fit...", flush=True)
        s = bench.fit_sparsegp(lik, op, ip, t_grid, sigma, n_per, LS_INIT,
                               xs_np)
        _report(f"SparseGP M={M}", s['mp']['m'][0], xs_np, s['ls'], s['var'],
                drift_pred_fn=lambda g, s_=s: bench.eval_sp_drift(s_, g),
                drift_true_fn=drift_fn, grid_pts=grid_pts)

    print("\n  INTERPRETATION:")
    print("   - If all methods show A-singular-values ≈ 1 and std-ratio ≈ 1,")
    print("     latents share the SAME normalization ⇒ the ℓ gap is a genuine")
    print("     SparseGP M-step bias, NOT a normalization artifact.")
    print("   - If SparseGP shows A-sv ≈ 1/1.8 ≈ 0.55 and std-ratio ≈ 1.8,")
    print("     its latents are ~1.8× rescaled ⇒ normalization explains the ℓ gap.")


if __name__ == '__main__':
    main()
