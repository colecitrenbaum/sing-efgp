"""Isolated, warm timing of the EFGP q(f) update (compute_mu_r_*) on GPU.

Compares parallelization of:
  * gmix        — pure-JAX scatter + FFT (current GPU default)
  * analytic-o1 — cuFINUFFT type-1 + order-1 Taylor envelope (current CPU default)
  * analytic-o2 — cuFINUFFT + order-2 Taylor
  * gemm        — dense NUDFT via GEMM, exact-S (Strategy 2)
  * gemm-chunk  — dense NUDFT via chunked GEMM (Strategy 2, memory-bounded)
  * gridnufft   — pure-JAX gridding NUFFT + order-1 Taylor (Strategy 1)

Times a WARM run (post-JIT, median of N_REP calls, block_until_ready) so the
metric is steady-state device throughput — the "does it parallelize" question,
not compile time (reported separately).  Accuracy is reported as mu_r relative
error vs the exact-S dense GEMM.

Run on GPU: python demos/bench_analytic_gpu_parallel.py
"""
from __future__ import annotations
import sys, math, time, gc
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
import sing.efgp_analytic_gpu as gpu

N_REP = 7
HET = 3.0          # per-source S heterogeneity (moderate, realistic mid-fit)


def make_inputs(N_src, het, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * math.pi, N_src)
    R = 1.0 + 0.3 * t / t.max()
    m = np.stack([R * np.cos(t), R * np.sin(t)], axis=-1)
    m += 0.02 * rng.standard_normal(m.shape)
    u = rng.random(N_src)
    base = 0.03 ** 2
    scale = 1.0 + het * u
    S = np.zeros((N_src, 2, 2))
    S[:, 0, 0] = base * scale
    S[:, 1, 1] = base * scale * (1.0 + 0.3 * het * rng.random(N_src))
    S[:, 0, 1] = S[:, 1, 0] = 0.2 * np.sqrt(S[:, 0, 0] * S[:, 1, 1]) * rng.random(N_src)
    d = 0.05 * rng.standard_normal((N_src, 2))
    C = 0.01 * rng.standard_normal((N_src, 2, 2))
    w = np.full(N_src, 0.01)
    return (jnp.asarray(m), jnp.asarray(S), jnp.asarray(d),
            jnp.asarray(C), jnp.asarray(w))


def rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def time_fn(fn, *args):
    """Compile+first-call (cold) and warm median over N_REP."""
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return out, cold, float(np.median(ts)) * 1e3   # warm in ms


def main():
    print(f"device={jax.devices()}  backend={jax.default_backend()}  "
          f"x64={jax.config.read('jax_enable_x64')}", flush=True)
    var, sds = 1.0, 0.1 ** 2
    common = dict(sigma_drift_sq=sds, D_lat=2, D_out=2)

    # (ls, N_src) configs.  Two ls values probe M-scaling (smaller ls → larger
    # spectral grid M); the N sweep probes source-count scaling.  Dense GEMM is
    # O(M·N) in memory+compute; gmix / NUFFT / sepgemm are ~O(N + M log M) /
    # O(√M·N), so the crossover should appear as N and M grow.
    LS_LIST = [0.4, 0.2]
    N_LIST = [2000, 10000, 50000, 100000]
    configs = [(ls, N) for ls in LS_LIST for N in N_LIST]
    for ls, N_src in configs:
        m, S, d, C, w = make_inputs(N_src, HET)
        grid = jp.spectral_grid_se(ls, var, m, eps=1e-3)
        M = grid.M
        Ld = (2 * (grid.mtot_per_dim[0] - 1) + 1)
        h_spec = float(grid.h_per_dim[0])
        sig_max = float(np.sqrt(np.linalg.eigvalsh(np.asarray(S)).max()))
        m_extent = float((np.asarray(m).max(0) - np.asarray(m).min(0)).max())
        fine_N = pick_grid_size(h_spec=h_spec, m_extent=m_extent,
                                sigma_max=sig_max, n_resolve=4.0)
        h_grid = 1.0 / (fine_N * h_spec)
        stencil_r = int(stencil_radius_for(S, h_grid, n_sigma=1.5))

        print(f"\n=== ls={ls} N_src={N_src}  M={M} (mtot={grid.mtot_per_dim[0]}, "
              f"Ld={Ld})  fine_N={fine_N} stencil_r={stencil_r} ===", flush=True)

        methods = {
            'gmix': jax.jit(lambda m, S, d, C, w: jpd.compute_mu_r_gmix_jax(
                m, S, d, C, w, grid, fine_N=fine_N, stencil_r=stencil_r, **common)),
            'analytic-o1': jax.jit(lambda m, S, d, C, w: jpd.compute_mu_r_analytic_jax(
                m, S, d, C, w, grid, order=1, **common)),
            'analytic-o2': jax.jit(lambda m, S, d, C, w: jpd.compute_mu_r_analytic_jax(
                m, S, d, C, w, grid, order=2, **common)),
            'gemm': jax.jit(lambda m, S, d, C, w: gpu.compute_mu_r_densegemm_jax(
                m, S, d, C, w, grid, **common)),
            'gemm-chunk': jax.jit(lambda m, S, d, C, w: gpu.compute_mu_r_densegemm_jax(
                m, S, d, C, w, grid, freq_chunk=4096, **common)),
            'sepgemm-x1': jax.jit(lambda m, S, d, C, w: gpu.compute_mu_r_sepgemm_jax(
                m, S, d, C, w, grid, xorder=1, **common)),
            'sepgemm-x2': jax.jit(lambda m, S, d, C, w: gpu.compute_mu_r_sepgemm_jax(
                m, S, d, C, w, grid, xorder=2, **common)),
            'gridnufft': jax.jit(lambda m, S, d, C, w: gpu.compute_mu_r_gridnufft_jax(
                m, S, d, C, w, grid, order=1, **common)),
        }

        ref = None
        results = {}
        for name, fn in methods.items():
            try:
                out, cold, warm = time_fn(fn, m, S, d, C, w)
                mu = out[0]
                results[name] = (cold, warm, mu)
                if name == 'gemm' or (name == 'gemm-chunk' and ref is None):
                    ref = mu  # exact-S reference (prefer dense; fall back to chunked)
            except Exception as e:
                results[name] = (None, None, None)
                print(f"  {name:12s}: FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
            gc.collect()

        print(f"  {'method':12s} {'cold(s)':>8s} {'warm(ms)':>9s} "
              f"{'speedup':>8s} {'relerr':>9s}", flush=True)
        base_warm = results['gmix'][1] if results.get('gmix') else None
        for name, (cold, warm, mu) in results.items():
            if warm is None:
                continue
            sp = (base_warm / warm) if base_warm else float('nan')
            re = rel(mu, ref) if (ref is not None and mu is not None) else float('nan')
            print(f"  {name:12s} {cold:8.2f} {warm:9.2f} {sp:7.2f}x {re:9.2e}",
                  flush=True)


if __name__ == '__main__':
    main()
