"""
diag_kzz_condition.py

Static analysis of cond(Kzz) and cond(A) across the (M, ℓ, σ²) sweep
range, to identify the two NaN failure modes:
  A)  Redundant inducing points (M dense, ℓ large) → cond(Kzz) blows up
  B)  Uncovered inducing points (M sparse, ℓ small, compact attractor)
      → cond(A) blows up because some rows of int_E_KzxKxz are ~0

No EM run, no autodiff — just numpy/jnp linalg on the static matrices.

Run:
  ~/myenv/bin/python demos/diag_kzz_condition.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.simulate_data import simulate_sde
from sing.initialization import initialize_zs


D = 2
JITTER = 1e-3


def kzz_se(zs, ls, var, jitter=JITTER):
    """Squared-exponential Kzz with jitter on diag.  Numpy version."""
    Z = np.asarray(zs)
    M = len(Z)
    sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K = var * np.exp(-0.5 * sq / (ls ** 2))
    K += jitter * np.eye(M)
    return K


def covered_mass(zs, xs, ls):
    """Per-inducing-point 'data weight': sum over t of exp(-||x_t-z_i||²/2ℓ²).
    Inducing pts with low covered_mass are 'invisible' to the data → their
    rows of int_E_KzxKxz are ~0."""
    Z = np.asarray(zs); X = np.asarray(xs)
    sq = ((X[:, None, :] - Z[None, :, :]) ** 2).sum(-1)  # (T, M)
    return np.exp(-0.5 * sq / (ls ** 2)).sum(0)


def simulate_attractor(short, T):
    """Quick reproduction of the bench attractors."""
    sigma = dict(linear=0.3, duffing=0.2, anharmonic=0.3)[short]
    if short == 'linear':
        A = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
        f = lambda x, t: A @ x
        x0 = jnp.array([2.0, 0.0]); seed = 7; t_max_b = 8.0
    elif short == 'duffing':
        f = lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]])
        x0 = jnp.array([1.2, 0.0]); seed = 13; t_max_b = 15.0
    else:  # anharmonic
        f = lambda x, t: jnp.stack([x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3])
        x0 = jnp.array([1.5, 0.0]); seed = 21; t_max_b = 10.0
    t_max = t_max_b * (T / 400.0)
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(seed), x0=x0, f=f, t_max=t_max,
                      n_timesteps=T, sigma=sigma_fn)
    return np.clip(np.asarray(xs), -3.0, 3.0)


def main():
    print(f"Static Kzz / coverage diagnostic (jitter={JITTER:.0e})\n")

    M_LIST = [25, 64, 144]
    LS_LIST = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    var = 1.0

    # ─── Part A: cond(Kzz) vs (M, ℓ) — mechanism A ─────────────────
    print("=" * 80)
    print("Part A: cond(Kzz) vs (M, ℓ).  σ² = 1.0.  Spacing = 5.0/(N-1).")
    print("=" * 80)
    print(f"  {'M':>4s}  {'spacing':>7s}  " +
          "  ".join(f"ℓ={ls:>4.1f}" for ls in LS_LIST))
    for M in M_LIST:
        N = int(round(math.sqrt(M)))
        zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=N)
        spacing = 5.0 / (N - 1)
        conds = []
        for ls in LS_LIST:
            K = kzz_se(zs, ls, var)
            try:
                c = float(np.linalg.cond(K))
            except np.linalg.LinAlgError:
                c = float('inf')
            conds.append(c)
        s = "  ".join(f"{c:>7.1e}" for c in conds)
        print(f"  M={M:>2d}  {spacing:>7.3f}  {s}")
    print()
    print("  Interpretation: at fixed σ², cond(Kzz) explodes as ℓ ↑")
    print("  (off-diagonals → 1).  M=144 hits cond ≳ 1e5 above ℓ ≈ 1.5,")
    print("  which is the float32-Cholesky-gradient danger zone.")

    # ─── Part B: data coverage — mechanism B ───────────────────────
    print()
    print("=" * 80)
    print("Part B: per-inducing-pt data coverage Σ_t exp(-||x_t-z||²/(2ℓ²))")
    print("        for T=2000 trajectories.  Below: # of inducing points")
    print("        with coverage < 1% of the maximum (i.e. effectively")
    print("        invisible to the data).")
    print("=" * 80)
    print(f"  {'bench':<11s}  {'M':>4s}  " +
          "  ".join(f"ℓ={ls:>4.1f}" for ls in LS_LIST))
    for short in ['linear', 'duffing', 'anharmonic']:
        xs = simulate_attractor(short, 2000)
        for M in M_LIST:
            N = int(round(math.sqrt(M)))
            zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=N)
            row = []
            for ls in LS_LIST:
                cov = covered_mass(zs, xs, ls)
                threshold = cov.max() * 0.01
                n_dead = int((cov < threshold).sum())
                row.append(n_dead)
            s = "  ".join(f"{n:>7d}" for n in row)
            print(f"  {short:<11s}  M={M:>2d}  {s}")
    print()
    print("  Interpretation: 'dead' inducing points contribute near-zero")
    print("  rows to A = Kzz + σ⁻² ∫ E[Kzx Kxz], so A is rank-deficient by")
    print("  ~n_dead.  Cholesky gradient through that ill-cond A → NaN.")
    print("  Duffing M=25 has many dead pts (compact attractor); larger M")
    print("  recovers coverage.")

    # ─── Part C: cond(Kzz) for jitter=1e-2 (proposed fix) ─────────
    print()
    print("=" * 80)
    print(f"Part C: cond(Kzz) at jitter=1e-2 (the proposed bump).")
    print("=" * 80)
    print(f"  {'M':>4s}  {'spacing':>7s}  " +
          "  ".join(f"ℓ={ls:>4.1f}" for ls in LS_LIST))
    for M in M_LIST:
        N = int(round(math.sqrt(M)))
        zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=N)
        spacing = 5.0 / (N - 1)
        conds = []
        for ls in LS_LIST:
            K = kzz_se(zs, ls, var, jitter=1e-2)
            try:
                c = float(np.linalg.cond(K))
            except np.linalg.LinAlgError:
                c = float('inf')
            conds.append(c)
        s = "  ".join(f"{c:>7.1e}" for c in conds)
        print(f"  M={M:>2d}  {spacing:>7.3f}  {s}")


if __name__ == '__main__':
    main()
