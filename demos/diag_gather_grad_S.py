"""Why does keep-all + gather diverge? Element-wise gather-vs-direct check of
BOTH keep-all gradients, dL/dm and dL/dS.

bench_gather_vs_direct.py compares only ||dL/dm|| (a norm, not element-wise,
and not dL/dS at all). But keep-all's entire contribution is that it RETAINS
the S-derivatives that production drops -- and the gmix gather is a Gaussian
stencil truncated at stencil_r. Its dL/dS carries an extra delta^2 weight
relative to the value, so the stencil tail decays slower in the derivative
than in the function: a stencil that is fine for Ef can still be badly wrong
for dEf/dS. That would corrupt exactly the term keep-all exists to keep.

Run:  python demos/diag_gather_grad_S.py [stencil_r ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax; jax.config.update("jax_enable_x64", True)   # MUST precede jax use
import jax.numpy as jnp, jax.random as jr, numpy as np

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.exp_full_moments import Ef_diff, Edfdx_diff

N, D = 2000, 2
# Match the canonical Duffing bench: box [-3,3]^2, ls_init=0.7, eps_grid=1e-3
# -> M=289, and the S scale q(x) actually reaches there.
GRID_X = jnp.array([[-3.0, -3.0], [3.0, 3.0]])
LS, VAR, EPS = 0.7, 1.0, 1e-3


def build():
    grid = jp.spectral_grid_se(LS, VAR, GRID_X, eps=EPS)
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(D,) + tuple(grid.mtot_per_dim))
    rev = tuple(slice(None, None, -1) for _ in range(D))
    mu_r = jnp.asarray((0.5 * (raw + raw[(slice(None),) + rev]))
                       .reshape(D, grid.M))
    ms = jr.uniform(jr.PRNGKey(0), (1, N, D), minval=-2, maxval=2)
    return grid, mu_r, ms


def make_loss(mode, grid, mu_r, gather_N, stencil_r):
    """Same scalar the keep-all E-step differentiates (Ef/Edf contractions)."""
    if mode == 'direct':
        def L(ms, Ss):
            m, S = ms[0], Ss[0]
            Ef = jax.vmap(lambda a, b: Ef_diff(a, b, mu_r, grid))(m, S)
            Edf = jax.vmap(lambda a, b: Edfdx_diff(a, b, mu_r, grid))(m, S)
            return (Ef ** 2).sum() + (Edf ** 2).sum()
    else:
        def L(ms, Ss):
            Ef, _, Edf = jpd.drift_moments_gmix_jax(
                mu_r, grid, ms, Ss, D_lat=D, D_out=D,
                gather_N=gather_N, stencil_r=stencil_r)
            return (Ef ** 2).sum() + (Edf ** 2).sum()
    return L


def rel(a, b):
    """Element-wise relative error, ||a-b|| / ||a||."""
    a, b = np.asarray(a), np.asarray(b)
    den = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / den) if den > 0 else np.nan


grid, mu_r, ms = build()
print(f"backend={jax.default_backend()}  M={grid.M} "
      f"mtot={grid.mtot_per_dim}  N={N}  ls={LS} eps={EPS}")

stencils = [int(a) for a in sys.argv[1:]] or [6, 8, 12, 16, 24]
gather_N = 64

# S sweep: q(x) starts near V0=0.1*I and shrinks as the fit localises.
for s_scale in (0.10, 0.03, 0.01):
    Ss = jnp.broadcast_to(jnp.eye(D) * s_scale, (1, N, D, D))
    Ld = jax.jit(jax.value_and_grad(
        make_loss('direct', grid, mu_r, gather_N, 6), argnums=(0, 1)))
    v_d, (gm_d, gS_d) = Ld(ms, Ss)
    print(f"\n--- S = {s_scale}*I   (sigma = {s_scale**0.5:.3f}, "
          f"ls/sigma = {LS/s_scale**0.5:.1f}) ---")
    print(f"{'stencil_r':>10} {'val rel':>10} {'dL/dm rel':>11} "
          f"{'dL/dS rel':>11}   {'S err / m err':>13}")
    for sr in stencils:
        Lg = jax.jit(jax.value_and_grad(
            make_loss('gather', grid, mu_r, gather_N, sr), argnums=(0, 1)))
        v_g, (gm_g, gS_g) = Lg(ms, Ss)
        r_v = abs(float(v_d - v_g)) / abs(float(v_d))
        r_m, r_S = rel(gm_d, gm_g), rel(gS_d, gS_g)
        print(f"{sr:>10} {r_v:>10.2e} {r_m:>11.2e} {r_S:>11.2e}"
              f"   {r_S / max(r_m, 1e-300):>13.1f}x")

print("\nIf the last column is >> 1, the stencil is far worse for dL/dS than"
      "\nfor dL/dm -- i.e. differentiating through the truncated gather"
      "\ncorrupts precisely the S-derivative that keep-all exists to retain.")
