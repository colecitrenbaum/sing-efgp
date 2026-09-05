"""Isolate WHY the gmix gather's dEf/dS is wrong (~23%, flat in stencil_r).

Two candidates:
  (a) spatial resolution -- the gather Riemann-sums on a lattice of spacing
      h_grid = 1/(gather_N * h_spec). dN(x;m,S)/dS is a sharper function of x
      than N itself, so it may need a finer grid (larger gather_N), which
      stencil_r cannot fix.
  (b) structural -- something in _gather_2d is not differentiable in S the way
      the closed form is, in which case NO gather_N/stencil_r rescues it.

Test: single source, compare dEf/dS from the gather against the exact closed
form  dEf_r/dS = sum_k mu_rk ws_k e^{2pi i xi.(m-xcen)} (-2pi^2 xi xi^T)
                 e^{-2pi^2 xi^T S xi},
sweeping gather_N at fixed generous stencil_r.
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
from sing.efgp_gmix_gather import gmix_inverse_nufft_2d

D = 2
GRID_X = jnp.array([[-3.0, -3.0], [3.0, 3.0]])
LS, VAR, EPS = 0.7, 1.0, 1e-3

grid = jp.spectral_grid_se(LS, VAR, GRID_X, eps=EPS)
M_per_dim = int(grid.mtot_per_dim[0])
h_spec = float(grid.h_per_dim[0])
rng = np.random.default_rng(1)
raw = rng.normal(size=(D,) + tuple(grid.mtot_per_dim))
rev = tuple(slice(None, None, -1) for _ in range(D))
mu_r = jnp.asarray((0.5 * (raw + raw[(slice(None),) + rev])).reshape(D, grid.M))

m0 = jnp.array([[0.37, -0.61]])                  # one source
print(f"backend={jax.default_backend()} M={grid.M} mtot={grid.mtot_per_dim} "
      f"h_spec={h_spec:.4f}")

# ---- exact closed form: value and dEf/dS for output dim r=0 ---------------
xis = grid.xis_flat                                        # (M,2)
ws = grid.ws.real


def exact_val_and_dS(m, S):
    phase = jnp.exp(2j * jnp.pi * (xis @ (m - grid.xcen)))
    quad = jnp.einsum('md,de,me->m', xis, S, xis)
    env = jnp.exp(-2 * jnp.pi ** 2 * quad)
    c = (ws * env).astype(phase.dtype) * phase             # (M,)
    val = (mu_r[0] * c).sum().real
    # dc/dS_ab = c * (-2 pi^2 xi_a xi_b)
    w = mu_r[0] * c                                        # (M,)
    dS = jnp.einsum('m,ma,mb->ab', w, xis, xis).real * (-2 * jnp.pi ** 2)
    return val, dS


# ---- gather: same quantity, by autodiff -----------------------------------
def gather_val(m, S, gather_N, stencil_r):
    """Ef_{r=0} through the gather. Mirrors drift_moments_gmix_jax's frame
    handling: mu_r is relative-frame, the gather wants absolute-frame."""
    cdtype = grid.ws.dtype
    xcen_phase_inv = jnp.exp(
        -2j * jnp.pi * (xis @ grid.xcen.astype(xis.dtype))).astype(cdtype)
    fk = (grid.ws.real.astype(cdtype) * (mu_r[0] * xcen_phase_inv))
    return gmix_inverse_nufft_2d(
        m, S, fk.reshape(M_per_dim, M_per_dim),
        xcen=grid.xcen, h_spec=h_spec, M_per_dim=M_per_dim,
        N=gather_N, stencil_r=stencil_r).real[0]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


for s_scale in (0.10, 0.03):
    S0 = jnp.eye(D) * s_scale
    v_ex, dS_ex = exact_val_and_dS(m0[0], S0)
    sig = s_scale ** 0.5
    print(f"\n--- S = {s_scale}*I  (sigma={sig:.3f}) ---")
    print(f"    exact Ef={float(v_ex):.10f}   ||dEf/dS||={np.linalg.norm(dS_ex):.4e}")
    print(f"{'gather_N':>9} {'h_grid':>9} {'sig/h':>7} {'r':>4} "
          f"{'val rel':>10} {'dEf/dS rel':>11}")
    for gN in (64, 128, 256, 512):
        h_grid = 1.0 / (gN * h_spec)
        r = int(min(gN // 2 - 1, max(8, np.ceil(6 * sig / h_grid))))
        f = lambda M_, S_: gather_val(M_, S_, gN, r)
        v_g = float(jax.jit(f)(m0, S0[None]))
        dS_g = np.asarray(jax.jit(jax.grad(f, argnums=1))(m0, S0[None]))[0]
        print(f"{gN:>9} {h_grid:>9.4f} {sig/h_grid:>7.2f} {r:>4} "
              f"{abs(v_g-float(v_ex))/abs(float(v_ex)):>10.2e} "
              f"{rel(dS_ex, dS_g):>11.2e}")
    print(f"    exact dEf/dS =\n{np.asarray(dS_ex)}")
    gN, sig_ = 256, sig
    h_grid = 1.0 / (gN * h_spec)
    r = int(min(gN // 2 - 1, max(8, np.ceil(6 * sig_ / h_grid))))
    f = lambda M_, S_: gather_val(M_, S_, gN, r)
    print(f"    gather dEf/dS (N={gN}, r={r}) =\n"
          f"{np.asarray(jax.jit(jax.grad(f, argnums=1))(m0, S0[None]))[0]}")
