"""(1) Confirm the gather's dL/dS matches the direct closed form ONCE both are
symmetrised the way nat_grad_batched already does, and (2) if so, find what
actually makes keep-all + gather diverge, by comparing the full keep-all
natural gradients {J,h,L} gather-vs-direct on real q(x) state.

Background: a raw element-wise dL/dS comparison is meaningless -- S is
symmetric, so dL/dS is only defined up to how the off-diagonal is split
between (a,b) and (b,a). The gather lumps it all into the upper triangle; the
direct closed form splits it evenly. nat_grad_batched applies
symm(A) = 0.5*(A + A^T), which maps one onto the other.
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
from sing.exp_batched_estep import nat_grad_batched

N, D = 2000, 2
GRID_X = jnp.array([[-3.0, -3.0], [3.0, 3.0]])
LS, VAR, EPS = 0.7, 1.0, 1e-3

grid = jp.spectral_grid_se(LS, VAR, GRID_X, eps=EPS)
rng = np.random.default_rng(1)
raw = rng.normal(size=(D,) + tuple(grid.mtot_per_dim))
rev = tuple(slice(None, None, -1) for _ in range(D))
mu_r = jnp.asarray((0.5 * (raw + raw[(slice(None),) + rev])).reshape(D, grid.M))
ms = jr.uniform(jr.PRNGKey(0), (1, N, D), minval=-2, maxval=2)

print(f"backend={jax.default_backend()} M={grid.M} N={N}")


def sym(A):
    return 0.5 * (np.asarray(A) + np.swapaxes(np.asarray(A), -1, -2))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / den) if den else np.nan


def make_loss(mode, gather_N, stencil_r):
    if mode == 'direct':
        def L(ms_, Ss_):
            m, S = ms_[0], Ss_[0]
            Ef = jax.vmap(lambda a, b: Ef_diff(a, b, mu_r, grid))(m, S)
            Edf = jax.vmap(lambda a, b: Edfdx_diff(a, b, mu_r, grid))(m, S)
            return (Ef ** 2).sum() + (Edf ** 2).sum()
    else:
        def L(ms_, Ss_):
            Ef, _, Edf = jpd.drift_moments_gmix_jax(
                mu_r, grid, ms_, Ss_, D_lat=D, D_out=D,
                gather_N=gather_N, stencil_r=stencil_r)
            return (Ef ** 2).sum() + (Edf ** 2).sum()
    return L


print("\n=== (1) dL/dS, RAW vs SYMMETRISED ===")
print(f"{'S':>7} {'r':>4} {'dL/dm rel':>11} {'dL/dS raw':>11} {'dL/dS symm':>12}")
for s_scale in (0.10, 0.03, 0.01):
    Ss = jnp.broadcast_to(jnp.eye(D) * s_scale, (1, N, D, D))
    _, (gm_d, gS_d) = jax.jit(jax.value_and_grad(
        make_loss('direct', 64, 6), argnums=(0, 1)))(ms, Ss)
    for sr in (8, 16):
        _, (gm_g, gS_g) = jax.jit(jax.value_and_grad(
            make_loss('gather', 64, sr), argnums=(0, 1)))(ms, Ss)
        print(f"{s_scale:>7} {sr:>4} {rel(gm_d, gm_g):>11.2e} "
              f"{rel(gS_d, gS_g):>11.2e} {rel(sym(gS_d), sym(gS_g)):>12.2e}")

# ---- (2) full keep-all natural gradients, gather vs direct ----------------
print("\n=== (2) keep-all nat grads {J,h,L}: gather vs direct ===")
t_grid = jnp.linspace(0., N * 0.0375, N)
trial_mask = jnp.ones((1, N), bool)
init_params = dict(mu0=jnp.zeros((1, D)), V0=jnp.eye(D)[None] * 0.1)
sigma = 0.4

for s_scale in (0.10, 0.03):
    S = jnp.broadcast_to(jnp.eye(D) * s_scale, (1, N, D, D))
    m = ms
    # mean params consistent with (m, S): ExxT = S + m m^T, ExxnT = m_t m_{t+1}^T
    mean_params = dict(
        Ex=m,
        ExxT=S + m[..., :, None] * m[..., None, :],
        ExxnT=(m[:, :-1, :, None] * m[:, 1:, None, :]))
    out = {}
    for mode, kw in (('direct', dict(gather=False)),
                     ('gather', dict(gather=True, gather_N=64, stencil_r=8))):
        out[mode] = nat_grad_batched(mean_params, mu_r, grid, t_grid,
                                     trial_mask, init_params, sigma,
                                     moment='exact', **kw)
    print(f"  S={s_scale}*I   " + "  ".join(
        f"{k}: {rel(out['direct'][k], out['gather'][k]):.2e}"
        for k in ('J', 'h', 'L')))
    for k in ('J', 'h', 'L'):
        a = np.asarray(out['direct'][k])
        print(f"     {k}: ||direct||={np.linalg.norm(a):.4e} "
              f"finite_gather={bool(np.all(np.isfinite(out['gather'][k])))}")
