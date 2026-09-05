"""Pre-flight for the keep-all Duffing scaling run.

(1) Dumps the existing canonical efgp/sp cells so the new keep-all cells can be
    compared against the same numbers the paper table already quotes.
(2) Reports the spectral-grid size M at the canonical eps_grid=1e-3 and the
    implied monolithic reverse-mode tape footprint -- the keep-all E-step is
    O(N*M) in memory (KEEPALL_AUTODIFF_NOTES.md item 3), so this is the go/no-go
    for running T=1e5 monolithic rather than chunked.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import jax
import jax.numpy as jnp

import demos.bench_gpdrift_x64 as base       # enables x64 on import; VAR_INIT
import sing.efgp_jax_primitives as jp

print(f"backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')}")

# ---- (1) existing canonical cells ------------------------------------------
print("\n=== existing canonical Duffing cells ===")
for d in sorted(Path('demos').glob('_bench_duffing_scaling*')):
    cells = sorted(d.glob('cell_T*.npz'))
    if not cells:
        continue
    print(f"\n-- {d} --")
    for c in cells:
        z = np.load(c, allow_pickle=True)
        g = lambda k, dflt='?': (z[k].item() if k in z.files else dflt)
        if g('status') != 'ok':
            print(f"  {c.name:46s} FAILED")
            continue
        print(f"  {c.name:46s} T={g('T'):>6} m={str(g('method')):12s} "
              f"M={g('M')!s:>3} wall={g('wall'):8.1f}s "
              f"nrmse={g('drift_nrmse'):.4f} lat_pc={g('lat_pc'):.4f} "
              f"l={g('ls_final'):.3f} var={g('var_final'):.3f} "
              f"estep={g('estep_method', 'n/a')}")

# ---- (2) grid size + monolithic tape footprint -----------------------------
# Replicate the library's default X_template box for this bench: init_params
# is mu0=0, V0=0.1*I -> half_span = max(3, 4*sqrt(0.1)) = 3 per dim, so the
# box is [-3,3]^2 (T-independent). ls_init=0.7, eps_grid=1e-3.
LS_INIT, EPS, D = 0.7, 1e-3, 2
half = 3.0
box = jnp.linspace(-1., 1., 64)[:, None] * jnp.full((1, D), half)
grid = jp.spectral_grid_se(LS_INIT, base.VAR_INIT, box, eps=EPS)
M = int(grid.M)
print(f"\n=== grid at ls_init={LS_INIT} var={base.VAR_INIT} eps={EPS} ===")
print(f"  mtot_per_dim={grid.mtot_per_dim}  M={M}  (T-independent: "
      f"adaptive-h fixes K_per_dim from the init grid)")

print(f"\n=== keep-all monolithic memory, M={M} ===")
print(f"{'T':>8} {'N=T-1':>8} {'(N,M) c128':>12} {'x12 tape':>12}")
for T in (1000, 10000, 100000):
    N = T - 1
    one = N * M * 16 / 2 ** 30          # one (N,M) complex128 array
    print(f"{T:>8} {N:>8} {one:>10.2f}GB {12 * one:>10.1f}GB")
print("\nH100_SXM5 = 80GB HBM. Compare the 'x12 tape' column; if it is not"
      "\ncomfortably under ~60GB, use nat_grad_batched_chunked for that T.")
