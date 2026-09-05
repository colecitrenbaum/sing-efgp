"""Gather (NUFFT, O(M log M + N)) vs direct char-function sums (O(N*M))
for the keep-all autodiff moments + their gradient. Shows the M crossover.
See KEEPALL_AUTODIFF_NOTES.md.

Default: sweep M (via domain scale) x {direct, gather}, print wall + grad
agreement.  For an isolated per-config peak-memory read, run one config:
    python demos/bench_gather_vs_direct.py <direct|gather> <domain>
(then read RSS / nvidia-smi externally).

On GPU: watch (1) the crossover M where gather overtakes direct, (2) that
gather memory stays flat in M while direct grows O(N*M).
"""
import sys, time, resource
import jax; jax.config.update("jax_enable_x64", True)   # MUST precede jax use
import jax.numpy as jnp, jax.random as jr, numpy as np
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.exp_full_moments import Ef_diff, Edfdx_diff

N, D = 10000, 2

def build(domain):
    X = jnp.array([[-domain, -domain], [domain, domain]])
    grid = jp.spectral_grid_se(0.8, 1.0, X, eps=1e-2)
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(D,) + tuple(grid.mtot_per_dim))
    rev = tuple(slice(None, None, -1) for _ in range(D))
    mu_r = jnp.asarray((0.5*(raw + raw[(slice(None),)+rev])).reshape(D, grid.M))
    ms = jr.uniform(jr.PRNGKey(0), (1, N, 2), minval=-2, maxval=2)
    Ss = jnp.broadcast_to(jnp.eye(2)*0.05, (1, N, 2, 2))
    return grid, mu_r, ms, Ss

def loss(mode, grid, mu_r):
    if mode == 'direct':
        def L(ms, Ss):
            m, S = ms[0], Ss[0]
            Ef  = jax.vmap(lambda a,b: Ef_diff(a,b,mu_r,grid))(m, S)
            Edf = jax.vmap(lambda a,b: Edfdx_diff(a,b,mu_r,grid))(m, S)
            return (Ef**2).sum() + (Edf**2).sum()
    else:
        def L(ms, Ss):
            Ef, Eff, Edf = jpd.drift_moments_gmix_jax(
                mu_r, grid, ms, Ss, D_lat=2, D_out=2, gather_N=64, stencil_r=6)
            return (Ef**2).sum() + (Edf**2).sum()
    return L

def timeit(mode, domain):
    grid, mu_r, ms, Ss = build(domain)
    g = jax.jit(jax.grad(loss(mode, grid, mu_r), argnums=(0, 1)))
    r = g(ms, Ss); jax.block_until_ready(r)
    t = time.time(); [jax.block_until_ready(g(ms, Ss)) for _ in range(5)]
    dt = (time.time() - t) / 5
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    return grid.M, dt, peak, float(jnp.linalg.norm(r[0]))

if len(sys.argv) == 3:                      # single config (for RSS isolation)
    M, dt, peak, gm = timeit(sys.argv[1], float(sys.argv[2]))
    print(f"mode={sys.argv[1]:6s} M={M} N={N} grad_wall={dt*1e3:.1f}ms "
          f"peakRSS={peak:.2f}GB |g_m|={gm:.6e}")
else:                                       # sweep
    print(f"backend={jax.default_backend()}  N={N}")
    print(f"{'M':>6} {'direct ms':>11} {'gather ms':>11} {'speedup':>8} {'grad rel diff':>14}")
    for dom in [3, 6, 10]:
        M, dtd, _, gd = timeit('direct', dom)
        _, dtg, _, gg = timeit('gather', dom)
        rel = abs(gd - gg) / abs(gd)
        print(f"{M:>6} {dtd*1e3:>11.1f} {dtg*1e3:>11.1f} {dtd/dtg:>7.1f}x {rel:>14.2e}",
              flush=True)
