"""Why is SING compile so slow?  Measure the smoother's XLA graph size and
compile wall vs T.

Hypothesis: the smoother (natural_to_*_params -> compute_log_normalizer_parallel
-> lax.associative_scan) UNROLLS into an O(T)-sized jaxpr at trace time (it is
not a rolled while_loop), so lowering + XLA compile grow with T.  This is shared
by EFGP and SparseGP (both call natural_to_mean/marginal_params).

Prints, per T: #jaxpr equations (recursive) for forward log-Z and for the
value_and_grad smoother, plus lower+compile wall.  A ~linear #eqns vs T
confirms unrolling.

Run: JAX_PLATFORMS=cpu python demos/investigate_compile.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap
from sing.utils.sing_helpers import (
    natural_to_marginal_params, compute_log_normalizer_parallel,
    dynamics_to_natural_params)


def count_eqns(jaxpr):
    """Recursively count primitive equations (descend into sub-jaxprs)."""
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for sj in jaxpr_subs(eqn):
            n += count_eqns(sj)
    return n


def jaxpr_subs(eqn):
    subs = []
    for v in eqn.params.values():
        if hasattr(v, 'eqns'):
            subs.append(v)
        elif hasattr(v, 'jaxpr') and hasattr(v.jaxpr, 'eqns'):
            subs.append(v.jaxpr)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if hasattr(x, 'eqns'):
                    subs.append(x)
                elif hasattr(x, 'jaxpr') and hasattr(x.jaxpr, 'eqns'):
                    subs.append(x.jaxpr)
    return subs


def make_nat(T, D=2, sigma=1.0):
    A = jnp.zeros((T - 1, D, D)); b = jnp.zeros((T, D)).at[0].set(jnp.zeros(D))
    Q = jnp.tile((sigma ** 2) * jnp.eye(D), (T, 1, 1)).at[0].set(jnp.eye(D) * 0.1)
    tm = jnp.ones((T,), dtype=bool)
    nat = dynamics_to_natural_params(A, b, Q, tm)
    return jax.tree_util.tree_map(lambda x: x[None], nat), tm[None]


def main():
    print(f"device={jax.devices()}", flush=True)
    print(f"{'T':>7} {'#eqns_fwd':>12} {'#eqns_grad':>12} "
          f"{'lower+compile_grad(s)':>22}", flush=True)
    for T in [500, 1000, 2000, 4000, 8000]:
        nat, tm = make_nat(T)
        # forward-only log-normalizer (per single trial)
        nat0 = jax.tree_util.tree_map(lambda x: x[0], nat)
        fwd = lambda p, m: compute_log_normalizer_parallel(
            (-2) * p['J'], (-1) * p['L'], p['h'], m)
        jx_fwd = jax.make_jaxpr(fwd)(nat0, tm[0])
        n_fwd = count_eqns(jx_fwd.jaxpr)

        # value_and_grad smoother (the real per-iter call), vmapped over K=1
        marg = jax.jit(vmap(natural_to_marginal_params))
        jx_grad = jax.make_jaxpr(vmap(natural_to_marginal_params))(nat, tm)
        n_grad = count_eqns(jx_grad.jaxpr)

        t0 = time.perf_counter()
        lowered = marg.lower(nat, tm)
        compiled = lowered.compile()
        t_cc = time.perf_counter() - t0

        print(f"{T:>7} {n_fwd:>12} {n_grad:>12} {t_cc:>22.2f}", flush=True)


if __name__ == '__main__':
    main()
