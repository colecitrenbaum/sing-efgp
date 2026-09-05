"""Prove the EFGP M-step recompiles every call, isolate the true math cost,
and demonstrate the (approximation-free) fix: hoist the jit so the compiled
gradient is reused.

Run: JAX_PLATFORMS=cpu python demos/diag_mstep_recompile.py
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jla
import optax
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd


def setup(M_per_dim=13, N_src=2000, seed=0):
    rng = np.random.default_rng(seed)
    ls, var = 0.7, 1.0
    t = np.linspace(0, 6 * math.pi, N_src)
    m = np.stack([np.cos(t), np.sin(t)], -1) + 0.02 * rng.standard_normal((N_src, 2))
    grid = jp.spectral_grid_se(ls, var, jnp.asarray(m), eps=1e-3)
    D = 2
    S = np.tile(0.01 * np.eye(2), (N_src, 1, 1))
    d = 0.05 * rng.standard_normal((N_src, 2))
    C = 0.01 * rng.standard_normal((N_src, 2, 2))
    w = np.full(N_src, 0.01)
    mu_r, _, top = jpd.compute_mu_r_gmix_jax(
        jnp.asarray(m), jnp.asarray(S), jnp.asarray(d), jnp.asarray(C),
        jnp.asarray(w), grid, sigma_drift_sq=0.01, D_lat=D, D_out=D,
        fine_N=256, stencil_r=8)
    ws = grid.ws
    z_r = (ws.real[None, :] * mu_r) / jnp.where(jnp.abs(ws.real) < 1e-30, 1e-30, ws.real)[None, :]
    return grid, top, mu_r, z_r, D


def time_call(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    print(f"device={jax.devices()}", flush=True)
    grid, top, mu_r, z_r, D = setup()
    M = grid.M
    print(f"M={M}  (dense A is {M}x{M})", flush=True)
    args = dict(mu_r_fixed=mu_r, z_r=z_r, top=top, xis_flat=grid.xis_flat,
                h_per_dim=grid.h_per_dim, D_lat=D, D_out=D, n_inner=4, lr=0.01)

    # (1) Current m_step_kernel_jax: jit(total_loss) built inside → recompiles
    print("\n(1) Current m_step_kernel_jax, 5 identical calls:", flush=True)
    ts = time_call(lambda: jpd.m_step_kernel_jax(math.log(0.7), math.log(1.0), **args))
    print("    per-call (s): " + ", ".join(f"{x:.3f}" for x in ts), flush=True)
    print(f"    -> all ~equal & large ⇒ RECOMPILE each call" if min(ts) > 0.1
          else "    -> cached", flush=True)

    # (2) Build the SAME objective once with a hoisted (module-level-style) jit,
    #     reused across calls.  Pure reuse — identical math.
    cdtype = top.v_fft.dtype
    eye_c = jnp.eye(M, dtype=cdtype)
    _v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    _ns_v = tuple(2 * n - 1 for n in top.ns)
    _v_conv = _v_pad[tuple(slice(0, L) for L in _ns_v)]
    _d = len(top.ns); _mi = jnp.indices(top.ns).reshape(_d, -1)
    _off = jnp.array([n - 1 for n in top.ns], dtype=jnp.int32)
    _diff = (_mi[:, :, None] - _mi[:, None, :] + _off[:, None, None])
    T_mat = _v_conv[tuple(_diff[k] for k in range(_d))]
    h_scalar = grid.h_per_dim[0]

    @jax.jit
    def loss_hoisted(log_ls, log_var):
        ws_real = jpd._ws_real_se(log_ls, log_var, grid.xis_flat, h_scalar, D)
        ws_c = ws_real.astype(cdtype)
        A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
        L = jnp.linalg.cholesky(A)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
        h = ws_c[None, :] * z_r
        def solve_one(b):
            y = jla.solve_triangular(L, b, lower=True)
            return jla.solve_triangular(L.conj().T, y, lower=False)
        mu = jax.vmap(solve_one)(h)
        det_loss = -0.5 * jnp.sum(jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
        return det_loss + 0.5 * D * logdet

    vg = jax.jit(jax.value_and_grad(loss_hoisted, argnums=(0, 1)))
    print("\n(2) Hoisted jit (compiled once, reused), 5 identical value_and_grad calls:", flush=True)
    ts = time_call(lambda: vg(jnp.float32(math.log(0.7)), jnp.float32(math.log(1.0))))
    print("    per-call (s): " + ", ".join(f"{x:.4f}" for x in ts), flush=True)
    print(f"    -> first compiles, rest reuse ⇒ steady-state = the TRUE math cost", flush=True)

    # (3) The M-step math is 4 Adam steps of that gradient.  Steady-state cost:
    warm = float(np.median(ts[1:])) if len(ts) > 1 else ts[-1]
    print(f"\n(3) True M-step math ≈ 4 x {warm*1e3:.3f} ms = {4*warm*1e3:.3f} ms "
          f"(vs current ~{min(time_call(lambda: jpd.m_step_kernel_jax(math.log(0.7), math.log(1.0), **args), n=3)):.2f} s/call)",
          flush=True)


if __name__ == '__main__':
    main()
