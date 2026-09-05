"""
replay_mstep_landscape.py  —  CPU-light M-step objective replay.

Loads an EFGP M-step dump (produced by the EFGP_DUMP_MSTEP hook in
sing/efgp_em.py) and replays the *exact* collapsed kernel M-step objective
`total_loss(log_ls, log_var)` (m_step_kernel_jax) on the FROZEN, theta-free
(top, z_r) it captured.  This isolates the M-step objective from the E-step:

    "Given this q(x), where does the M-step objective L_M(l, s2) want l?"

The dump also carries the raw Stein sources (m_src, S_src, d_src, C_src, w_src)
and grid geometry, so we can REBUILD (top, z_r) with S_src scaled by a factor,
testing the hypothesis that the posterior variance S is what regularizes the
objective away from small-l collapse.

Usage (login node, light — but see --rebuild which spreads all sources):
    PY=/home/users/ccitren/venvs/sing-py312/bin/python
    $PY demos/replay_mstep_landscape.py --glob 'demos/_mstep_dump/mstep_oracle*_it008.npz'
    $PY demos/replay_mstep_landscape.py --dump demos/_mstep_dump/mstep_oracleK100_it008.npz \
        --rebuild --s-scales 0 0.25 1 4 16
"""
from __future__ import annotations
import os
# be gentle on the login node
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import glob as _glob
import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jla

from sing.efgp_jax_drift import _ws_real_se


def build_T_mat(v_fft, ns):
    """Reconstruct the dense (M,M) BTTB Gram from the FFT-cached conv vector,
    replicating m_step_kernel_jax lines 822-831."""
    v_fft = jnp.asarray(v_fft)
    ns = tuple(int(n) for n in ns)
    cdtype = v_fft.dtype
    _v_pad = jnp.fft.ifftn(v_fft).astype(cdtype)
    _ns_v = tuple(2 * n - 1 for n in ns)
    _v_conv = _v_pad[tuple(slice(0, L) for L in _ns_v)]
    _d = len(ns)
    _mi = jnp.indices(ns).reshape(_d, -1)
    _offset = jnp.array([n - 1 for n in ns], dtype=jnp.int32)
    _diff = (_mi[:, :, None] - _mi[:, None, :] + _offset[:, None, None])
    T_mat = _v_conv[tuple(_diff[k] for k in range(_d))]
    return T_mat


def make_loss_fn(T_mat, z_r, xis_flat, h_scalar, D_lat, D_out):
    T_mat = jnp.asarray(T_mat)
    z_r = jnp.asarray(z_r)
    xis_flat = jnp.asarray(xis_flat)
    cdtype = T_mat.dtype
    M = T_mat.shape[0]
    eye_c = jnp.eye(M, dtype=cdtype)

    def total_loss(log_ls, log_var):
        ws_real = _ws_real_se(log_ls, log_var, xis_flat, h_scalar, D_lat)
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
        return det_loss + 0.5 * D_out * logdet

    return jax.jit(total_loss)


def sweep_1d(loss_fn, log_var_fixed, ls_grid):
    vals = np.array([float(loss_fn(jnp.asarray(math.log(l)),
                                   jnp.asarray(log_var_fixed)))
                     for l in ls_grid])
    return vals


def analyze_dump(path, ls_grid, verbose=True):
    z = np.load(path, allow_pickle=True)
    D = int(z['D'])
    T_mat = build_T_mat(z['top_v_fft'], z['top_ns'])
    h_scalar = float(np.asarray(z['h_per_dim']).reshape(-1)[0])
    loss_fn = make_loss_fn(T_mat, z['z_r'], z['xis_flat'], h_scalar, D, D)
    log_var = float(z['log_var'])
    log_ls_cur = float(z['log_ls'])
    M = int(T_mat.shape[0])
    vals = sweep_1d(loss_fn, log_var, ls_grid)
    i_arg = int(np.argmin(vals))
    # gradient wrt log_ls at the current l
    g = jax.grad(loss_fn, argnums=0)(jnp.asarray(log_ls_cur),
                                     jnp.asarray(log_var))
    if verbose:
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"  it={int(z['it'])}  M={M}  cur l={math.exp(log_ls_cur):.3f} "
              f"var={math.exp(log_var):.3f}")
        print(f"  argmin_l (fixed var) = {ls_grid[i_arg]:.3f}  "
              f"(loss {vals[i_arg]:.4g})")
        print(f"  dLoss/dlog_ls at cur l = {float(g):+.4g}  "
              f"({'push l DOWN' if float(g) > 0 else 'push l UP'})")
    return dict(path=path, it=int(z['it']), M=M, argmin_l=float(ls_grid[i_arg]),
                ls_grid=ls_grid, vals=vals, cur_l=math.exp(log_ls_cur),
                cur_var=math.exp(log_var), grad_logls=float(g))


def rebuild_with_scaled_S(path, s_scales, ls_grid):
    """Rebuild (top, z_r) via the gmix assembly with S_src *= scale, and report
    argmin_l for each scale.  Heavier (spreads all sources) — a few builds."""
    import sing.efgp_jax_drift as jpd
    import sing.efgp_jax_primitives as jp
    z = np.load(path, allow_pickle=True)
    D = int(z['D'])
    m_src = jnp.asarray(z['m_src']); S_src = jnp.asarray(z['S_src'])
    d_src = jnp.asarray(z['d_src']); C_src = jnp.asarray(z['C_src'])
    w_src = jnp.asarray(z['w_src'])
    sig2 = float(z['sigma_drift_sq'])
    fine_N = int(z['gmix_fine_N']); stencil_r = int(z['gmix_stencil_r'])
    h_scalar = float(np.asarray(z['h_per_dim']).reshape(-1)[0])
    # reconstruct a minimal JaxGridState from dumped geometry
    xis_flat = jnp.asarray(z['xis_flat'])
    mtot = tuple(int(v) for v in np.asarray(z['mtot_per_dim']).reshape(-1))
    grid = jp.JaxGridState(
        xis_flat=xis_flat,
        ws=jnp.asarray(z['ws_real']).astype(jnp.complex128),
        h_per_dim=jnp.asarray(z['h_per_dim']),
        mtot_per_dim=mtot,
        xcen=jnp.asarray(z['xcen']),
        M=int(xis_flat.shape[0]),
        d=int(xis_flat.shape[1]),
    )
    results = []
    for scale in s_scales:
        mu_r, _, top = jpd.compute_mu_r_gmix_jax(
            m_src, S_src * scale, d_src, C_src, w_src, grid,
            sigma_drift_sq=sig2, D_lat=D, D_out=D,
            fine_N=fine_N, stencil_r=stencil_r)
        ws_real_c = grid.ws.real.astype(grid.ws.dtype)
        ws_safe = jnp.where(jnp.abs(ws_real_c) < 1e-30,
                            jnp.array(1e-30, dtype=ws_real_c.dtype), ws_real_c)
        from sing.efgp_jax_primitives import make_A_apply
        A_apply = make_A_apply(grid.ws, top, sigmasq=1.0)
        h_r0 = jax.vmap(A_apply)(mu_r)
        z_r = h_r0 / ws_safe
        T_mat = build_T_mat(top.v_fft, top.ns)
        loss_fn = make_loss_fn(T_mat, z_r, z['xis_flat'], h_scalar, D, D)
        vals = sweep_1d(loss_fn, float(z['log_var']), ls_grid)
        i = int(np.argmin(vals))
        print(f"  S x {scale:>6.3g}:  argmin_l = {ls_grid[i]:.3f}  "
              f"(tr S mean = {float(jnp.mean(jnp.trace(S_src*scale, axis1=-1, axis2=-2))):.4g})")
        results.append((scale, float(ls_grid[i]), vals))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None)
    ap.add_argument('--glob', default=None)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--s-scales', type=float, nargs='+',
                    default=[0.0, 0.25, 1.0, 4.0, 16.0])
    ap.add_argument('--n-grid', type=int, default=41)
    ap.add_argument('--l-lo', type=float, default=0.1)
    ap.add_argument('--l-hi', type=float, default=4.0)
    ap.add_argument('--save', default=None, help='npz to save landscapes')
    args = ap.parse_args()

    ls_grid = np.exp(np.linspace(math.log(args.l_lo), math.log(args.l_hi),
                                 args.n_grid))
    paths = []
    if args.glob:
        paths = sorted(_glob.glob(args.glob))
    if args.dump:
        paths.append(args.dump)
    if not paths:
        raise SystemExit("no dumps matched")

    out = {}
    for p in paths:
        r = analyze_dump(p, ls_grid)
        out[os.path.basename(p)] = r
        if args.rebuild:
            print("  -- S-inflation rebuild --")
            rebuild_with_scaled_S(p, args.s_scales, ls_grid)

    if args.save:
        np.savez(args.save, **{k: v['vals'] for k, v in out.items()},
                 ls_grid=ls_grid)
        print(f"\nsaved -> {args.save}")


if __name__ == '__main__':
    main()
