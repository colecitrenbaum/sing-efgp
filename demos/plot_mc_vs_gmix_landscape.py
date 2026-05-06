"""Render hyperparameter learning paths (MC vs GMIX) on log-marginal-likelihood
contours.

Reads from the bench script's saved q(x) for both methods (assumes
``demos/bench_gpdrift_mc_vs_gmix.py`` was run already), or runs a quick
fit if not.  Outputs PNG to ``demos/_bench_mc_vs_gmix_out/``.

The log-marginal landscape is the **collapsed EFGP marginal likelihood**
at the converged GMIX q(x):
    L(θ) = -½ Σ_r Re⟨h_θ,r, A_θ^{-1} h_θ,r⟩ + ½ D_out log|A_θ|
exactly the loss minimised by ``m_step_kernel_jax``.
"""
from __future__ import annotations

import sys, os, math, time, json
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs

# Reuse the bench's drift-builder
from demos.bench_gpdrift_mc_vs_gmix import (
    make_gp_drift, GLik, _fit_one, evaluate, D, SIGMA, ALPHA_RESTORE
)

OUT_DIR = Path(__file__).resolve().parent / "_bench_mc_vs_gmix_out"
OUT_DIR.mkdir(exist_ok=True)


def _build_landscape_evaluator(grid, top, z_r, D_lat, D_out):
    """Closure over fixed q(x)-derived (top, z_r) returning loss(log_ls, log_var)."""
    h_scalar = grid.h_per_dim[0]
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    eye_c = jnp.eye(M, dtype=cdtype)

    # Materialise dense T_mat (one-time, OK for M up to ~2000).
    _v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    _ns_v = tuple(2 * n - 1 for n in top.ns)
    _v_conv = _v_pad[tuple(slice(0, L) for L in _ns_v)]
    _d = len(top.ns)
    _mi = jnp.indices(top.ns).reshape(_d, -1)
    _offset = jnp.array([n - 1 for n in top.ns], dtype=jnp.int32)
    _diff = (_mi[:, :, None] - _mi[:, None, :]
             + _offset[:, None, None])
    T_mat = _v_conv[tuple(_diff[k] for k in range(_d))]

    @jax.jit
    def loss(log_ls, log_var):
        ws_real = jpd._ws_real_se(log_ls, log_var,
                                    grid.xis_flat, h_scalar, D_lat)
        ws_c = ws_real.astype(cdtype)
        A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
        L = jnp.linalg.cholesky(A)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
        h = ws_c[None, :] * z_r
        def solve_one(b):
            y = jla.solve_triangular(L, b, lower=True)
            return jla.solve_triangular(L.conj().T, y, lower=False)
        mu = jax.vmap(solve_one)(h)
        det_loss = -0.5 * jnp.sum(
            jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
        return det_loss + 0.5 * D_out * logdet
    return loss


def _build_zr_top_from_qx(ms, Ss, SSs, t_grid, grid, ls_at, var_at):
    """Construct (z_r, top) under hypers (ls_at, var_at) and the given q(x)."""
    del_t = t_grid[1:] - t_grid[:-1]
    # Use gmix to build mu_r at (ls_at, var_at); deterministic, no MC noise.
    # First, refresh ws on grid for the chosen hypers.
    ws_at = jpd._ws_real_se(jnp.log(jnp.asarray(ls_at)),
                              jnp.log(jnp.asarray(var_at)),
                              grid.xis_flat, grid.h_per_dim[0], D
                              ).astype(grid.ws.dtype)
    grid_at = grid._replace(ws=ws_at)
    from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
    h_spec = float(grid_at.h_per_dim[0])
    sigma_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(Ss))))
    fine_N = pick_grid_size(h_spec=h_spec, m_extent=6.0, sigma_max=sigma_max)
    h_grid = 1.0 / (fine_N * h_spec)
    stencil_r = stencil_radius_for(Ss, h_grid)
    mu_r, _, top = jpd.compute_mu_r_gmix_jax(
        ms, Ss, SSs, del_t, grid_at,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=int(fine_N), stencil_r=int(stencil_r),
    )
    from sing.efgp_jax_primitives import make_A_apply
    A_apply = make_A_apply(grid_at.ws, top, sigmasq=1.0)
    h_r0 = jax.vmap(A_apply)(mu_r)
    ws_real_c = grid_at.ws.real.astype(grid_at.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real_c) < 1e-30,
                         jnp.array(1e-30, dtype=ws_real_c.dtype),
                         ws_real_c)
    z_r = h_r0 / ws_safe
    return z_r, top, grid_at


def run(T=2000, ls_true=0.8, var_true=1.0, n_em=25, n_estep=10):
    print(f"\n=== Landscape plot: T={T}, ls_true={ls_true} ===")
    drift_jax, drift_np = make_gp_drift(ls_true, var_true, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([0.0, 0.0]),
                       f=drift_jax, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    N_obs = 8
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N_obs, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    ip = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)

    ls_init = 0.3 if ls_true >= 0.6 else 1.5

    print("  fitting MC...")
    mp_mc, _, hist_mc, t_mc = _fit_one(
        'mc', ls_true, var_true, lik, t_grid, out_true, ip,
        ls_init, n_em, n_estep, S_marginal=2)
    print(f"    wall {t_mc:.1f}s, ℓ_final={hist_mc.lengthscale[-1]:.3f}, "
          f"σ²_final={hist_mc.variance[-1]:.3f}")
    print("  fitting GMIX...")
    mp_gx, _, hist_gx, t_gx = _fit_one(
        'gmix', ls_true, var_true, lik, t_grid, out_true, ip,
        ls_init, n_em, n_estep, S_marginal=2)
    print(f"    wall {t_gx:.1f}s, ℓ_final={hist_gx.lengthscale[-1]:.3f}, "
          f"σ²_final={hist_gx.variance[-1]:.3f}")

    # Build landscape under GMIX's converged q(x).
    print("  building log-marginal landscape under GMIX final q(x)...")
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    pin_ls = ls_true * 0.75
    grid = jp.spectral_grid_se(pin_ls, hist_gx.variance[-1],
                                X_template, eps=1e-2)
    ms_g = mp_gx['m'][0]
    Ss_g = jnp.asarray(np.asarray(mp_gx['S'][0]))
    SSs_g = jnp.asarray(np.asarray(mp_gx['SS'][0]))
    z_r, top, grid_at = _build_zr_top_from_qx(
        ms_g, Ss_g, SSs_g, t_grid, grid,
        hist_gx.lengthscale[-1], hist_gx.variance[-1])
    loss_fn = _build_landscape_evaluator(grid_at, top, z_r, D, D)

    # Grid in (log_ls, log_var)
    log_ls_grid = np.linspace(np.log(0.15), np.log(2.0), 41)
    log_var_grid = np.linspace(np.log(0.2), np.log(3.0), 41)
    LL, LV = np.meshgrid(log_ls_grid, log_var_grid, indexing='ij')

    print("  evaluating landscape...")
    t0 = time.time()
    L_grid = np.zeros_like(LL)
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            L_grid[i, j] = float(loss_fn(LL[i, j], LV[i, j]))
    print(f"    {time.time() - t0:.1f}s for {LL.size} grid points")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    LL_grid = np.exp(LL); LV_grid = np.exp(LV)
    # Subtract the min so contours are nicer
    L_normed = L_grid - L_grid.min()
    levels = np.percentile(L_normed[np.isfinite(L_normed)],
                            [1, 5, 10, 20, 30, 40, 55, 70, 85, 95])
    cs = ax.contourf(LL_grid, LV_grid, L_normed, levels=20, cmap='viridis')
    ax.contour(LL_grid, LV_grid, L_normed, levels=levels, colors='white',
                linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, ax=ax, label='-log p(y|θ) − min')

    # Overlay paths
    ls_mc = np.array(hist_mc.lengthscale)
    var_mc = np.array(hist_mc.variance)
    ls_gx = np.array(hist_gx.lengthscale)
    var_gx = np.array(hist_gx.variance)
    ax.plot(ls_mc, var_mc, 'o-', color='tab:blue', lw=1.5, ms=4,
             label=f'MC ({len(ls_mc)} EM iters)')
    ax.plot(ls_gx, var_gx, 's-', color='tab:orange', lw=1.5, ms=4,
             label=f'GMIX ({len(ls_gx)} EM iters)')
    ax.scatter([ls_mc[0]], [var_mc[0]], marker='*', s=180,
                color='tab:blue', edgecolor='k', zorder=5)
    ax.scatter([ls_gx[0]], [var_gx[0]], marker='*', s=180,
                color='tab:orange', edgecolor='k', zorder=5)
    ax.scatter([ls_true], [var_true], marker='X', s=220,
                color='red', edgecolor='k', zorder=6, label='truth')
    ax.set_xlabel(r'lengthscale $\ell$')
    ax.set_ylabel(r'variance $\sigma^2$')
    ax.set_title(f'Hyperparameter paths on log-marginal landscape  '
                  f'(T={T}, ℓ_true={ls_true}, σ²_true={var_true})')
    ax.legend(loc='best')
    plt.tight_layout()
    out = OUT_DIR / f"landscape_paths_T{T}_ls{ls_true}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # Also re-render the comparison figure as PNG
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    iters = np.arange(1, n_em + 1)
    axes[0].plot(iters, hist_mc.lengthscale, 'o-', label='MC',
                  color='tab:blue')
    axes[0].plot(iters, hist_gx.lengthscale, 's-', label='GMIX',
                  color='tab:orange')
    axes[0].axhline(ls_true, color='k', ls='--', alpha=0.6, label='truth')
    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel('ℓ')
    axes[0].set_title(f'Lengthscale path  (T={T})')
    axes[0].legend()

    axes[1].plot(iters, hist_mc.variance, 'o-', label='MC', color='tab:blue')
    axes[1].plot(iters, hist_gx.variance, 's-', label='GMIX',
                  color='tab:orange')
    axes[1].axhline(var_true, color='k', ls='--', alpha=0.6, label='truth')
    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_title(f'Variance path')
    axes[1].legend()

    diag_mc = evaluate('mc', mp_mc, hist_mc, drift_np, xs, t_grid,
                        ls_true, var_true, 'mc')
    from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
    h_spec0 = float(grid_at.h_per_dim[0])
    sigma0_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(ip['V0'][0]))))
    fine_N = pick_grid_size(h_spec=h_spec0, m_extent=6.0, sigma_max=sigma0_max)
    h_g = 1.0 / (fine_N * h_spec0)
    stencil_r = stencil_radius_for(jnp.asarray(ip['V0'][0])[None], h_g)
    diag_gx = evaluate('gmix', mp_gx, hist_gx, drift_np, xs, t_grid,
                        ls_true, var_true, 'gmix',
                        gmix_fine_N=fine_N, gmix_stencil_r=stencil_r)
    times = np.linspace(0, t_max, T)
    axes[2].plot(times, diag_mc['err_traj'], color='tab:blue', alpha=0.85,
                  label=f'MC  (RMSE {diag_mc["drift_rmse_traj"]:.3f})')
    axes[2].plot(times, diag_gx['err_traj'], color='tab:orange', alpha=0.85,
                  label=f'GMIX  (RMSE {diag_gx["drift_rmse_traj"]:.3f})')
    axes[2].set_xlabel('time'); axes[2].set_ylabel('||f_inferred − f_true||')
    axes[2].set_title('Drift error along true trajectory')
    axes[2].legend()
    plt.tight_layout()
    out2 = OUT_DIR / f"compare_T{T}_ls{ls_true}.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"  wrote {out2}")


if __name__ == "__main__":
    run(T=2000, ls_true=0.8, var_true=1.0, n_em=25)
