"""
diag_K_landscape_bias.py

Test hypothesis: EFGP collapsed loss landscape's bias toward large ℓ is a
K-truncation artifact. With K=5 modes per dim, at small ℓ the SE spectral
density extends well beyond the truncated lattice -> loss is inflated -> the
minimum is artificially pulled toward large ℓ.

Method:
  - Use the linear-row EFGP-converged state from /tmp/bench_two_obj_linear.npz
    (or re-fit if needed).
  - For K ∈ {5, 8, 12, 16, 20, 24}, build the EFGP collapsed loss landscape
    on a (log ℓ, log σ²) grid, with the same frozen smoother.
  - Track:
    * (ℓ_min, σ²_min, L_min) per K
    * cross-section L(ℓ; σ² fixed at GT MLE) per K to visualize the shape
  - Save plot to /tmp/diag_K_landscape_bias.png

If hypothesis true: ℓ_min shifts monotonically toward smaller ℓ as K grows,
and the small-ℓ end of the cross-section flattens out (loss no longer
artificially inflated).

Run:
  ~/myenv/bin/python demos/diag_K_landscape_bias.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import _ws_real_se
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jla
from jax import vmap

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
N_EM = 50
LS_INIT = 0.7
VAR_INIT = 1.0


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

# Duffing — the EFGP-loss-bias outlier (truth wants ℓ ≈ 1.10, EFGP picks ≈ 2.3)
CFG = dict(
    name='Duffing double-well',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    T=400, t_max=15.0, sigma=0.2, N_obs=8, seed=13,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _make_data(cfg):
    """Mirrors bench_two_objectives.make_data."""
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=cfg['t_max'],
                      n_timesteps=cfg['T'], sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = rng.standard_normal((cfg['N_obs'], D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_efgp(lik, op, ip, t_grid, sigma):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    ls_final = float(hist.lengthscale[-1])
    var_final = float(hist.variance[-1])
    return mp, ls_final, var_final


def _build_grid_and_smoother(mp, t_grid, sigma, ls_pin, var_pin, K_per_dim):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin, dtype=jnp.float32)),
        K_per_dim=K_per_dim, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r)
    z_r = h0 / ws_safe
    return grid, top, z_r


def _efgp_loss_at(grid, top, z_r, log_ls, log_var):
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                            jnp.asarray(log_var, dtype=jnp.float32),
                            grid.xis_flat, grid.h_per_dim[0], D)
    ws_c = ws_real.astype(cdtype)
    v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    ns_v = tuple(2 * n - 1 for n in top.ns)
    v_conv = v_pad[tuple(slice(0, L) for L in ns_v)]
    d_ = len(top.ns)
    mi = jnp.indices(top.ns).reshape(d_, -1)
    offset = jnp.array([n - 1 for n in top.ns], dtype=jnp.int32)
    diff = mi[:, :, None] - mi[:, None, :] + offset[:, None, None]
    T_mat = v_conv[tuple(diff[k] for k in range(d_))]
    eye_c = jnp.eye(M, dtype=cdtype)
    A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
    L = jnp.linalg.cholesky(A)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
    h = ws_c[None, :] * z_r
    def solve_one(b):
        y = jla.solve_triangular(L, b, lower=True)
        return jla.solve_triangular(L.conj().T, y, lower=False)
    mu = jax.vmap(solve_one)(h)
    det_loss = -0.5 * jnp.sum(jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
    return float(det_loss + 0.5 * D * logdet)


def main():
    print(f"[diag] EFGP-loss K-truncation bias diagnostic ({CFG['name']})",
          flush=True)
    cfg = CFG
    sigma = cfg['sigma']

    # Reuse converged smoother from previous bench if available
    npz_path = '/tmp/bench_two_obj_linear.npz'
    have_npz = Path(npz_path).is_file()
    if have_npz:
        d_npz = np.load(npz_path, allow_pickle=True)
        ls_pin = float(d_npz['e_ls'][-1])
        var_pin = float(d_npz['e_var'][-1])
        print(f'[diag] using cached EFGP-converged hypers '
              f'(ℓ={ls_pin:.3f}, σ²={var_pin:.3f}) from bench npz', flush=True)
        print('[diag] re-running EFGP fit to capture full smoother (mp)...',
              flush=True)
    xs, ys, lik, op, ip, t_grid = _make_data(cfg)

    print('[diag] fitting EFGP for converged smoother...', flush=True)
    t0 = time.perf_counter()
    mp, ls_final, var_final = fit_efgp(lik, op, ip, t_grid, sigma)
    print(f'[diag]   final ℓ={ls_final:.3f}, σ²={var_final:.3f}, '
          f'wall={time.perf_counter()-t0:.1f}s', flush=True)

    # Set up landscape grid
    LOG_LS = np.linspace(-2.0, 1.5, 21)
    LOG_VAR = np.linspace(-2.5, 2.0, 21)

    K_LIST = [5, 8, 12, 16, 20, 24]
    out = {}

    for K in K_LIST:
        print(f'[diag] K={K}...', flush=True)
        t0 = time.perf_counter()
        grid, top, z_r = _build_grid_and_smoother(
            mp, t_grid, sigma, ls_final, var_final, K_per_dim=K)
        print(f'    M={grid.M}, mtot={grid.mtot_per_dim}', flush=True)

        L = np.zeros((len(LOG_LS), len(LOG_VAR)))
        for i, ll in enumerate(LOG_LS):
            for j, lv in enumerate(LOG_VAR):
                L[i, j] = _efgp_loss_at(grid, top, z_r, float(ll), float(lv))
        idx = np.unravel_index(np.argmin(L), L.shape)
        ll_min = float(LOG_LS[idx[0]]); lv_min = float(LOG_VAR[idx[1]])
        L_min = float(L[idx])
        print(f'    landscape: {time.perf_counter()-t0:.1f}s, '
              f'min ℓ={math.exp(ll_min):.3f}, σ²={math.exp(lv_min):.3f}, '
              f'L_min={L_min:.2f}', flush=True)
        out[K] = dict(L=L, M=int(grid.M), mtot=tuple(int(m) for m in grid.mtot_per_dim),
                       ll_min=ll_min, lv_min=lv_min, L_min=L_min,
                       LOG_LS=LOG_LS, LOG_VAR=LOG_VAR)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nK = len(K_LIST)
    fig, axes = plt.subplots(2, nK, figsize=(4 * nK, 8))

    # Top row: contour plot per K
    for col, K in enumerate(K_LIST):
        ax = axes[0, col]
        d = out[K]
        L_norm = d['L'] - d['L_min']
        finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            levels = list(np.logspace(np.log10(max(lo, hi/1000)),
                                       np.log10(hi), 10))
        else:
            levels = [0.5, 1, 2, 5, 10, 50, 100, 500]
        ax.contourf(d['LOG_LS'], d['LOG_VAR'], L_norm.T,
                     levels=[0]+levels, cmap='viridis_r', extend='max')
        cs = ax.contour(d['LOG_LS'], d['LOG_VAR'], L_norm.T,
                         levels=levels, colors='k', linewidths=0.4)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
        ax.scatter([d['ll_min']], [d['lv_min']], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=7,
                    label=f"min: ℓ={math.exp(d['ll_min']):.2f}, "
                          f"σ²={math.exp(d['lv_min']):.2f}")
        ax.set_xlabel('log ℓ'); ax.set_ylabel('log σ²')
        ax.set_title(f"K={K} per dim, M={d['M']}")
        ax.legend(loc='lower right', fontsize=8)

    # Bottom row: cross-section L(ℓ) at fixed σ² close to the K=24 minimum
    sigma_sq_ref_idx = np.argmin(np.abs(LOG_VAR - out[K_LIST[-1]]['lv_min']))
    for col, K in enumerate(K_LIST):
        ax = axes[1, col]
        d = out[K]
        L_norm = d['L'] - d['L_min']
        ax.plot(LOG_LS, L_norm[:, sigma_sq_ref_idx], '-o',
                markersize=3, color=f'C{col}',
                label=f"K={K}")
        ax.axvline(d['ll_min'], color='gold', linestyle='--', alpha=0.8,
                    label=f"min ℓ={math.exp(d['ll_min']):.2f}")
        ax.set_xlabel('log ℓ')
        ax.set_ylabel('L(ℓ) - L_min')
        ax.set_title(f"K={K}: cross-section at log σ²={LOG_VAR[sigma_sq_ref_idx]:.2f}")
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale('symlog')

    fig.suptitle(f'EFGP collapsed-loss landscape vs K (linear row, frozen smoother @ '
                 f'EFGP-converged)\nK_FD=5 used in production '
                 f'-- if minimum shifts left as K grows, K-truncation bias confirmed',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/tmp/diag_K_landscape_bias.png', dpi=110)
    print('[diag] Saved /tmp/diag_K_landscape_bias.png', flush=True)

    # Save raw data
    np.savez('/tmp/diag_K_landscape_bias.npz',
              K_list=np.array(K_LIST),
              ll_mins=np.array([out[K]['ll_min'] for K in K_LIST]),
              lv_mins=np.array([out[K]['lv_min'] for K in K_LIST]),
              L_mins=np.array([out[K]['L_min'] for K in K_LIST]),
              **{f'L_K{K}': out[K]['L'] for K in K_LIST},
              LOG_LS=LOG_LS, LOG_VAR=LOG_VAR,
              ls_final=ls_final, var_final=var_final)
    print('[diag] Saved /tmp/diag_K_landscape_bias.npz', flush=True)

    # Summary
    print('\n[diag] Summary: how does the loss minimum shift with K?', flush=True)
    print(f"  {'K':>4s}  {'M':>5s}  {'ℓ_min':>8s}  {'σ²_min':>8s}  {'L_min':>10s}",
          flush=True)
    for K in K_LIST:
        d = out[K]
        print(f"  {K:>4d}  {d['M']:>5d}  {math.exp(d['ll_min']):>8.3f}  "
              f"{math.exp(d['lv_min']):>8.3f}  {d['L_min']:>10.2f}",
              flush=True)


if __name__ == '__main__':
    main()
