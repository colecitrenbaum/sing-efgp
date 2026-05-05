"""
T12: Compute the EFGP loss landscape at the q(x) EFGP ACTUALLY sees.

The T10 landscape was computed at a SparseGP-converged smoother at TRUE
hypers. But EFGP's M-step never sees that smoother — EFGP's q(x) evolves
from (ℓ_init, σ²_init) through 8 warmup iters before the M-step kicks in.
The landscape EFGP optimizes is therefore *different* from the static
T10 landscape.

This script:
  1. Runs EFGP for 8 kernel-warmup iters (no M-step) at (ℓ_init=2.0,
     σ²_init=0.3) → captures the warmed-up smoother.
  2. Computes the EFGP collapsed loss landscape at THIS smoother.
  3. Compares its loss minimum to: truth, init, EFGP's eventual final
     hypers (from T10).

If EFGP's final (ℓ, σ²) is at/near the min of THIS landscape, EFGP is
correctly minimizing its actual objective; the "wrong direction" on the
T10 plot was just an artifact of using the wrong reference smoother.
If EFGP's final is far from the min of its own landscape, there's a real
M-step bug.

Run:
  ~/myenv/bin/python demos/diag_efgp_own_landscape.py
"""
from __future__ import annotations

import math
import sys
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

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


D = 2
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0
LS_INIT = 2.0
VAR_INIT = 0.3
K_FD = 8


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps

    def drift(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    return drift


def setup():
    drift = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=vt[:D].T, d=ys.mean(0),
                   R=jnp.full((N_OBS,), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op_init, ip_init, t_grid


def run_efgp_capture_q(lik, op, ip, t_grid, n_em_warmup=8):
    """Run EFGP for n_em_warmup outer iters with NO kernel M-step
    (kernel_warmup_iters=n_em_warmup means M-step never fires). Returns
    the smoother q(x) at the end of warmup, plus the final marginals."""
    rho_sched = jnp.linspace(0.05, 0.7, n_em_warmup)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em_warmup, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=False,   # M-step OFF
        kernel_warmup_iters=n_em_warmup,
        verbose=False)
    return mp


def run_efgp_to_completion(lik, op, ip, t_grid, n_em=25):
    """Run a full EFGP EM with M-step, return final q(x) and hyper trajectory."""
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    return mp, np.array(hist.lengthscale), np.array(hist.variance)


def build_landscape_state(mp, t_grid, ls_pin, var_pin, eps_grid=1e-3):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=eps_grid)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=16,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r_init)
    z_r = h0 / ws_safe
    return grid, top, z_r


def loss_at(log_ls, log_var, *, grid, top, z_r):
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                          jnp.asarray(log_var, dtype=jnp.float32),
                          grid.xis_flat, grid.h_per_dim[0], D)
    ws_c = ws_real.astype(cdtype)
    eye_c = jnp.eye(M, dtype=cdtype)
    T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
    T_mat = T_cols.T
    D_w = ws_c[:, None] * eye_c
    A = eye_c + D_w @ T_mat @ D_w
    h = ws_c[None, :] * z_r
    mu = jax.vmap(lambda b: jnp.linalg.solve(A, b))(h)

    def per_r(r):
        t1 = jnp.vdot(h[r], mu[r]).real
        t2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
        return -(t1 - t2)
    det_loss = jax.vmap(per_r)(jnp.arange(D)).sum()
    sign, logdet = jnp.linalg.slogdet(A)
    return float(det_loss + 0.5 * D * logdet.real)


def landscape_summary(label, *, grid, top, z_r):
    """Compute landscape, find min, return useful summary stats."""
    LOG_LS = np.linspace(-2.0, 1.5, 25)
    LOG_VAR = np.linspace(-3.0, 2.5, 25)
    L = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            L[i, j] = loss_at(float(ll), float(lv),
                                grid=grid, top=top, z_r=z_r)
    idx_min = np.unravel_index(np.argmin(L), L.shape)
    ll_min = float(LOG_LS[idx_min[0]])
    lv_min = float(LOG_VAR[idx_min[1]])
    L_min = float(L[idx_min])
    print(f"  [{label}] loss min: ℓ={math.exp(ll_min):.3f}, "
          f"σ²={math.exp(lv_min):.3f}, α={math.exp(lv_min-2*ll_min):.3f}",
          flush=True)
    return dict(LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L=L, L_min=L_min,
                ll_min=ll_min, lv_min=lv_min, label=label)


def main():
    print(f"[T12] JAX devices: {jax.devices()}", flush=True)
    print(f"[T12] EFGP loss landscape at MULTIPLE smoother choices",
          flush=True)
    print(f"[T12] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}, α={VAR_TRUE/LS_TRUE**2:.3f}",
          flush=True)
    print(f"[T12] init:  ℓ={LS_INIT}, σ²={VAR_INIT}, α={VAR_INIT/LS_INIT**2:.3f}",
          flush=True)

    xs, ys, lik, op, ip, t_grid = setup()

    # 1. q(x) at TRUE hypers (8 warmup iters with hypers fixed at truth)
    print(f"\n[T12] q(x) at TRUE hypers (warmed up at ℓ=0.8, σ²=1.0)...",
          flush=True)
    rho_sched_8 = jnp.linspace(0.05, 0.7, 8)
    mp_truth, *_ = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_TRUE, variance=VAR_TRUE, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=8, n_estep_iters=10, rho_sched=rho_sched_8,
        learn_emissions=True, update_R=False, learn_kernel=False,
        kernel_warmup_iters=8, verbose=False)
    grid_truth, top_truth, z_r_truth = build_landscape_state(
        mp_truth, t_grid, LS_TRUE, VAR_TRUE)

    # 2. q(x) at INIT hypers (8 warmup iters with hypers fixed at init)
    #    This is the smoother EFGP's M-step actually FIRST sees
    print(f"\n[T12] q(x) at INIT hypers (warmed up at ℓ=2.0, σ²=0.3)...",
          flush=True)
    mp_warmup_init = run_efgp_capture_q(lik, op, ip, t_grid, n_em_warmup=8)
    grid_init, top_init, z_r_init = build_landscape_state(
        mp_warmup_init, t_grid, LS_INIT, VAR_INIT)

    # 3. Run EFGP fully and get its FINAL q(x)
    print(f"\n[T12] Running EFGP n_em=25 fully (with M-step)...", flush=True)
    mp_efgp_final, ls_traj, var_traj = run_efgp_to_completion(
        lik, op, ip, t_grid, n_em=25)
    ls_final = ls_traj[-1]; var_final = var_traj[-1]
    print(f"  EFGP final hypers: ℓ={ls_final:.3f}, σ²={var_final:.3f}",
          flush=True)
    grid_final, top_final, z_r_final = build_landscape_state(
        mp_efgp_final, t_grid, float(ls_final), float(var_final))

    print(f"\n[T12] Comparing landscape minima across smoother choices:",
          flush=True)
    print(f"  Truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}, α={VAR_TRUE/LS_TRUE**2:.3f}",
          flush=True)

    summaries = {}
    summaries['truth'] = landscape_summary(
        '@truth-warmup', grid=grid_truth, top=top_truth, z_r=z_r_truth)
    summaries['init'] = landscape_summary(
        '@init-warmup', grid=grid_init, top=top_init, z_r=z_r_init)
    summaries['efgp_final'] = landscape_summary(
        '@efgp-final', grid=grid_final, top=top_final, z_r=z_r_final)

    # Save
    np.savez('/tmp/T12_efgp_own_landscape.npz',
              **{f'{k}_LOG_LS': v['LOG_LS'] for k, v in summaries.items()},
              **{f'{k}_LOG_VAR': v['LOG_VAR'] for k, v in summaries.items()},
              **{f'{k}_L': v['L'] for k, v in summaries.items()},
              **{f'{k}_ll_min': v['ll_min'] for k, v in summaries.items()},
              **{f'{k}_lv_min': v['lv_min'] for k, v in summaries.items()},
              **{f'{k}_L_min': v['L_min'] for k, v in summaries.items()},
              ls_init=LS_INIT, var_init=VAR_INIT,
              ls_true=LS_TRUE, var_true=VAR_TRUE,
              ls_final=ls_final, var_final=var_final,
              ls_traj=ls_traj, var_traj=var_traj)
    print(f"\n[T12] Saved /tmp/T12_efgp_own_landscape.npz", flush=True)

    # Render: 3-panel plot showing loss landscape at each smoother
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, (key, s) in zip(axes, summaries.items()):
            L_norm = s['L'] - s['L_min']
            levels = [0.5, 1, 2, 5, 10, 20, 50]
            cf = ax.contourf(s['LOG_LS'], s['LOG_VAR'], L_norm.T,
                              levels=[0] + levels, cmap='viridis_r', extend='max')
            cs = ax.contour(s['LOG_LS'], s['LOG_VAR'], L_norm.T,
                             levels=levels, colors='k', linewidths=0.5)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
            ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                        marker='+', s=200, color='black', zorder=6)
            ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                        marker='X', s=180, color='red', edgecolor='k', zorder=6)
            ax.scatter([math.log(float(ls_final))], [math.log(float(var_final))],
                        marker='o', s=100, color='C0', edgecolor='k', zorder=6)
            ax.scatter([s['ll_min']], [s['lv_min']],
                        marker='*', s=200, color='gold', edgecolor='k', zorder=5)
            for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                ax.plot(s['LOG_LS'], s['LOG_LS'] * 2 + log_alpha_show,
                        color='white', linestyle=':', linewidth=0.6, alpha=0.5)
            ax.plot(np.log(ls_traj), np.log(var_traj), '-o', color='C0',
                    markersize=3, linewidth=1.0, alpha=0.7)
            ax.set_xlabel('log ℓ')
            ax.set_xlim(s['LOG_LS'].min(), s['LOG_LS'].max())
            ax.set_ylim(s['LOG_VAR'].min(), s['LOG_VAR'].max())
            ax.set_title(f"q(x) at {key}\nloss min: ℓ={math.exp(s['ll_min']):.2f}, "
                          f"σ²={math.exp(s['lv_min']):.2f}, "
                          f"α={math.exp(s['lv_min']-2*s['ll_min']):.2f}",
                          fontsize=10)
        axes[0].set_ylabel('log σ²')
        fig.suptitle("EFGP loss landscape under different smoother choices "
                     "(red ✕ truth, gold ★ landscape min, blue ○ EFGP final, + init)",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig('/tmp/T12_efgp_own_landscape.png', dpi=140)
        print(f"[T12] Saved /tmp/T12_efgp_own_landscape.png", flush=True)
    except Exception as e:
        print(f"[T12] Plot rendering failed: {e}", flush=True)


if __name__ == '__main__':
    main()
