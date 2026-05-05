"""
T19: EFGP on GP-drift problem with EXACT slogdet trace (no Hutchinson).

User observation: T18 plot shows EFGP trajectories noisy at all LRs, SparseGP's
is smooth.  T18 used production Hutchinson (n_hutchinson_mstep=4) — that's the
noise.  Test: monkey-patch m_step_kernel_jax to use dense slogdet, rerun on
GP-drift at lr ∈ {0.003, 0.006, 0.01, 0.02}, see if (a) trajectory becomes
smooth and (b) walks further.

Run:
  ~/myenv/bin/python demos/diag_efgp_gp_drift_exact_slogdet.py
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
import optax

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
N_EM = 100
LR_LIST = [0.003, 0.006, 0.01, 0.02]


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
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


# ---------------------------------------------------------------------------
# Exact-slogdet replacement for m_step_kernel_jax  (drop-in)
# ---------------------------------------------------------------------------
def m_step_kernel_jax_exact(
    log_ls0, log_var0, *,
    mu_r_fixed, z_r, top, xis_flat, h_per_dim,
    D_lat, D_out,
    n_inner=10, lr=0.05,
    n_hutchinson=4, include_trace=True,
    cg_tol_trace=1e-4, max_cg_iter_trace=None,
    key=None,
):
    """Drop-in replacement for m_step_kernel_jax that uses
    jax.linalg.slogdet on the explicit A matrix instead of Hutchinson."""
    h_scalar = h_per_dim[0]
    M = int(xis_flat.shape[0])
    cdtype = top.v_fft.dtype

    opt = optax.adam(lr)
    params = (jnp.asarray(log_ls0, dtype=jnp.float32),
              jnp.asarray(log_var0, dtype=jnp.float32))
    opt_state = opt.init(params)

    def total_loss(ll, lv):
        ws_real = _ws_real_se(ll, lv, xis_flat, h_scalar, D_lat)
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
        det_loss = jax.vmap(per_r)(jnp.arange(D_out)).sum()
        sign, logdet = jnp.linalg.slogdet(A)
        return det_loss + 0.5 * D_out * logdet.real

    grad_fn = jax.jit(jax.value_and_grad(total_loss, argnums=(0, 1)))
    loss_history = []
    for step in range(n_inner):
        loss, (g_ls, g_var) = grad_fn(params[0], params[1])
        updates, opt_state = opt.update((g_ls, g_var), opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(float(loss))
    return params[0], params[1], loss_history


def fit_efgp_with_exact_slogdet(lik, op, ip, t_grid, mstep_lr):
    """Run EFGP EM with monkey-patched exact-slogdet M-step."""
    original = jpd.m_step_kernel_jax
    jpd.m_step_kernel_jax = m_step_kernel_jax_exact
    try:
        rho_sched = jnp.linspace(0.05, 0.7, N_EM)
        mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
            likelihood=lik, t_grid=t_grid,
            output_params=op, init_params=ip, latent_dim=D,
            lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
            sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
            n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
            learn_emissions=True, update_R=False,
            learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
            n_hutchinson_mstep=4,   # ignored by exact path
            kernel_warmup_iters=8,
            verbose=False)
    finally:
        jpd.m_step_kernel_jax = original
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def main():
    print(f"[T19] EFGP with EXACT slogdet trace on GP-drift", flush=True)
    print(f"[T19] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)
    xs, ys, lik, op, ip, t_grid = setup()
    results = {}
    for lr in LR_LIST:
        print(f"\n[T19] EFGP @ mstep_lr={lr} (exact slogdet)...", flush=True)
        ls_t, var_t = fit_efgp_with_exact_slogdet(lik, op, ip, t_grid, lr)
        results[lr] = (ls_t, var_t)
        print(f"  Final: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}", flush=True)
        for chk in [10, 25, 50, 75, 100]:
            print(f"    iter {chk}: ℓ={ls_t[chk]:.3f}, σ²={var_t[chk]:.3f}",
                  flush=True)
    np.savez('/tmp/T19_exact_slogdet.npz',
              **{f'lr_{lr}_ls': v[0] for lr, v in results.items()},
              **{f'lr_{lr}_var': v[1] for lr, v in results.items()})

    # Plot vs SparseGP@3e-3 + Hutchinson EFGP at same LRs (from T14/T18)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d_landscape = np.load('/tmp/T10_gp_drift_landscape.npz', allow_pickle=True)
        d11 = np.load('/tmp/T11_gp_drift_long.npz', allow_pickle=True)
        d14 = np.load('/tmp/T14_efgp_lr_sweep.npz', allow_pickle=True)
        d18 = np.load('/tmp/T18_lr_match.npz', allow_pickle=True)
        LOG_LS = d_landscape['log_ls_grid']; LOG_VAR = d_landscape['log_var_grid']
        L_grid = d_landscape['L_grid']
        L_min = float(d_landscape['L_min'])
        ll_min = float(d_landscape['ll_min']); lv_min = float(d_landscape['lv_min'])

        sp_3e3 = d11['sp_lr0.003_nem100']

        fig, ax = plt.subplots(figsize=(11, 7.5))
        L_norm = L_grid - L_min
        levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                          levels=[0] + levels, cmap='viridis_r', extend='max')
        fig.colorbar(cf, ax=ax, label='L − L_min')
        cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                        levels=levels, colors='k', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
        for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ax.plot(LOG_LS, LOG_LS * 2 + log_alpha_show,
                    color='white', linestyle=':', linewidth=0.7, alpha=0.5)

        # EFGP-Hutchinson (default) at lr=0.003 and 0.006 from T14/T18
        ax.plot(np.log(d14['mstep_lr_0.003_ls']),
                np.log(d14['mstep_lr_0.003_var']),
                '-o', color='C0', markersize=2, linewidth=0.9, alpha=0.5,
                label='EFGP HUTCH @ lr=0.003')
        ax.plot(np.log(d18['lr_0.006_ls']), np.log(d18['lr_0.006_var']),
                '-o', color='C2', markersize=2, linewidth=0.9, alpha=0.5,
                label='EFGP HUTCH @ lr=0.006')

        # EFGP-EXACT at multiple lr
        colors = ['#000080', '#0033CC', '#0066FF', '#3399FF']
        for color, lr in zip(colors, LR_LIST):
            ls_t, var_t = results[lr]
            ax.plot(np.log(ls_t), np.log(var_t), '-D',
                    color=color, markersize=3, linewidth=1.4,
                    label=f'EFGP EXACT @ lr={lr}')

        ax.plot(np.log(sp_3e3[0]), np.log(sp_3e3[1]),
                '-s', color='C3', markersize=3, linewidth=1.6,
                label='SparseGP @ lr=3e-3')

        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6, label='init')
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                    marker='X', s=240, color='red', edgecolor='k', zorder=6,
                    label='TRUTH')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=5, label='loss min')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_title('Exact slogdet vs Hutchinson — does noise explain "shorter walks"?')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig('/tmp/T19_exact_slogdet.png', dpi=140)
        print(f"\n[T19] Saved /tmp/T19_exact_slogdet.png", flush=True)
    except Exception as e:
        print(f"[T19] Plot failed: {e}", flush=True)


if __name__ == '__main__':
    main()
