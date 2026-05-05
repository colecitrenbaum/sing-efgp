"""
T9: Does EFGP wander because of Hutchinson noise, or for a structural reason?

Approach. Patch ``m_step_kernel_jax`` in-place to use an exact dense
slogdet trace gradient (no Hutchinson, no CG). Run EFGP on Duffing with
the same init/schedule as T8. Overlay on the same landscape.

If EFGP's trajectory becomes much more direct (converges to loss min and
stops), the wandering was Hutchinson noise + Adam interaction.
If it still wanders along the ridge, the issue is in the
deterministic-gradient + Adam dynamics — i.e., something about Adam
following along the (mostly-flat) ridge direction.

Run:
  ~/myenv/bin/python demos/diag_trajectories_exact_trace.py
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
SIGMA = 0.2
N_OBS = 8
SEED = 13
T_MAX = 15.0
N_EM = 20
LS_INIT = 1.0
VAR_INIT = 1.0


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def setup():
    drift = lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]])
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.2, 0.0]),
                      f=drift, t_max=T_MAX, n_timesteps=T,
                      sigma=sigma_fn)
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
    t_grid = jnp.linspace(0., T_MAX, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op_init, ip_init, t_grid


# ---------------------------------------------------------------------------
# Exact-slogdet replacement for m_step_kernel_jax
# ---------------------------------------------------------------------------
def m_step_kernel_jax_exact(
    log_ls0, log_var0, *,
    mu_r_fixed, z_r, top, xis_flat, h_per_dim,
    D_lat, D_out,
    n_inner=10, lr=0.05,
    n_hutchinson=4, include_trace=True,    # ignored
    cg_tol_trace=1e-4, max_cg_iter_trace=None,
    key=None,
):
    """Drop-in replacement for m_step_kernel_jax that uses
    jax.linalg.slogdet on the explicit A matrix instead of Hutchinson.

    Same signature as m_step_kernel_jax — all kwargs accepted (some ignored)."""
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
        # Refresh μ at this θ (envelope-correct)
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


def fit_efgp_with_traj(lik, op, ip, t_grid, *, exact_slogdet: bool):
    """Run EFGP EM, capture (ℓ, σ²) per outer iter."""
    if exact_slogdet:
        # Monkey-patch m_step_kernel_jax for the duration of this run
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
            learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
            n_hutchinson_mstep=4, kernel_warmup_iters=8,
            verbose=False,
        )
    finally:
        if exact_slogdet:
            jpd.m_step_kernel_jax = original
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def main():
    print(f"[T9] JAX devices: {jax.devices()}", flush=True)
    print(f"[T9] Duffing T={T}, sigma={SIGMA}, n_em={N_EM}, init=(ℓ=1.0, σ²=1.0)",
          flush=True)
    xs, ys, lik, op, ip, t_grid = setup()

    print("\n[T9] Running EFGP with HUTCHINSON trace (production)...",
          flush=True)
    h_ls, h_var = fit_efgp_with_traj(lik, op, ip, t_grid, exact_slogdet=False)
    print(f"  Final: ℓ={h_ls[-1]:.3f}, σ²={h_var[-1]:.3f}", flush=True)

    print("\n[T9] Running EFGP with EXACT slogdet trace (no Hutchinson)...",
          flush=True)
    e_ls, e_var = fit_efgp_with_traj(lik, op, ip, t_grid, exact_slogdet=True)
    print(f"  Final: ℓ={e_ls[-1]:.3f}, σ²={e_var[-1]:.3f}", flush=True)

    np.savez('/tmp/T9_duffing_exact_trace.npz',
              hutch_ls=h_ls, hutch_var=h_var,
              exact_ls=e_ls, exact_var=e_var)
    print("\n[T9] Saved /tmp/T9_duffing_exact_trace.npz", flush=True)

    # Render overlay
    try:
        landscape_npz = '/tmp/T7_duffing_landscape.npz'
        traj_npz = '/tmp/T8_duffing_trajectories.npz'
        data = np.load(landscape_npz, allow_pickle=True)
        L_grid = data['L_grid']
        LOG_LS = data['log_ls_grid']; LOG_VAR = data['log_var_grid']
        L_min = float(data['L_min'])
        ll_min = float(data['ll_min']); lv_min = float(data['lv_min'])

        sp_data = np.load(traj_npz, allow_pickle=True)
        sp1_ls = sp_data['sp1_ls']; sp1_var = sp_data['sp1_var']
        sp2_ls = sp_data.get('sp2_ls'); sp2_var = sp_data.get('sp2_var')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7))
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

        # Overlay trajectories
        ax.plot(np.log(h_ls), np.log(h_var), '-o', color='C0',
                markersize=4, linewidth=1.5,
                label='EFGP (Hutchinson n=4)', zorder=5)
        ax.plot(np.log(e_ls), np.log(e_var), '-o', color='C2',
                markersize=4, linewidth=1.5,
                label='EFGP (exact slogdet)', zorder=5)
        ax.plot(np.log(sp1_ls), np.log(sp1_var), '-s', color='C1',
                markersize=4, linewidth=1.5, label='SparseGP @ lr=1e-3',
                zorder=4)
        if sp2_ls is not None:
            ax.plot(np.log(sp2_ls), np.log(sp2_var), '-^', color='C3',
                    markersize=4, linewidth=1.5,
                    label='SparseGP @ lr=2e-3', zorder=4)

        ax.scatter([np.log(h_ls[0])], [np.log(h_var[0])],
                    marker='+', s=200, color='black', zorder=6,
                    label='init (ℓ=1, σ²=1)')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=5,
                    label=f'loss min: ℓ={math.exp(ll_min):.2f}, σ²={math.exp(lv_min):.2f}')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_title('Duffing: EFGP with Hutchinson vs exact slogdet trace\n'
                      '(white dotted: constant α=σ²/ℓ²)')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig('/tmp/T9_duffing_exact_trace.png', dpi=150)
        print('[T9] Saved /tmp/T9_duffing_exact_trace.png', flush=True)
    except Exception as e:
        print(f'[T9] Plot rendering failed: {e}', flush=True)


if __name__ == '__main__':
    main()
