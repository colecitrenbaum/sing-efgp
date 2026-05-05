"""
T20: Does Hutchinson at large n + tight CG converge to exact-slogdet
endpoint?

User's sharp question: Hutchinson is unbiased; at enough probes it should
match exact. T19 showed Hutchinson and exact EM converge to DIFFERENT
fixed points despite identical setup otherwise. Possible explanations:
  (a) n_hutchinson_mstep=4 just too few probes — Adam interaction with
      noise reduces effective step
  (b) cg_tol_trace=1e-4 too loose — CG truncation introduces bias
  (c) bug somewhere in hutchinson_trace_grads (dws formula, dA_apply, etc.)

Test: same GP-drift problem, EFGP at mstep_lr=0.003, n_em=100. Sweep:
  - n_hutchinson_mstep ∈ {4, 16, 64, 256}
  - cg_tol_trace ∈ {1e-4, 1e-7}
And exact-slogdet for reference. If Hutchinson@n=256+cg=1e-7 matches exact
endpoint, then (a)+(b) explain it. If still off, there's a real bug.

Run:
  ~/myenv/bin/python demos/diag_hutch_n_cgtol_sweep.py
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
MSTEP_LR = 0.003

# Each row: (label, n_hutch, cg_tol_trace) — cg_tol_trace=None means use exact slogdet
CONFIGS = [
    ('exact slogdet',         None, None),
    ('hutch n=4,   cg=1e-4',  4,    1e-4),  # production default
    ('hutch n=16,  cg=1e-4',  16,   1e-4),
    ('hutch n=64,  cg=1e-4',  64,   1e-4),
    ('hutch n=256, cg=1e-4',  256,  1e-4),
    ('hutch n=4,   cg=1e-7',  4,    1e-7),
    ('hutch n=16,  cg=1e-7',  16,   1e-7),
    ('hutch n=64,  cg=1e-7',  64,   1e-7),
    ('hutch n=256, cg=1e-7',  256,  1e-7),
]


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


# Drop-in replacement m_step that uses dense slogdet
def m_step_kernel_jax_exact(
    log_ls0, log_var0, *,
    mu_r_fixed, z_r, top, xis_flat, h_per_dim,
    D_lat, D_out,
    n_inner=10, lr=0.05,
    n_hutchinson=4, include_trace=True,
    cg_tol_trace=1e-4, max_cg_iter_trace=None,
    key=None,
):
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


def fit(lik, op, ip, t_grid, *, n_hutch, cg_tol_trace, mstep_lr=MSTEP_LR):
    """If n_hutch is None, use exact slogdet. Else use Hutchinson at given n,cg_tol."""
    if n_hutch is None:
        original = jpd.m_step_kernel_jax
        jpd.m_step_kernel_jax = m_step_kernel_jax_exact
        try:
            return _fit(lik, op, ip, t_grid, n_hutch=1, cg_tol_trace=1e-4,
                          mstep_lr=mstep_lr)
        finally:
            jpd.m_step_kernel_jax = original
    else:
        return _fit(lik, op, ip, t_grid, n_hutch=n_hutch,
                      cg_tol_trace=cg_tol_trace, mstep_lr=mstep_lr)


def _fit(lik, op, ip, t_grid, *, n_hutch, cg_tol_trace, mstep_lr):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    # cg_tol_trace is a parameter of m_step_kernel_jax (fixed default 1e-4
    # in production). To make it tunable here we'd need to thread it through
    # fit_efgp_sing_jax — but it's not exposed.  Workaround: monkey-patch
    # m_step_kernel_jax to use the desired cg_tol_trace.
    if cg_tol_trace != 1e-4:
        original = jpd.m_step_kernel_jax
        def m_step_with_cg_tol(*args, **kwargs):
            kwargs['cg_tol_trace'] = cg_tol_trace
            return original(*args, **kwargs)
        jpd.m_step_kernel_jax = m_step_with_cg_tol
    else:
        original = None
    try:
        mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
            likelihood=lik, t_grid=t_grid,
            output_params=op, init_params=ip, latent_dim=D,
            lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
            sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
            n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
            learn_emissions=True, update_R=False,
            learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
            n_hutchinson_mstep=n_hutch,
            kernel_warmup_iters=8, verbose=False)
    finally:
        if original is not None:
            jpd.m_step_kernel_jax = original
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def main():
    print(f"[T20] Hutchinson n × cg_tol sweep on GP-drift", flush=True)
    print(f"[T20] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}, mstep_lr={MSTEP_LR}",
          flush=True)
    xs, ys, lik, op, ip, t_grid = setup()
    results = {}
    for label, n_hutch, cg_tol in CONFIGS:
        print(f"\n[T20] {label}...", flush=True)
        ls_t, var_t = fit(lik, op, ip, t_grid,
                            n_hutch=n_hutch, cg_tol_trace=cg_tol)
        results[label] = (ls_t, var_t)
        print(f"  Final: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}", flush=True)

    print(f"\n[T20] Endpoint summary:", flush=True)
    print(f"  {'config':30s}  {'ℓ':>7}  {'σ²':>7}  "
          f"{'Δ from exact ℓ':>14}  {'Δ from exact σ²':>16}", flush=True)
    ex_ls = results['exact slogdet'][0][-1]
    ex_var = results['exact slogdet'][1][-1]
    for label, _, _ in CONFIGS:
        ls_f = results[label][0][-1]
        var_f = results[label][1][-1]
        d_ls = ls_f - ex_ls
        d_var = var_f - ex_var
        print(f"  {label:30s}  {ls_f:>7.3f}  {var_f:>7.3f}  "
              f"{d_ls:>+14.4f}  {d_var:>+16.4f}", flush=True)

    np.savez('/tmp/T20_hutch_sweep.npz',
              **{label.replace(' ', '_').replace(',', '_').replace('=', '_'): np.stack(v)
                  for label, v in results.items()})

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d_landscape = np.load('/tmp/T10_gp_drift_landscape.npz', allow_pickle=True)
        LOG_LS = d_landscape['log_ls_grid']; LOG_VAR = d_landscape['log_var_grid']
        L_grid = d_landscape['L_grid']
        L_min = float(d_landscape['L_min'])
        ll_min = float(d_landscape['ll_min']); lv_min = float(d_landscape['lv_min'])

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

        # cg=1e-4 family in one color cycle, cg=1e-7 family in another
        cg4_colors = ['#1f77b4', '#aec7e8', '#7f7f7f', '#c7c7c7']
        cg7_colors = ['#2ca02c', '#98df8a', '#9467bd', '#c5b0d5']
        cg4_idx = 0; cg7_idx = 0
        for label, n_hutch, cg_tol in CONFIGS:
            ls_t, var_t = results[label]
            if n_hutch is None:
                ax.plot(np.log(ls_t), np.log(var_t), '-D',
                        color='gold', markersize=4, linewidth=2.0, zorder=6,
                        label=label)
            elif cg_tol == 1e-4:
                ax.plot(np.log(ls_t), np.log(var_t), '-o',
                        color=cg4_colors[cg4_idx], markersize=2,
                        linewidth=0.9, alpha=0.6, label=label)
                cg4_idx += 1
            else:
                ax.plot(np.log(ls_t), np.log(var_t), '-o',
                        color=cg7_colors[cg7_idx], markersize=2,
                        linewidth=0.9, alpha=0.6, label=label)
                cg7_idx += 1

        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=7, label='init')
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                    marker='X', s=240, color='red', edgecolor='k', zorder=7,
                    label='TRUTH')
        ax.scatter([ll_min], [lv_min], marker='*', s=320,
                    color='gold', edgecolor='k', zorder=8, label='loss min')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_title('Does Hutchinson at large n + tight CG match exact slogdet?')
        ax.legend(loc='lower right', fontsize=7)
        plt.tight_layout()
        plt.savefig('/tmp/T20_hutch_sweep.png', dpi=140)
        print(f"\n[T20] Saved /tmp/T20_hutch_sweep.png", flush=True)
    except Exception as e:
        print(f"[T20] Plot failed: {e}", flush=True)


if __name__ == '__main__':
    main()
