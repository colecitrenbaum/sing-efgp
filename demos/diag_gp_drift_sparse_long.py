"""
T11: Where does SparseGP *actually* converge on the GP-drift problem?

User's sharp observation (T10 plot): SparseGP walked toward the truth in
both ℓ and σ², while EFGP walked toward truth only in σ² (wrong direction
on ℓ). But SparseGP at lr=1e-3 barely moved (~0.05 on log_ℓ) — so we
can't tell from one trajectory whether its small motion is genuinely
better-directed or just the gradient at init that hasn't propagated yet.

Test: run SparseGP at lr ∈ {1e-3, 3e-3, 1e-2} for n_em = 100, see where
it converges. If it walks past truth toward the EFGP endpoint, the
trajectories converge to the same place; if it stops at/near truth,
SparseGP genuinely has a less-biased objective.

Also rerun EFGP for n_em=100 to see if it pulls back toward truth at
longer horizon.

Run:
  ~/myenv/bin/python demos/diag_gp_drift_sparse_long.py
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
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0
LS_INIT = 2.0
VAR_INIT = 0.3
N_EM_LIST = [25, 50, 100]
SP_LR_LIST = [1e-3, 3e-3]


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


def fit_efgp(lik, op, ip, t_grid, n_em):
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
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def fit_sparsegp(lik, op, ip, t_grid, n_em, lr):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    history = []
    fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=SIGMA, rho_sched=rho_sched, n_iters=n_em, n_iters_e=10,
        n_iters_m=4, perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((n_em,), lr),
        print_interval=999, drift_params_history=history)
    ls_traj = np.array([LS_INIT] + [float(np.mean(np.asarray(d['length_scales'])))
                                      for d in history])
    var_traj = np.array([VAR_INIT] + [float(np.asarray(d['output_scale'])) ** 2
                                       for d in history])
    return ls_traj, var_traj


def main():
    print(f"[T11] JAX devices: {jax.devices()}", flush=True)
    print(f"[T11] GP-drift convergence test: where does each method actually go?",
          flush=True)
    print(f"[T11] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}, α={VAR_TRUE/LS_TRUE**2:.3f}",
          flush=True)
    print(f"[T11] init:  ℓ={LS_INIT}, σ²={VAR_INIT}, α={VAR_INIT/LS_INIT**2:.3f}",
          flush=True)
    xs, ys, lik, op, ip, t_grid = setup()

    results = {}

    print("\n[T11] Running EFGP at varied n_em...", flush=True)
    for n_em in N_EM_LIST:
        ls_t, var_t = fit_efgp(lik, op, ip, t_grid, n_em)
        a = var_t[-1] / ls_t[-1]**2
        print(f"  n_em={n_em:>3}: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}, α={a:.3f}",
              flush=True)
        results[f'efgp_nem{n_em}'] = (ls_t, var_t)

    for lr in SP_LR_LIST:
        print(f"\n[T11] Running SparseGP @ lr={lr} at varied n_em...",
              flush=True)
        for n_em in N_EM_LIST:
            try:
                ls_t, var_t = fit_sparsegp(lik, op, ip, t_grid, n_em, lr)
                a = var_t[-1] / ls_t[-1]**2
                print(f"  n_em={n_em:>3}: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}, "
                      f"α={a:.3f}", flush=True)
                results[f'sp_lr{lr}_nem{n_em}'] = (ls_t, var_t)
            except Exception as e:
                print(f"  n_em={n_em:>3}: NaN'd ({e})", flush=True)

    np.savez('/tmp/T11_gp_drift_long.npz',
              **{k: np.stack(v) for k, v in results.items()
                  if k.startswith('efgp')},
              **{k: np.stack(v) for k, v in results.items()
                  if k.startswith('sp')},
              ls_true=LS_TRUE, var_true=VAR_TRUE,
              ls_init=LS_INIT, var_init=VAR_INIT)
    print("\n[T11] Saved /tmp/T11_gp_drift_long.npz", flush=True)

    # Plot: overlay all trajectories on the T10 landscape
    try:
        landscape = np.load('/tmp/T10_gp_drift_landscape.npz', allow_pickle=True)
        L_grid = landscape['L_grid']
        LOG_LS = landscape['log_ls_grid']; LOG_VAR = landscape['log_var_grid']
        L_min = float(landscape['L_min'])
        ll_min = float(landscape['ll_min']); lv_min = float(landscape['lv_min'])

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7.5))
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

        # EFGP at varying n_em (different colors)
        efgp_colors = {25: 'C0', 50: '#0066ff', 100: '#000080'}
        for n_em in N_EM_LIST:
            key = f'efgp_nem{n_em}'
            if key not in results:
                continue
            ls_t, var_t = results[key]
            ax.plot(np.log(ls_t), np.log(var_t), '-o', color=efgp_colors[n_em],
                    markersize=3, linewidth=1.2,
                    label=f'EFGP n_em={n_em}', zorder=4)

        # SparseGP at varying (lr, n_em)
        sp_colors = {1e-3: 'C1', 3e-3: 'C3'}
        sp_styles = {25: '-', 50: '--', 100: ':'}
        for lr in SP_LR_LIST:
            for n_em in N_EM_LIST:
                key = f'sp_lr{lr}_nem{n_em}'
                if key not in results:
                    continue
                ls_t, var_t = results[key]
                ax.plot(np.log(ls_t), np.log(var_t),
                        sp_styles[n_em], marker='s', color=sp_colors[lr],
                        markersize=3, linewidth=1.2,
                        label=f'Sparse lr={lr:g} n_em={n_em}', zorder=4)

        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6,
                    label=f'init (ℓ={LS_INIT}, σ²={VAR_INIT})')
        ax.scatter([math.log(LS_TRUE)], [math.log(VAR_TRUE)],
                    marker='X', s=240, color='red', edgecolor='k', zorder=6,
                    label=f'TRUTH (ℓ={LS_TRUE}, σ²={VAR_TRUE})')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=5,
                    label=f'loss min: ℓ={math.exp(ll_min):.2f}, σ²={math.exp(lv_min):.2f}')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        ax.set_title(f'GP-drifted data: where do methods converge at varied n_em / lr?\n'
                      f'(red ✕ truth, gold ★ landscape min, black + init)')
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig('/tmp/T11_gp_drift_long.png', dpi=150)
        print('[T11] Saved /tmp/T11_gp_drift_long.png', flush=True)
    except Exception as e:
        print(f'[T11] Plot rendering failed: {e}', flush=True)


if __name__ == '__main__':
    main()
