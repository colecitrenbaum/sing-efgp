"""
T8: Capture full (ℓ, σ²) trajectories during EM for both EFGP and SparseGP
on Duffing. Overlay them on the marginal-likelihood landscape from T7.

Goal: visualize *where* on the landscape each method walks, and *whether*
the trajectories follow the ridge or cross it. If both end inside the
L-1 contour but at different (ℓ, σ²), they're tracking the same ridge.

Output:
  /tmp/T8_duffing_trajectories.npz   (efgp_traj, sparse_traj_lr1e-3, sparse_traj_lr2e-3)
  /tmp/T8_duffing_landscape_with_trajectories.png

Run:
  ~/myenv/bin/python demos/diag_trajectories_duffing.py
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
SIGMA = 0.2          # Match notebook EXPS[1] Duffing config
N_OBS = 8
SEED = 13            # Match notebook EXPS[1] seed
T_MAX = 15.0
N_EM = 20            # Match notebook
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


def fit_efgp_with_traj(lik, op, ip, t_grid):
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
    # hist.lengthscale and .variance are lists, length n_em
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def fit_sparsegp_with_traj(lik, op, ip, t_grid, lr=1e-3):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []   # populated by fit_variational_em via drift_params_history hook
    fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0,
        init_params=ip, output_params=op, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((N_EM,), lr),
        print_interval=999, drift_params_history=history,
    )
    # history is a list of dicts {length_scales, output_scale, ...}
    ls_traj = np.array([LS_INIT] + [float(np.mean(np.asarray(d['length_scales'])))
                                      for d in history])
    var_traj = np.array([VAR_INIT] + [float(np.asarray(d['output_scale'])) ** 2
                                       for d in history])
    return ls_traj, var_traj


def main():
    print(f"[T8] JAX devices: {jax.devices()}", flush=True)
    print(f"[T8] Duffing T={T}, sigma={SIGMA}, n_em={N_EM}, "
          f"init=(ℓ={LS_INIT}, σ²={VAR_INIT})", flush=True)
    xs, ys, lik, op, ip, t_grid = setup()

    print("\n[T8] Running EFGP...", flush=True)
    efgp_ls, efgp_var = fit_efgp_with_traj(lik, op, ip, t_grid)
    print(f"  EFGP final: ℓ={efgp_ls[-1]:.3f}, σ²={efgp_var[-1]:.3f}",
          flush=True)

    print("\n[T8] Running SparseGP @ lr=1e-3...", flush=True)
    sp1_ls, sp1_var = fit_sparsegp_with_traj(lik, op, ip, t_grid, lr=1e-3)
    print(f"  Sparse@1e-3 final: ℓ={sp1_ls[-1]:.3f}, σ²={sp1_var[-1]:.3f}",
          flush=True)

    print("\n[T8] Running SparseGP @ lr=2e-3...", flush=True)
    try:
        sp2_ls, sp2_var = fit_sparsegp_with_traj(lik, op, ip, t_grid, lr=2e-3)
        print(f"  Sparse@2e-3 final: ℓ={sp2_ls[-1]:.3f}, σ²={sp2_var[-1]:.3f}",
              flush=True)
    except Exception as e:
        print(f"  Sparse@2e-3 NaN'd: {e}", flush=True)
        sp2_ls = sp2_var = None

    # Save
    save_dict = dict(
        efgp_ls=efgp_ls, efgp_var=efgp_var,
        sp1_ls=sp1_ls, sp1_var=sp1_var,
    )
    if sp2_ls is not None:
        save_dict['sp2_ls'] = sp2_ls
        save_dict['sp2_var'] = sp2_var
    np.savez('/tmp/T8_duffing_trajectories.npz', **save_dict)
    print("\n[T8] Saved /tmp/T8_duffing_trajectories.npz", flush=True)

    # Overlay on landscape if available
    landscape_npz = '/tmp/T7_duffing_landscape.npz'
    try:
        data = np.load(landscape_npz, allow_pickle=True)
        L_grid = data['L_grid']
        LOG_LS = data['log_ls_grid']; LOG_VAR = data['log_var_grid']
        L_min = float(data['L_min'])
        ll_min = float(data['ll_min']); lv_min = float(data['lv_min'])

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

        # constant-α dotted lines
        for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ax.plot(LOG_LS, LOG_LS * 2 + log_alpha_show,
                    color='white', linestyle=':', linewidth=0.7, alpha=0.5)

        # Trajectories
        ax.plot(np.log(efgp_ls), np.log(efgp_var), '-o', color='C0',
                markersize=4, linewidth=1.5, label='EFGP', zorder=4)
        ax.plot(np.log(sp1_ls), np.log(sp1_var), '-s', color='C1',
                markersize=4, linewidth=1.5, label='SparseGP @ lr=1e-3',
                zorder=4)
        if sp2_ls is not None:
            ax.plot(np.log(sp2_ls), np.log(sp2_var), '-^', color='C3',
                    markersize=4, linewidth=1.5, label='SparseGP @ lr=2e-3',
                    zorder=4)

        # Mark start and finish
        ax.scatter([np.log(efgp_ls[0])], [np.log(efgp_var[0])], marker='+',
                   s=200, color='black', zorder=6, label='init (ℓ=1, σ²=1)')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                   color='gold', edgecolor='k', zorder=5,
                   label=f'loss min: ℓ={math.exp(ll_min):.2f}, σ²={math.exp(lv_min):.2f}')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_title('Duffing: full M-step trajectories on the loss landscape\n'
                      '(white dotted: constant α=σ²/ℓ²)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(LOG_LS.min(), LOG_LS.max())
        ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
        plt.tight_layout()
        plt.savefig('/tmp/T8_duffing_landscape_with_trajectories.png', dpi=150)
        print('[T8] Saved /tmp/T8_duffing_landscape_with_trajectories.png',
              flush=True)
    except FileNotFoundError:
        print(f'[T8] {landscape_npz} not found; run diag_marginal_likelihood_landscape.py first',
              flush=True)


if __name__ == '__main__':
    main()
