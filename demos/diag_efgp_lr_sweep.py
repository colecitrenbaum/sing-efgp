"""
T14: Does EFGP recover truth at lower mstep_lr?

Hypothesis: EFGP's mstep_lr=0.05 (50× SparseGP's lr=1e-3) is too
aggressive — overshoots and oscillates. SparseGP at lr=1e-3 walks
monotonically toward truth on the GP-drift problem. If EFGP at lower
mstep_lr does the same, the divergence is just optimizer-scale.

Test: same GP-drift setup as T11. EFGP at mstep_lr ∈ {0.05, 0.01, 0.003,
0.001} for n_em=100. Compare individual ℓ, σ² trajectories.

Run:
  ~/myenv/bin/python demos/diag_efgp_lr_sweep.py
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
N_EM = 100
MSTEP_LR_LIST = [0.05, 0.01, 0.003, 0.001]


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


def fit_efgp(lik, op, ip, t_grid, mstep_lr):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def main():
    print(f"[T14] EFGP mstep_lr sweep on GP-drift problem", flush=True)
    print(f"[T14] truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}", flush=True)
    print(f"[T14] init:  ℓ={LS_INIT}, σ²={VAR_INIT}", flush=True)
    xs, ys, lik, op, ip, t_grid = setup()

    results = {}
    for mstep_lr in MSTEP_LR_LIST:
        print(f"\n[T14] EFGP mstep_lr={mstep_lr}, n_em={N_EM}...", flush=True)
        ls_t, var_t = fit_efgp(lik, op, ip, t_grid, mstep_lr)
        results[mstep_lr] = (ls_t, var_t)
        print(f"  Final: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}, "
              f"α={var_t[-1]/ls_t[-1]**2:.3f}", flush=True)
        # Sample iter checkpoints
        for chk in [10, 25, 50, 75, 100]:
            if chk < len(ls_t):
                print(f"    iter {chk}: ℓ={ls_t[chk]:.3f}, σ²={var_t[chk]:.3f}",
                      flush=True)

    np.savez('/tmp/T14_efgp_lr_sweep.npz',
              **{f'mstep_lr_{lr}_ls': v[0] for lr, v in results.items()},
              **{f'mstep_lr_{lr}_var': v[1] for lr, v in results.items()})
    print(f"\n[T14] Saved /tmp/T14_efgp_lr_sweep.npz", flush=True)

    # Plot ℓ and σ² trajectories
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['C0', 'C2', 'C4', 'C6']
        for ax, (var_label, ls_or_var, truth) in zip(
                axes, [('ℓ', 0, LS_TRUE), ('σ²', 1, VAR_TRUE)]):
            for color, lr in zip(colors, MSTEP_LR_LIST):
                trajs = results[lr]
                ax.plot(np.arange(len(trajs[ls_or_var])), trajs[ls_or_var],
                        '-', color=color, linewidth=1.5,
                        label=f'mstep_lr={lr}')
            ax.axhline(truth, color='red', linestyle='--', linewidth=1.5,
                        label=f'truth = {truth}')
            ax.axhline(LS_INIT if var_label == 'ℓ' else VAR_INIT,
                        color='black', linestyle=':', linewidth=1,
                        label=f'init')
            ax.set_xlabel('EM iteration')
            ax.set_ylabel(var_label)
            ax.set_title(f'EFGP recovery of {var_label} on GP-drift problem')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('/tmp/T14_efgp_lr_sweep.png', dpi=150)
        print(f"[T14] Saved /tmp/T14_efgp_lr_sweep.png", flush=True)
    except Exception as e:
        print(f"[T14] Plot failed: {e}", flush=True)


if __name__ == '__main__':
    main()
