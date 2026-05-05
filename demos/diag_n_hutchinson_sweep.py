"""
T6: Demonstrate the fix — sweep n_hutchinson_mstep on the linear row.

Following T1c-seeds finding (Hutchinson is unbiased but high-variance),
this script verifies the recommended fix: bump ``n_hutchinson_mstep``
from production default 4 → 32+, observe whether the linear-row ℓ
trajectory becomes (a) less variable run-to-run, and (b) reaches a
similar (or higher) value with fewer EM iters.

Two experiments:
  Exp A: 5 seeds × {4, 16, 32, 64} probes — measure run-to-run spread
  Exp B: 1 seed × {4, 16, 32, 64} probes at n_em=50 — check final
         ℓ converges to a stable value.

Run:
  ~/myenv/bin/python demos/diag_n_hutchinson_sweep.py
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


D = 2
T = 300
SIGMA = 0.3
N_OBS = 8
N_EM = 30   # enough to see M-step trajectory; smaller than n_em=100 user uses
N_HUTCHINSON_LIST = [4, 16, 32, 64]
SEED_LIST_A = [7, 13, 23, 31, 47]   # 5 seeds for spread
SEED_FOR_B = 7


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def setup(seed):
    A = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
    drift = lambda x, t: A @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(seed), x0=jnp.array([2.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(seed)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(seed + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=vt[:D].T, d=ys.mean(0),
                   R=jnp.full((N_OBS,), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op_init, ip_init, t_grid


def fit(n_hutchinson, lik, op, ip, t_grid, seed):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    mp, _, op_efgp, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.7, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=n_hutchinson, kernel_warmup_iters=8,
        seed=seed, verbose=False,
    )
    return np.array(hist.lengthscale), np.array(hist.variance)


def main():
    print(f"[T6] JAX devices: {jax.devices()}", flush=True)
    print(f"[T6] linear, T={T}, n_em={N_EM}", flush=True)
    print(f"[T6] sweeping n_hutchinson_mstep ∈ {N_HUTCHINSON_LIST}", flush=True)

    # ---- Exp A: spread across seeds ----
    print(f"\n[T6] Exp A: 5 seeds × n_hutch sweep — measure run-to-run "
          f"spread of final ℓ", flush=True)
    print(f"  {'n_hutch':>7s}  {'final ℓ across seeds':>32s}  "
          f"{'mean ℓ':>7s}  {'std ℓ':>7s}", flush=True)
    for n_h in N_HUTCHINSON_LIST:
        finals = []
        for seed in SEED_LIST_A:
            xs, ys, lik, op, ip, t_grid = setup(seed)
            ls_traj, _ = fit(n_h, lik, op, ip, t_grid, seed)
            finals.append(ls_traj[-1])
        finals = np.array(finals)
        print(f"  {n_h:>7d}  "
              f"{[f'{v:.3f}' for v in finals]}  "
              f"{finals.mean():>7.4f}  {finals.std():>7.4f}", flush=True)

    # ---- Exp B: single seed, ℓ trajectory ----
    print(f"\n[T6] Exp B: seed={SEED_FOR_B} ℓ trajectory across iters", flush=True)
    trajectories = {}
    for n_h in N_HUTCHINSON_LIST:
        xs, ys, lik, op, ip, t_grid = setup(SEED_FOR_B)
        ls_traj, var_traj = fit(n_h, lik, op, ip, t_grid, SEED_FOR_B)
        trajectories[n_h] = (ls_traj, var_traj)

    # Print final values + trajectory
    print(f"\n[T6] Final values (seed {SEED_FOR_B}, n_em={N_EM}):", flush=True)
    print(f"  {'n_hutch':>7s}  {'final ℓ':>9s}  {'final σ²':>9s}  "
          f"{'α=σ²/ℓ²':>9s}", flush=True)
    for n_h in N_HUTCHINSON_LIST:
        ls_t, var_t = trajectories[n_h]
        ls_f = ls_t[-1]; var_f = var_t[-1]
        a = var_f / (ls_f ** 2)
        print(f"  {n_h:>7d}  {ls_f:>9.4f}  {var_f:>9.4f}  {a:>9.4f}",
              flush=True)

    # Brief trajectory at every 5 iters
    print(f"\n[T6] ℓ trajectory snippet (every 5 EM iters):", flush=True)
    iter_idxs = list(range(4, N_EM, 5))   # iter 5, 10, 15, 20, 25, 30
    print(f"  {'n_hutch':>7s}  " + "  ".join(f"{'iter ' + str(i+1):>9s}"
                                                  for i in iter_idxs),
          flush=True)
    for n_h in N_HUTCHINSON_LIST:
        ls_t, _ = trajectories[n_h]
        cells = [f"{ls_t[i]:>9.3f}" for i in iter_idxs]
        print(f"  {n_h:>7d}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
