"""
Wall-time benchmark: EFGP-SING (JAX, pinned grid) vs. SING-SparseGP across
several T values.  All hypers held fixed (no M-step) so per-iter cost is
the apples-to-apples comparison.

Both methods scale linearly in T per iter, but with different constants:

    EFGP per iter:        O(T D + M log M)              with M = 49
    SparseGP per iter:    O(T D M_inducing²)            with M_inducing² = 4096

So the per-iter ratio (EFGP / SparseGP) should DECREASE as T grows.

Run as its own python process (no torch import):

    python demos/bench_T_jax_vs_sparsegp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import numpy as np

# JAX-finufft loaded first
import sing.efgp_em as em
import sing.efgp_jax_primitives as jp

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
sigma = 0.4
A_true = jnp.array([[-1.5, 0.0], [0.0, -1.5]])

class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def run_one(T: int, n_em: int = 6, n_estep: int = 8):
    print(f"\n{'='*60}\n  T = {T}\n{'='*60}")
    drift_fn = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.2, -0.8]),
                      f=drift_fn, t_max=2.0, n_timesteps=T,
                      sigma=sigma_fn)

    N = 8
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                 dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., 2.0, T)

    # --- EFGP-SING (JAX, pinned grid) ---
    t0 = time.time()
    mp_efgp, _, op_efgp, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.7, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=jnp.linspace(0.2, 0.9, n_em),
        learn_emissions=True, update_R=False,
        learn_kernel=False, pin_grid=True,
        verbose=False,
    )
    t_efgp = time.time() - t0

    m = np.asarray(mp_efgp['m'][0])
    y_hat = m @ np.asarray(op_efgp['C']).T + np.asarray(op_efgp['d'])
    rmse_efgp = float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))

    # --- SparseGP ---
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(
        length_scales=jnp.full((D,), 0.7),
        output_scale=jnp.asarray(1.0),
    )
    t0 = time.time()
    mp_sp, _, _, _, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=sigma,
        rho_sched=jnp.linspace(0.2, 0.9, n_em),
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((n_em,), 0.05),
        print_interval=999,
    )
    t_sp = time.time() - t0
    m = np.asarray(mp_sp['m'][0])
    y_hat = m @ np.asarray(op_sp['C']).T + np.asarray(op_sp['d'])
    rmse_sp = float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))

    trivial = float(np.sqrt(np.mean((ys - ys.mean(0)) ** 2)))
    print(f"  EFGP-SING (JAX pinned):  {t_efgp:6.2f}s   obs RMSE {rmse_efgp:.4f}")
    print(f"  SING-SparseGP:           {t_sp:6.2f}s   obs RMSE {rmse_sp:.4f}")
    print(f"  trivial baseline:                       obs RMSE {trivial:.4f}")
    print(f"  EFGP/SparseGP wall ratio: {t_efgp/t_sp:.2f}x")
    return dict(T=T, t_efgp=t_efgp, t_sp=t_sp,
                rmse_efgp=rmse_efgp, rmse_sp=rmse_sp, trivial=trivial)


def main():
    rows = []
    for T in [60, 120, 240, 500]:
        rows.append(run_one(T))

    print("\n" + "=" * 80)
    print(f"{'T':>5} {'EFGP s':>10} {'SparseGP s':>12} {'ratio':>8} "
          f"{'EFGP rmse':>10} {'SparseGP rmse':>14} {'trivial':>10}")
    print("-" * 80)
    for r in rows:
        ratio = r['t_efgp'] / r['t_sp']
        print(f"{r['T']:>5} {r['t_efgp']:>10.2f} {r['t_sp']:>12.2f} "
              f"{ratio:>7.2f}x {r['rmse_efgp']:>10.4f} "
              f"{r['rmse_sp']:>14.4f} {r['trivial']:>10.4f}")


if __name__ == "__main__":
    main()
