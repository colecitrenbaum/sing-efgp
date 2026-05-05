"""
Convergence verification: run both methods at varying ``n_em`` on the
Duffing T=500 problem and check whether final RMSE & hypers stabilize.

If results don't change between e.g. n_em=20 and n_em=40, both methods
have converged (independent of learning-rate choices).  If they do change,
we report by how much so the comparison can be controlled for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import time
import numpy as np

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

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
sigma = 0.3
ALPHA, BETA, GAMMA = 1.0, 1.0, 0.5
T = 500


def drift_duffing(x, t):
    return jnp.array([x[1], ALPHA * x[0] - BETA * x[0] ** 3 - GAMMA * x[1]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def predict_obs_rmse(m_jax, output_params, ys):
    m = np.asarray(m_jax)
    C = np.asarray(output_params['C']); d = np.asarray(output_params['d'])
    y_hat = m @ C.T + d
    return float(np.sqrt(np.mean((y_hat - np.asarray(ys)) ** 2)))


def proc_lat_rmse(m_inferred, m_true):
    Xi = np.asarray(m_inferred); Xt = np.asarray(m_true)
    bi = Xi.mean(0); bt = Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    return float(np.sqrt(np.mean(((Xi - bi) @ A_T + bt - Xt) ** 2)))


def setup():
    t_max = T * 0.05
    sigma_fn = lambda x, t: sigma * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.5, 0.0]),
                       f=drift_duffing, t_max=t_max, n_timesteps=T,
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
    t_grid = jnp.linspace(0., t_max, T)
    return xs, ys, lik, op, ip, t_grid


def fit_efgp(n_em, lik, op, ip, t_grid, xs):
    rho_sched = jnp.linspace(0.1, 0.7, n_em)
    t0 = time.time()
    mp_efgp, _, op_efgp, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.7, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=2,
        pin_grid=True, pin_grid_lengthscale=0.5, verbose=False,
    )
    return dict(
        wall=time.time() - t0,
        rmse_obs=predict_obs_rmse(mp_efgp['m'][0], op_efgp, lik.ys_obs[0]),
        lat_rmse=proc_lat_rmse(mp_efgp['m'][0], xs),
        ls=hist.lengthscale[-1], var=hist.variance[-1],
    )


def fit_sparsegp(n_em, lik, op, ip, t_grid, xs):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.full((D,), 0.7),
                                 output_scale=jnp.asarray(1.0))
    rho_sched = jnp.linspace(0.1, 0.7, n_em)
    t0 = time.time()
    mp_sp, _, _, sp_drift_params, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=sigma,
        rho_sched=rho_sched,
        n_iters=n_em, n_iters_e=10, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((n_em,), 0.05),
        print_interval=999,
    )
    return dict(
        wall=time.time() - t0,
        rmse_obs=predict_obs_rmse(mp_sp['m'][0], op_sp, lik.ys_obs[0]),
        lat_rmse=proc_lat_rmse(mp_sp['m'][0], xs),
        ls=float(jnp.mean(sp_drift_params['length_scales'])),
        var=float(sp_drift_params['output_scale']) ** 2,
    )


def main():
    xs, ys, lik, op, ip, t_grid = setup()
    print(f"\nDuffing T={T}, sigma={sigma}")
    print(f"trivial latent RMSE: {float(np.sqrt(np.mean(np.asarray(xs)**2))):.4f}")

    grid_em = [10, 20, 40]

    print("\n" + "=" * 96)
    print(f"{'n_em':>5}   {'EFGP':>40s}   {'SparseGP':>40s}")
    print(f"{'':>5}   {'wall  obs_rmse  lat_rmse  ℓ      σ²':>40}"
          f"   {'wall  obs_rmse  lat_rmse  ℓ      σ²':>40}")
    print("-" * 96)
    for nem in grid_em:
        e = fit_efgp(nem, lik, op, ip, t_grid, xs)
        s = fit_sparsegp(nem, lik, op, ip, t_grid, xs)
        print(f"{nem:>5}   {e['wall']:5.1f} {e['rmse_obs']:.4f}  "
              f"{e['lat_rmse']:.4f}  {e['ls']:.3f}  {e['var']:.3f}   "
              f"{s['wall']:5.1f} {s['rmse_obs']:.4f}  "
              f"{s['lat_rmse']:.4f}  {s['ls']:.3f}  {s['var']:.3f}")


if __name__ == "__main__":
    main()
