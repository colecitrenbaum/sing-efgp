"""
T2a: Long-horizon alignment between EFGP and SparseGP.

The user's monotone-ℓ-climb on the linear row at n_em=25/50/100 is
believed to be a budget effect, not a fixed-point bias — the user's
shared-smoother diagnostic confirmed both methods move α in the same
direction, and lr=2e-3 SparseGP closes part of the gap. Question: at
large n_em with the same conservative schedule, do the two methods
land at the same (ℓ, σ², α)?

Run both methods on the linear row at n_em ∈ {50, 100, 200} with the
same ρ schedule. Look for monotone trend toward shared asymptote.

We also report on the Duffing row to see if the alignment story
generalizes beyond linear.

Run:
  ~/myenv/bin/python demos/diag_alignment_long_horizon.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# IMPORTANT: import jax_finufft (via efgp_jax_primitives) BEFORE any
# torch-using SING module. fit_efgp_sing_jax is pure JAX; SparseGP path
# imports torch. We segregate by running them in two subprocess-style
# blocks below — but here a single process runs both, so import EFGP
# primitives first.
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
SIGMA = 0.3
N_OBS = 8
SEED = 7

T_LINEAR = 300
T_DUFFING = 300
N_EM_LIST = [25, 50, 100]


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def setup(regime, T):
    if regime == "linear":
        A = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
        drift = lambda x, t: A @ x
        x0 = jnp.array([2.0, 0.0])
        t_max = 8.0
    elif regime == "duffing":
        drift = lambda x, t: jnp.array(
            [x[1], 1.0 * x[0] - 1.0 * x[0]**3 - 0.5 * x[1]])
        x0 = jnp.array([1.5, 0.0])
        t_max = T * 0.05
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(SEED), x0=x0, f=drift,
                      t_max=t_max, n_timesteps=T, sigma=sigma_fn)
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


def predict_obs_rmse(m_jax, output_params, ys):
    m = np.asarray(m_jax)
    C = np.asarray(output_params['C']); d = np.asarray(output_params['d'])
    return float(np.sqrt(np.mean((m @ C.T + d - np.asarray(ys)) ** 2)))


def proc_lat_rmse(m_inferred, m_true):
    Xi = np.asarray(m_inferred); Xt = np.asarray(m_true)
    bi = Xi.mean(0); bt = Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    return float(np.sqrt(np.mean(((Xi - bi) @ A_T + bt - Xt) ** 2)))


def fit_efgp(n_em, lik, op, ip, t_grid, xs):
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.time()
    mp, _, op_efgp, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.7, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False,
    )
    return dict(
        wall=time.time() - t0,
        rmse_obs=predict_obs_rmse(mp['m'][0], op_efgp, lik.ys_obs[0]),
        lat_rmse=proc_lat_rmse(mp['m'][0], xs),
        ls=hist.lengthscale[-1], var=hist.variance[-1],
    )


def fit_sparsegp(n_em, lik, op, ip, t_grid, xs, lr=1e-3):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(length_scales=jnp.full((D,), 0.7),
                                output_scale=jnp.asarray(1.0))
    rho_sched = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.time()
    mp, _, _, sp_dp, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=n_em, n_iters_e=10, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((n_em,), lr),
        print_interval=999,
    )
    return dict(
        wall=time.time() - t0,
        rmse_obs=predict_obs_rmse(mp['m'][0], op_sp, lik.ys_obs[0]),
        lat_rmse=proc_lat_rmse(mp['m'][0], xs),
        ls=float(jnp.mean(sp_dp['length_scales'])),
        var=float(sp_dp['output_scale']) ** 2,
    )


def alpha(ell, var):
    return float(var) / (float(ell) ** 2)


def run_regime(regime, T):
    print(f"\n[T2a] === regime: {regime}  T={T} ===", flush=True)
    xs, ys, lik, op, ip, t_grid = setup(regime, T)
    rows = []
    for n_em in N_EM_LIST:
        e = fit_efgp(n_em, lik, op, ip, t_grid, xs)
        # SparseGP: lr=1e-3 (notebook) and lr=2e-3 (user-confirmed stable)
        s_default = fit_sparsegp(n_em, lik, op, ip, t_grid, xs, lr=1e-3)
        try:
            s_active = fit_sparsegp(n_em, lik, op, ip, t_grid, xs, lr=2e-3)
        except Exception as ex:
            print(f"  [T2a] sparse lr=2e-3 NaN'd: {ex}", flush=True)
            s_active = None
        print(f"  n_em={n_em:>3}  EFGP    ℓ={e['ls']:.3f} σ²={e['var']:.3f} "
              f"α={alpha(e['ls'], e['var']):.3f}  lat={e['lat_rmse']:.4f} "
              f"obs={e['rmse_obs']:.4f}  ({e['wall']:.0f}s)", flush=True)
        print(f"             SP1e-3   ℓ={s_default['ls']:.3f} σ²={s_default['var']:.3f} "
              f"α={alpha(s_default['ls'], s_default['var']):.3f}  "
              f"lat={s_default['lat_rmse']:.4f} obs={s_default['rmse_obs']:.4f}  "
              f"({s_default['wall']:.0f}s)", flush=True)
        if s_active is not None:
            print(f"             SP2e-3   ℓ={s_active['ls']:.3f} σ²={s_active['var']:.3f} "
                  f"α={alpha(s_active['ls'], s_active['var']):.3f}  "
                  f"lat={s_active['lat_rmse']:.4f} obs={s_active['rmse_obs']:.4f}  "
                  f"({s_active['wall']:.0f}s)", flush=True)
        rows.append(dict(n_em=n_em, efgp=e, sp_default=s_default,
                          sp_active=s_active))
    return rows


def main():
    print(f"[T2a] JAX devices: {jax.devices()}", flush=True)
    out = {}
    out['linear'] = run_regime('linear', T_LINEAR)
    out['duffing'] = run_regime('duffing', T_DUFFING)

    print("\n[T2a] Asymptotic (n_em=200) summary:", flush=True)
    print(f"  {'regime':10s}  {'method':10s}  {'ℓ':>6s}  {'σ²':>6s}  {'α':>6s}  "
          f"{'lat RMSE':>9s}", flush=True)
    for regime, rows in out.items():
        last = rows[-1]   # n_em=200
        e = last['efgp']; sd = last['sp_default']; sa = last['sp_active']
        print(f"  {regime:10s}  EFGP        {e['ls']:>6.3f}  {e['var']:>6.3f}  "
              f"{alpha(e['ls'], e['var']):>6.3f}  {e['lat_rmse']:>9.4f}",
              flush=True)
        print(f"  {regime:10s}  Sparse 1e-3  {sd['ls']:>6.3f}  {sd['var']:>6.3f}  "
              f"{alpha(sd['ls'], sd['var']):>6.3f}  {sd['lat_rmse']:>9.4f}",
              flush=True)
        if sa is not None:
            print(f"  {regime:10s}  Sparse 2e-3  {sa['ls']:>6.3f}  {sa['var']:>6.3f}  "
                  f"{alpha(sa['ls'], sa['var']):>6.3f}  {sa['lat_rmse']:>9.4f}",
                  flush=True)

    print("\n[T2a] Convergence trend on linear row (ℓ vs n_em):", flush=True)
    for n_em_idx, n_em in enumerate(N_EM_LIST):
        e = out['linear'][n_em_idx]['efgp']
        sd = out['linear'][n_em_idx]['sp_default']
        sa = out['linear'][n_em_idx]['sp_active']
        sa_ls = f"{sa['ls']:.3f}" if sa is not None else "NaN"
        print(f"  n_em={n_em:>3}: EFGP ℓ={e['ls']:.3f}  "
              f"Sparse(1e-3) ℓ={sd['ls']:.3f}  Sparse(2e-3) ℓ={sa_ls}",
              flush=True)


if __name__ == "__main__":
    main()
