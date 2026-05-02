"""
Convergence benchmark: EFGP-SING (JAX, pinned grid, learn_kernel=True) vs.
SING-SparseGP (with full M-step) at fixed T.  Tracks per-EM-iter
predictive observation RMSE and final kernel hypers.

Lets you see (a) whether the optimization plateaued, and (b) where the
two methods land in (ℓ, σ²) space.

Run as its own python process (no torch import):

    python demos/bench_convergence.py
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

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params


D = 2
sigma = 0.4
A_true = jnp.array([[-1.5, 0.0], [0.0, -1.5]])

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


def run_one(T: int, n_em: int = 25, n_estep: int = 10,
            ls_init: float = 0.7, var_init: float = 1.0):
    print(f"\n{'='*70}\n  T = {T}, n_em = {n_em}\n{'='*70}")
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
    rho_sched = jnp.linspace(0.1, 0.7, n_em)

    # ============================================================
    # EFGP-SING (JAX, pinned grid, learn_kernel=True)
    # ============================================================
    t0 = time.time()
    mp_efgp, _, op_efgp, _, hist_efgp = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=var_init, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True,
        n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=2,
        pin_grid=True,
        true_xs=np.asarray(xs), verbose=False,
    )
    t_efgp = time.time() - t0
    rmse_efgp_final = predict_obs_rmse(mp_efgp['m'][0], op_efgp, ys)

    # ============================================================
    # SING-SparseGP (full M-step)
    # ============================================================
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sparse_drift_params = dict(
        length_scales=jnp.full((D,), float(ls_init)),
        output_scale=jnp.asarray(float(var_init) ** 0.5),
    )
    t0 = time.time()
    mp_sp, _, _, sp_drift_params, _, op_sp, _, _ = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=op, sigma=sigma,
        rho_sched=rho_sched,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((n_em,), 0.05),
        print_interval=999,
    )
    t_sp = time.time() - t0
    rmse_sp_final = predict_obs_rmse(mp_sp['m'][0], op_sp, ys)
    sp_ls_final = float(jnp.mean(sp_drift_params['length_scales']))
    sp_var_final = float(sp_drift_params['output_scale']) ** 2

    trivial = float(np.sqrt(np.mean((ys - ys.mean(0)) ** 2)))

    print(f"\n  EFGP-SING (JAX pinned, learn_kernel):")
    print(f"    wall    {t_efgp:.1f}s")
    print(f"    final RMSE {rmse_efgp_final:.4f}")
    print(f"    final ℓ {hist_efgp.lengthscale[-1]:.3f}, "
          f"σ² {hist_efgp.variance[-1]:.3f}")
    print(f"  SING-SparseGP (full M-step):")
    print(f"    wall    {t_sp:.1f}s")
    print(f"    final RMSE {rmse_sp_final:.4f}")
    print(f"    final ℓ {sp_ls_final:.3f}, σ² {sp_var_final:.3f}")
    print(f"  trivial RMSE {trivial:.4f}")

    return dict(
        T=T, n_em=n_em,
        t_efgp=t_efgp, t_sp=t_sp,
        rmse_efgp=rmse_efgp_final, rmse_sp=rmse_sp_final, trivial=trivial,
        ls_efgp=hist_efgp.lengthscale, var_efgp=hist_efgp.variance,
        ls_sp_final=sp_ls_final, var_sp_final=sp_var_final,
        latent_rmse_efgp=hist_efgp.latent_rmse,
    )


def main():
    rows = []
    for T in [60, 240]:
        rows.append(run_one(T, n_em=25, n_estep=10))

    print("\n" + "=" * 80)
    for r in rows:
        print(f"\nT={r['T']}")
        print(f"  EFGP final ℓ trajectory: {[f'{x:.3f}' for x in r['ls_efgp']]}")
        print(f"  EFGP final σ² trajectory:{[f'{x:.3f}' for x in r['var_efgp']]}")
        print(f"  SparseGP final ℓ:        {r['ls_sp_final']:.3f}  σ²: {r['var_sp_final']:.3f}")
        print(f"  EFGP RMSE:    {r['rmse_efgp']:.4f}    SparseGP RMSE:  {r['rmse_sp']:.4f}    trivial: {r['trivial']:.4f}")
        print(f"  EFGP wall:    {r['t_efgp']:.1f}s         SparseGP wall:  {r['t_sp']:.1f}s")

    # Save the EFGP latent_rmse curve
    fig, ax = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4))
    if len(rows) == 1:
        ax = [ax]
    for i, r in enumerate(rows):
        ax[i].plot(r['latent_rmse_efgp'], "o-", label="EFGP latent RMSE (raw)")
        ax[i].set_xlabel("EM iter"); ax[i].set_ylabel("latent RMSE")
        ax[i].set_title(f"T = {r['T']}")
        ax[i].legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "_efgp_demo_out_jax" / "convergence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved convergence plot to {out}")


if __name__ == "__main__":
    main()
