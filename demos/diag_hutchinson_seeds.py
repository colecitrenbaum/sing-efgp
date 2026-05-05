"""
T1c-debug-seeds: Is the Hutchinson bias real or just an unlucky seed?

The previous T1c found 4 positive estimates while truth is negative —
suggestive of bias. But possibly just unlucky with PRNGKey(0). This
script runs Hutchinson at n=1024 with 32 independent seeds and reports
the mean and stderr. If the seed-mean is far from exact (more than
~3 stderrs), it's a real bias.

Run:
  ~/myenv/bin/python demos/diag_hutchinson_seeds.py
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

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Gaussian
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
T = 200
SIGMA = 0.3
LS = 1.0
VAR = 1.0
SEED = 7
K_FD = 8
N_PER_BATCH = 1024
N_BATCHES = 32
THETA_GRID = [
    (0.0, 0.0),
    (-0.3, 0.2),
    (0.3, -0.2),   # the suspect
]


def setup():
    A_true = jnp.diag(jnp.array([-1.0, -1.5]))
    drift_true = lambda x, t: A_true @ x
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 2.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([1.0, -0.6]),
                      f=drift_true, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    N = 6
    C_true = rng.standard_normal((N, D)) * 0.6
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, t_max


def build_state():
    xs, ys, lik, t_max = setup()
    t_grid = jnp.linspace(0., t_max, T)
    del_t = t_grid[1:] - t_grid[:-1]
    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=Vt[:D].T, d=ys.mean(0),
                   R=jnp.full((ys.shape[1],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(LS)),
                 output_scale=jnp.asarray(math.sqrt(VAR)))
    rho_sched = jnp.linspace(0.05, 0.4, 10)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=ip_init, output_params=op_init, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=10, n_iters_e=6, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((10,), 0.05), print_interval=999,
    )
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    _, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M,
    )
    return grid, top


def hutch_dense_one_batch(A, dA_mat, key, n_probes):
    M = int(A.shape[0])
    cdtype = A.dtype
    keys = jr.split(key, n_probes)
    def one_probe(key_p):
        v_real = (jax.random.bernoulli(key_p, shape=(M,))
                  .astype(jnp.float32) * 2.0 - 1.0)
        v = v_real.astype(cdtype)
        b = dA_mat @ v
        u = jnp.linalg.solve(A, b)
        return jnp.vdot(v, u).real
    return float(jnp.mean(jax.vmap(one_probe)(keys)))


def main():
    print(f"[seeds] JAX devices: {jax.devices()}", flush=True)
    grid, top = build_state()
    M = int(grid.M)
    print(f"[seeds] K={K_FD}, M={M}", flush=True)

    cdtype = top.v_fft.dtype
    h_scalar = grid.h_per_dim[0]

    for log_ls, log_var in THETA_GRID:
        print(f"\n[seeds] === θ=(log_ls={log_ls}, log_var={log_var}) ===",
              flush=True)
        ll = jnp.asarray(log_ls, dtype=jnp.float32)
        lv = jnp.asarray(log_var, dtype=jnp.float32)
        ws_real = _ws_real_se(ll, lv, grid.xis_flat, h_scalar, D)
        ws_c = ws_real.astype(cdtype)

        ls = math.exp(log_ls)
        ls_sq = ls * ls
        xi_norm_sq = (grid.xis_flat * grid.xis_flat).sum(axis=1)
        dws = ws_real * 0.5 * (D - 4 * math.pi * math.pi * ls_sq * xi_norm_sq)
        dws_c = dws.astype(cdtype)

        # Build A and dA dense
        eye_c = jnp.eye(M, dtype=cdtype)
        T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
        T_mat = T_cols.T
        D_w = ws_c[:, None] * jnp.eye(M, dtype=cdtype)
        dD_w = dws_c[:, None] * jnp.eye(M, dtype=cdtype)
        A = eye_c + D_w @ T_mat @ D_w
        dA_mat = dD_w @ T_mat @ D_w + D_w @ T_mat @ dD_w

        # Exact reference via slogdet autodiff
        def logdet_term(ll_):
            ws_real_ = _ws_real_se(ll_, lv, grid.xis_flat, h_scalar, D)
            ws_c_ = ws_real_.astype(cdtype)
            D_w_ = ws_c_[:, None] * jnp.eye(M, dtype=cdtype)
            A_ = eye_c + D_w_ @ T_mat @ D_w_
            sign, logdet_ = jnp.linalg.slogdet(A_)
            return logdet_.real
        exact = float(jax.grad(logdet_term)(ll))
        print(f"  exact tr[A^-1 ∂A] = {exact:.4f}", flush=True)

        # Hutchinson with N_BATCHES independent seed batches
        batch_results = []
        for b in range(N_BATCHES):
            key = jr.PRNGKey(1000 + b)
            tr_h = hutch_dense_one_batch(A, dA_mat, key, N_PER_BATCH)
            batch_results.append(tr_h)
        batch_results = np.array(batch_results)
        mean_h = batch_results.mean()
        sem_h = batch_results.std(ddof=1) / np.sqrt(N_BATCHES)
        # Total = N_BATCHES * N_PER_BATCH probes pooled
        z = (mean_h - exact) / sem_h if sem_h > 0 else float('inf')
        print(f"  Hutchinson over {N_BATCHES} batches × {N_PER_BATCH} probes:",
              flush=True)
        print(f"    batch mean = {mean_h:+.4f}  (sem={sem_h:.4f})", flush=True)
        print(f"    z-score (mean-exact)/sem = {z:+.2f}", flush=True)
        if abs(z) < 3.0:
            print(f"    NOISE: estimator consistent with exact at "
                  f"{N_BATCHES * N_PER_BATCH} pooled probes", flush=True)
        else:
            print(f"    BIAS: estimator differs from exact by {z:.1f} sem — "
                  f"REAL BIAS", flush=True)
        print(f"    individual batch range [{batch_results.min():+.4f}, "
              f"{batch_results.max():+.4f}]", flush=True)


if __name__ == "__main__":
    main()
