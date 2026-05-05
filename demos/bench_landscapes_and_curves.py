"""
Comprehensive benchmark with both:
  (A) per-iter EFGP-loss curves using each method's own time-varying smoother.
  (B) static landscapes at EACH method's converged smoother (so we see
      both trajectories on the EFGP-smoother landscape AND on the SparseGP-
      smoother landscape).

Layout: 3 cols (one per benchmark) × 3 rows
  row 0: contours at EFGP-converged smoother + both trajectories
  row 1: contours at SparseGP-converged smoother + both trajectories
  row 2: EFGP-loss-per-iter curves (using each method's own smoother at
         each iter), with truth-loss reference

Custom fit wrappers (`fit_efgp_capture` / `fit_sparse_capture`) intercept
the EM Python loop to capture per-iter (m, S, SS, ℓ, σ²).

Run:
  ~/myenv/bin/python demos/bench_landscapes_and_curves.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import _ws_real_se, FrozenEFGPDrift as _FrozenEFGPDrift
import sing.efgp_em as efgp_em

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jla
from jax import vmap

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em, nat_grad_likelihood, nat_grad_transition
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import (
    natural_to_marginal_params, natural_to_mean_params,
    dynamics_to_natural_params, update_init_params)
from sing.efgp_emissions import update_emissions_gaussian


D = 2
N_EM = 50
LS_INIT = 0.7
VAR_INIT = 1.0
K_FD = 5

LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)
N_GRID_LL = 21
N_GRID_LV = 21


_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

BENCHMARKS = [
    dict(
        name='Damped rotation (linear)', short='linear',
        drift_fn=lambda x, t: _A_rot @ x,
        x0=jnp.array([2.0, 0.0]),
        T=400, t_max=8.0, sigma=0.3, N_obs=8, seed=7,
    ),
    dict(
        name='Duffing double-well', short='duffing',
        drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
        x0=jnp.array([1.2, 0.0]),
        T=400, t_max=15.0, sigma=0.2, N_obs=8, seed=13,
    ),
    dict(
        name='Anharmonic oscillator', short='anharmonic',
        drift_fn=lambda x, t: jnp.stack([x[1], -x[0] - 0.3*x[1] - 0.5*x[0]**3]),
        x0=jnp.array([1.5, 0.0]),
        T=400, t_max=10.0, sigma=0.3, N_obs=8, seed=21,
    ),
]


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg):
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=cfg['t_max'],
                      n_timesteps=cfg['T'], sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = rng.standard_normal((cfg['N_obs'], D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


# ----------------------------------------------------------------------------
# EFGP fit wrapper that captures per-iter (m, S, SS) + (ℓ, σ²)
# ----------------------------------------------------------------------------
def fit_efgp_capture(lik, op, ip, t_grid, sigma, mstep_lr=0.01):
    """Fork of fit_efgp_sing_jax that captures per-iter smoother marginals
    AND post-M-step hypers.  Returns (mp_final, ls_per_iter, var_per_iter,
    ms_per_iter, Ss_per_iter, SSs_per_iter, wall_time, top_per_iter,
    grid_per_iter, mu_r_per_iter, z_r_per_iter)."""
    from sing.efgp_em import _build_jit_estep_scan_jax
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    ys_obs = lik.ys_obs; t_mask = lik.t_mask
    n_trials, T, _ = ys_obs.shape
    trial_mask = jnp.ones((n_trials, T), dtype=bool)
    output_params = op
    init_params = ip

    nat_to_marg_b = jax.jit(vmap(natural_to_marginal_params))

    # E-step builder
    scan_estep_inner_refresh, _, _ = _build_jit_estep_scan_jax(
        D=D, T=T, t_grid=t_grid, lik=lik, sigma=sigma,
        sigma_drift_sq=sigma ** 2,
        S_marginal=2, n_estep_iters=10,
        qf_cg_tol=1e-5, qf_max_cg_iter=2000, qf_nufft_eps=1e-7,
    )

    # X_template (default support)
    diag_std0 = jnp.sqrt(jnp.diag(jnp.asarray(init_params['V0'][0])))
    half_span = jnp.maximum(jnp.full_like(diag_std0, 3.0), 4.0 * diag_std0)
    X_template = (jnp.asarray(init_params['mu0'][0])[None, :]
                  + jnp.linspace(-1., 1., max(T, 16))[:, None] * half_span[None, :])
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)

    ls = float(LS_INIT); var_ = float(VAR_INIT)
    log_ls = math.log(ls); log_var = math.log(var_)

    # Adaptive-h grid (matches default fit policy)
    def make_grid(ls_, var_):
        return jp.spectral_grid_se_fixed_K(
            log_ls=jnp.log(jnp.asarray(ls_, dtype=jnp.float32)),
            log_var=jnp.log(jnp.asarray(var_, dtype=jnp.float32)),
            K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    grid = make_grid(ls, var_)

    # Init natural params (diffusion-only SDE)
    A_init = jnp.zeros((T - 1, D, D))
    b_init = jnp.zeros((T, D)).at[0].set(jnp.asarray(init_params['mu0'][0]))
    Q_init = jnp.tile((sigma ** 2) * jnp.eye(D), (T, 1, 1)).at[0].set(
        jnp.asarray(init_params['V0'][0]))
    nat_one = dynamics_to_natural_params(A_init, b_init, Q_init, trial_mask[0])
    natural_params = jax.tree_util.tree_map(lambda x: x[None], nat_one)

    init_key, key = jr.split(jr.PRNGKey(0), 2)

    # Storage
    ms_per = []; Ss_per = []; SSs_per = []
    ls_per = [ls]; var_per = [var_]
    top_per = []; grid_per = []; mu_r_per = []; z_r_per = []

    t0 = time.perf_counter()
    for it in range(N_EM):
        rho_jax = jnp.asarray(float(rho_sched[it]), dtype=jnp.float32)
        em_key, key = jr.split(key, 2)
        natural_params = scan_estep_inner_refresh(
            natural_params, init_params,
            ys_obs, t_mask, output_params,
            em_key, rho_jax,
            grid.xis_flat, grid.ws,
            grid.h_per_dim, grid.xcen,
            mtot_per_dim=grid.mtot_per_dim)

        mp_b, _ = nat_to_marg_b(natural_params, trial_mask)
        ms_t = mp_b['m'][0]; Ss_t = mp_b['S'][0]; SSs_t = mp_b['SS'][0]
        ms_per.append(np.asarray(ms_t))
        Ss_per.append(np.asarray(Ss_t))
        SSs_per.append(np.asarray(SSs_t))

        # Closed-form emissions update (after warmup)
        if it >= 8:
            C_new, d_new, R_new = update_emissions_gaussian(
                ms_t, Ss_t, ys_obs[0], t_mask[0], update_R=False)
            output_params = {**output_params, 'C': C_new, 'd': d_new}
        init_params = vmap(update_init_params)(ms_t[None, 0], Ss_t[None, 0])

        # M-step (after warmup)
        if it >= 8:
            mstep_key, key = jr.split(key, 2)
            del_t = t_grid[1:] - t_grid[:-1]
            mu_r, _, top = jpd.compute_mu_r_jax(
                ms_t, Ss_t, SSs_t, del_t, grid, mstep_key,
                sigma_drift_sq=sigma ** 2, S_marginal=2,
                D_lat=D, D_out=D, cg_tol=1e-5, max_cg_iter=2000,
                nufft_eps=1e-7)
            ws_real_c = grid.ws.real.astype(grid.ws.dtype)
            ws_safe = jnp.where(jnp.abs(ws_real_c) < 1e-30,
                                jnp.array(1e-30, dtype=ws_real_c.dtype),
                                ws_real_c)
            from sing.efgp_jax_primitives import make_A_apply
            A_at_theta0 = make_A_apply(grid.ws, top, sigmasq=1.0)
            h_r0 = jax.vmap(A_at_theta0)(mu_r)
            z_r = h_r0 / ws_safe
            log_ls_new, log_var_new, _ = jpd.m_step_kernel_jax(
                log_ls, log_var,
                mu_r_fixed=mu_r, z_r=z_r, top=top,
                xis_flat=grid.xis_flat, h_per_dim=grid.h_per_dim,
                D_lat=D, D_out=D,
                n_inner=4, lr=mstep_lr,
                n_hutchinson=4, include_trace=True,
                key=mstep_key)
            new_ls = float(jnp.exp(log_ls_new))
            new_var = float(jnp.exp(log_var_new))
            if 0.05 < new_ls < 50.0 and 0.01 < new_var < 100.0:
                ls = new_ls; var_ = new_var
                log_ls = float(log_ls_new); log_var = float(log_var_new)
                grid = make_grid(ls, var_)
            top_per.append(top)
            grid_per.append(grid)
            mu_r_per.append(mu_r)
            z_r_per.append(z_r)
        else:
            top_per.append(None); grid_per.append(None)
            mu_r_per.append(None); z_r_per.append(None)
        ls_per.append(ls); var_per.append(var_)
    wall = time.perf_counter() - t0

    mp_final, _ = nat_to_marg_b(natural_params, trial_mask)
    return dict(
        mp_final=mp_final,
        ls_per_iter=np.array(ls_per),
        var_per_iter=np.array(var_per),
        ms_per_iter=ms_per, Ss_per_iter=Ss_per, SSs_per_iter=SSs_per,
        top_per_iter=top_per, grid_per_iter=grid_per,
        mu_r_per_iter=mu_r_per, z_r_per_iter=z_r_per,
        wall=wall,
    )


# ----------------------------------------------------------------------------
# SparseGP fit wrapper that captures per-iter (m, S, SS) + (ℓ, σ²)
# ----------------------------------------------------------------------------
def fit_sparse_capture(lik, op, ip, t_grid, sigma, lr=3e-3):
    """Fork of fit_variational_em outer loop with per-iter capture.

    Calls fit_variational_em with n_iters=1, repeatedly, threading state.
    """
    from sing.sing import sing_update, fit_variational_em as _fit
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                         output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)

    ms_per = []; Ss_per = []; SSs_per = []
    ls_per = [LS_INIT]; var_per = [VAR_INIT]

    init_params_cur = ip
    output_params_cur = op
    drift_history = []

    t0 = time.perf_counter()
    # Just call fit_variational_em with the full schedule + drift_params_history
    mp, _, _, _, _, _, _, _ = _fit(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params, init_params=init_params_cur,
        output_params=output_params_cur, sigma=sigma,
        rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10, n_iters_m=4,
        perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((N_EM,), lr),
        print_interval=999, drift_params_history=drift_history)
    wall = time.perf_counter() - t0

    # SparseGP fit doesn't expose per-iter mp.  We approximate by re-running
    # fixed-hyper EM at each (ℓ_t, σ²_t) for a few iters to "freeze" a
    # corresponding smoother.  This is a steady-state approximation of
    # what q(x) would converge to at θ_t.  Cheap because we run for only
    # 4 outer iters.
    ls_per_post = np.array([float(np.mean(np.asarray(d['length_scales'])))
                              for d in drift_history])
    var_per_post = np.array([float(np.asarray(d['output_scale']))**2
                               for d in drift_history])
    ls_per = np.concatenate([[LS_INIT], ls_per_post])
    var_per = np.concatenate([[VAR_INIT], var_per_post])
    return dict(
        mp_final=mp,
        ls_per_iter=ls_per,
        var_per_iter=var_per,
        wall=wall,
    )


# ----------------------------------------------------------------------------
# Loss eval at a given smoother + θ
# ----------------------------------------------------------------------------
def build_state_from_smoother(ms, Ss, SSs, t_grid, sigma, ls_pin, var_pin):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, top = jpd.compute_mu_r_jax(
        jnp.asarray(ms), jnp.asarray(Ss), jnp.asarray(SSs),
        del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r)
    z_r = h0 / ws_safe
    return grid, top, z_r


def loss_at(grid, top, z_r, log_ls, log_var):
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                          jnp.asarray(log_var, dtype=jnp.float32),
                          grid.xis_flat, grid.h_per_dim[0], D)
    ws_c = ws_real.astype(cdtype)
    v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    ns_v = tuple(2*n-1 for n in top.ns)
    v_conv = v_pad[tuple(slice(0, L) for L in ns_v)]
    d_ = len(top.ns)
    mi = jnp.indices(top.ns).reshape(d_, -1)
    offset = jnp.array([n-1 for n in top.ns], dtype=jnp.int32)
    diff = mi[:, :, None] - mi[:, None, :] + offset[:, None, None]
    T_mat = v_conv[tuple(diff[k] for k in range(d_))]
    eye_c = jnp.eye(M, dtype=cdtype)
    A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
    L = jnp.linalg.cholesky(A)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
    h = ws_c[None, :] * z_r
    def solve_one(b):
        y = jla.solve_triangular(L, b, lower=True)
        return jla.solve_triangular(L.conj().T, y, lower=False)
    mu = jax.vmap(solve_one)(h)
    det_loss = -0.5 * jnp.sum(jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
    return float(det_loss + 0.5 * D * logdet)


# Approximate "smoother at θ_t" via short fixed-hyper EFGP run
def smoother_at_theta(lik, op, ip, t_grid, sigma, ls_pin, var_pin,
                       n_em_warmup=4):
    rho_sched = jnp.linspace(0.05, 0.7, n_em_warmup)
    mp, *_ = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_pin, variance=var_pin, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=n_em_warmup, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=False, update_R=False, learn_kernel=False,
        kernel_warmup_iters=n_em_warmup, verbose=False)
    return (np.asarray(mp['m'][0]), np.asarray(mp['S'][0]),
            np.asarray(mp['SS'][0]))


def run_one_benchmark(cfg):
    print(f"\n[bench] === {cfg['name']} ===", flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(cfg)
    sigma = cfg['sigma']

    print(f"  Running EFGP (Cholesky) with per-iter capture...", flush=True)
    e = fit_efgp_capture(lik, op, ip, t_grid, sigma, mstep_lr=0.01)
    print(f"    final ℓ={e['ls_per_iter'][-1]:.3f}, σ²={e['var_per_iter'][-1]:.3f}, "
          f"wall={e['wall']:.1f}s", flush=True)

    print(f"  Running SparseGP @ lr=3e-3...", flush=True)
    s = fit_sparse_capture(lik, op, ip, t_grid, sigma, lr=3e-3)
    print(f"    final ℓ={s['ls_per_iter'][-1]:.3f}, σ²={s['var_per_iter'][-1]:.3f}, "
          f"wall={s['wall']:.1f}s", flush=True)

    # === Build per-method final-smoother landscape (option B) ===
    print(f"  Building EFGP-converged-smoother landscape...", flush=True)
    e_grid_final, e_top_final, e_zr_final = build_state_from_smoother(
        e['ms_per_iter'][-1], e['Ss_per_iter'][-1], e['SSs_per_iter'][-1],
        t_grid, sigma,
        e['ls_per_iter'][-1], e['var_per_iter'][-1])
    LOG_LS = np.linspace(*LOG_LS_RANGE, N_GRID_LL)
    LOG_VAR = np.linspace(*LOG_VAR_RANGE, N_GRID_LV)
    L_efgp_smoother = np.zeros((N_GRID_LL, N_GRID_LV))
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            L_efgp_smoother[i, j] = loss_at(
                e_grid_final, e_top_final, e_zr_final, float(ll), float(lv))
    eb_idx = np.unravel_index(np.argmin(L_efgp_smoother), L_efgp_smoother.shape)
    eb_ll = float(LOG_LS[eb_idx[0]]); eb_lv = float(LOG_VAR[eb_idx[1]])
    print(f"    EFGP-smoother loss min: ℓ={math.exp(eb_ll):.2f}, "
          f"σ²={math.exp(eb_lv):.2f}", flush=True)

    print(f"  Building SparseGP-converged-smoother landscape (via fixed-hyper "
          f"EFGP smoother at SparseGP final hypers)...", flush=True)
    # SparseGP doesn't expose per-iter mp; approximate via fixed-hyper EFGP at
    # SparseGP's final (ℓ, σ²)
    s_ms, s_Ss, s_SSs = smoother_at_theta(
        lik, op, ip, t_grid, sigma,
        float(s['ls_per_iter'][-1]), float(s['var_per_iter'][-1]))
    s_grid_final, s_top_final, s_zr_final = build_state_from_smoother(
        s_ms, s_Ss, s_SSs, t_grid, sigma,
        float(s['ls_per_iter'][-1]), float(s['var_per_iter'][-1]))
    L_sparse_smoother = np.zeros((N_GRID_LL, N_GRID_LV))
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            L_sparse_smoother[i, j] = loss_at(
                s_grid_final, s_top_final, s_zr_final, float(ll), float(lv))
    sb_idx = np.unravel_index(np.argmin(L_sparse_smoother), L_sparse_smoother.shape)
    sb_ll = float(LOG_LS[sb_idx[0]]); sb_lv = float(LOG_VAR[sb_idx[1]])
    print(f"    SparseGP-smoother loss min: ℓ={math.exp(sb_ll):.2f}, "
          f"σ²={math.exp(sb_lv):.2f}", flush=True)

    # === Per-iter loss curves (option A) ===
    print(f"  Computing per-iter loss for EFGP traj using EFGP per-iter "
          f"smoothers...", flush=True)
    e_loss_curve = []
    for t in range(N_EM):
        if e['top_per_iter'][t] is None:
            # Pre-warmup iter: use init smoother
            grid_t, top_t, zr_t = build_state_from_smoother(
                e['ms_per_iter'][t], e['Ss_per_iter'][t], e['SSs_per_iter'][t],
                t_grid, sigma, LS_INIT, VAR_INIT)
        else:
            grid_t = e['grid_per_iter'][t]
            top_t = e['top_per_iter'][t]
            zr_t = e['z_r_per_iter'][t]
        e_loss_curve.append(loss_at(
            grid_t, top_t, zr_t,
            float(np.log(e['ls_per_iter'][t+1])),
            float(np.log(e['var_per_iter'][t+1]))))

    print(f"  Computing per-iter loss for SparseGP traj (approx smoother at "
          f"θ_t via fixed-hyper EFGP)...", flush=True)
    # For SparseGP, we don't have per-iter smoothers.  Approximate by
    # building a steady-state smoother at each (ℓ_t, σ²_t).  Sparse this to
    # ~10 checkpoints to keep cost manageable.
    s_loss_curve = []
    n_iters = len(s['ls_per_iter']) - 1  # exclude init
    s_checkpoints = list(range(0, n_iters, max(1, n_iters // 12)))
    if s_checkpoints[-1] != n_iters - 1:
        s_checkpoints.append(n_iters - 1)
    for t in s_checkpoints:
        ls_t = float(s['ls_per_iter'][t+1])
        var_t = float(s['var_per_iter'][t+1])
        s_ms_t, s_Ss_t, s_SSs_t = smoother_at_theta(
            lik, op, ip, t_grid, sigma, ls_t, var_t)
        grid_t, top_t, zr_t = build_state_from_smoother(
            s_ms_t, s_Ss_t, s_SSs_t, t_grid, sigma, ls_t, var_t)
        s_loss_curve.append(loss_at(
            grid_t, top_t, zr_t, math.log(ls_t), math.log(var_t)))

    out = dict(
        name=cfg['name'], short=cfg['short'],
        LOG_LS_GRID=LOG_LS, LOG_VAR_GRID=LOG_VAR,
        L_efgp_smoother=L_efgp_smoother, eb_ll=eb_ll, eb_lv=eb_lv,
        L_efgp_min=float(L_efgp_smoother[eb_idx]),
        L_sparse_smoother=L_sparse_smoother, sb_ll=sb_ll, sb_lv=sb_lv,
        L_sparse_min=float(L_sparse_smoother[sb_idx]),
        e_ls=e['ls_per_iter'], e_var=e['var_per_iter'], e_wall=e['wall'],
        s_ls=s['ls_per_iter'], s_var=s['var_per_iter'], s_wall=s['wall'],
        e_loss_curve=np.array(e_loss_curve),
        s_loss_curve=np.array(s_loss_curve),
        s_checkpoints=np.array(s_checkpoints),
    )
    np.savez(f"/tmp/bench_lc_{cfg['short']}.npz", **out)
    return out


def main():
    print(f"[bench] Landscapes (per-method smoother) + per-iter loss curves",
          flush=True)
    print(f"[bench] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), n_em={N_EM}", flush=True)
    results = [run_one_benchmark(cfg) for cfg in BENCHMARKS]

    # Plot 3 cols × 3 rows
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        for col, r in enumerate(results):
            LOG_LS = r['LOG_LS_GRID']; LOG_VAR = r['LOG_VAR_GRID']
            # Row 0: EFGP-smoother landscape
            ax = axes[0, col]
            L_norm = r['L_efgp_smoother'] - r['L_efgp_min']
            levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
            cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                              levels=[0]+levels, cmap='viridis_r', extend='max')
            cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                            levels=levels, colors='k', linewidths=0.4)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
            for log_a in [-2, -1, 0, 1, 2]:
                ax.plot(LOG_LS, LOG_LS*2 + log_a,
                        color='white', linestyle=':', linewidth=0.6, alpha=0.5)
            ax.plot(np.log(r['e_ls']), np.log(r['e_var']), '-o', color='C0',
                    markersize=3, linewidth=1.5,
                    label=f"EFGP  [{r['e_wall']:.0f}s]")
            ax.plot(np.log(r['s_ls']), np.log(r['s_var']), '-s', color='C3',
                    markersize=3, linewidth=1.4,
                    label=f"SparseGP  [{r['s_wall']:.0f}s]")
            ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                        marker='+', s=200, color='black', zorder=6, label='init')
            ax.scatter([r['eb_ll']], [r['eb_lv']], marker='*', s=240,
                        color='gold', edgecolor='k', zorder=7, label='loss min')
            ax.set_xlim(LOG_LS.min(), LOG_LS.max())
            ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
            if col == 0:
                ax.set_ylabel('log σ²', fontsize=10)
            ax.set_xlabel('log ℓ', fontsize=9)
            ax.set_title(f"{r['name']}\nlandscape @ EFGP-converged smoother",
                          fontsize=9)
            ax.legend(loc='lower right', fontsize=7)

            # Row 1: SparseGP-smoother landscape
            ax = axes[1, col]
            L_norm = r['L_sparse_smoother'] - r['L_sparse_min']
            cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                              levels=[0]+levels, cmap='viridis_r', extend='max')
            cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                            levels=levels, colors='k', linewidths=0.4)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
            for log_a in [-2, -1, 0, 1, 2]:
                ax.plot(LOG_LS, LOG_LS*2 + log_a,
                        color='white', linestyle=':', linewidth=0.6, alpha=0.5)
            ax.plot(np.log(r['e_ls']), np.log(r['e_var']), '-o', color='C0',
                    markersize=3, linewidth=1.5,
                    label=f"EFGP  [{r['e_wall']:.0f}s]")
            ax.plot(np.log(r['s_ls']), np.log(r['s_var']), '-s', color='C3',
                    markersize=3, linewidth=1.4,
                    label=f"SparseGP  [{r['s_wall']:.0f}s]")
            ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                        marker='+', s=200, color='black', zorder=6, label='init')
            ax.scatter([r['sb_ll']], [r['sb_lv']], marker='*', s=240,
                        color='gold', edgecolor='k', zorder=7, label='loss min')
            ax.set_xlim(LOG_LS.min(), LOG_LS.max())
            ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
            if col == 0:
                ax.set_ylabel('log σ²', fontsize=10)
            ax.set_xlabel('log ℓ', fontsize=9)
            ax.set_title('landscape @ SparseGP-converged smoother',
                          fontsize=9)
            ax.legend(loc='lower right', fontsize=7)

            # Row 2: per-iter loss curves
            ax = axes[2, col]
            iters_e = np.arange(1, len(r['e_loss_curve']) + 1)
            ax.plot(iters_e, r['e_loss_curve'], '-o', color='C0',
                    markersize=3, linewidth=1.5,
                    label='EFGP (own smoother per iter)')
            ax.plot(r['s_checkpoints'] + 1, r['s_loss_curve'], '-s', color='C3',
                    markersize=4, linewidth=1.5,
                    label='SparseGP (steady-state smoother @ θ_t)')
            ax.set_xlabel('EM iter', fontsize=9)
            if col == 0:
                ax.set_ylabel('EFGP collapsed loss\n(lower = closer to MAP)',
                              fontsize=9)
            ax.set_title('per-iter EFGP loss', fontsize=9)
            ax.legend(loc='best', fontsize=7)
            ax.grid(alpha=0.3)

        # Single colorbar on the right
        cbar_ax = fig.add_axes([0.93, 0.4, 0.012, 0.45])
        fig.colorbar(cf, cax=cbar_ax, label='L − L_min')
        fig.suptitle('Hyper trajectories on EFGP collapsed log-marginal '
                     'landscape (top: at EFGP smoother; mid: at SparseGP '
                     'smoother); per-iter loss curves (bottom)',
                     fontsize=11)
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        plt.savefig('/tmp/bench_lc.png', dpi=120)
        print(f"\n[bench] Saved /tmp/bench_lc.png", flush=True)
    except Exception as e:
        print(f"[bench] Plot failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    print(f"\n[bench] Wall-time summary:", flush=True)
    for r in results:
        e = r['e_wall']; s = r['s_wall']
        print(f"  {r['name']:25s}  EFGP={e:>5.1f}s  Sparse={s:>5.1f}s  "
              f"speedup={s/e:.2f}x", flush=True)


if __name__ == '__main__':
    main()
