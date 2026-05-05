"""
Hyper trajectories on BOTH objectives:
  - EFGP collapsed log-marginal (the EFGP M-step loss)
  - SparseGP −ELBO (the SparseGP M-step loss)

For each benchmark and each objective, plot the loss landscape with both
methods' trajectories overlaid.  Also include per-iter own-objective
loss curves.

Layout: 3 cols (one per benchmark) × 3 rows
  row 0: EFGP collapsed loss surface + both trajectories
  row 1: SparseGP −ELBO surface + both trajectories
  row 2: per-iter own-objective loss curves

Both surfaces are evaluated at the corresponding method's converged
state (smoother + inducing pts), with kernel hypers (ℓ, σ²) varying.

Run:
  ~/myenv/bin/python demos/bench_two_objectives.py
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

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import _ws_real_se
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
from sing.sing import fit_variational_em, compute_elbo_over_batch
from sing.initialization import initialize_zs
from sing.utils.sing_helpers import natural_to_marginal_params


D = 2
import os as _os_init
N_EM = int(_os_init.environ.get('N_EM', '50'))
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


def procrustes_align(m_inf, m_true):
    """Return (A, b) such that m_inf @ A.T + b minimizes RMSE vs m_true."""
    Xi, Xt = np.asarray(m_inf), np.asarray(m_true)
    bi, bt = Xi.mean(0), Xt.mean(0)
    A_T, *_ = np.linalg.lstsq(Xi - bi, Xt - bt, rcond=None)
    A = A_T.T
    return A, bt - A @ bi


def compute_lat_rmse(mp, xs_true):
    """Procrustes-aligned latent RMSE between smoother mean and ground truth."""
    m_inf = np.asarray(mp['m'][0])
    A, b = procrustes_align(m_inf, np.asarray(xs_true))
    aligned = m_inf @ A.T + b
    return float(np.sqrt(np.mean((aligned - np.asarray(xs_true))**2)))


def make_data(cfg, *, oracle_emissions: bool = False):
    """If oracle_emissions=True, return op with C=C_true (frozen oracle frame)
    so emissions identifiability is removed and the drift inference can be
    judged on its own merits."""
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=cfg['t_max'],
                      n_timesteps=cfg['T'], sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    if oracle_emissions:
        op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                   R=jnp.full((cfg['N_obs'],), 0.1))
    else:
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
# EFGP fit (using new Cholesky M-step automatically)
# ----------------------------------------------------------------------------
def fit_efgp(lik, op, ip, t_grid, sigma, mstep_lr=0.01,
              learn_emissions=True):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=learn_emissions, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return dict(
        mp=mp, ls_traj=ls_traj, var_traj=var_traj, wall=wall,
    )


# ----------------------------------------------------------------------------
# SparseGP fit + capture state needed for -ELBO landscape
# ----------------------------------------------------------------------------
def fit_sparsegp(lik, op, ip, t_grid, sigma, lr=0.01,
                  learn_emissions=True,
                  n_iters_m=4, zs_lim=2.5, num_per_dim=8):
    """Matched comparison: same lr, same n_iters_m as EFGP. Stability
    comes from Cholesky-based gradients in sde.py (no inv(Kzz),
    Cholesky-stable KL in prior_term)."""
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=zs_lim, num_per_dim=num_per_dim)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)  # matched to EFGP
    history = []
    t0 = time.perf_counter()
    mp, nat_p, gp_post, dp_final, ip_final, op_final, ie_final, _ = (
        fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
            n_iters_m=n_iters_m, perform_m_step=True,
            learn_output_params=learn_emissions,
            learning_rate=jnp.full((N_EM,), lr),
            print_interval=999, drift_params_history=history))
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT]
                        + [float(np.mean(np.asarray(d['length_scales'])))
                           for d in history])
    var_traj = np.array([VAR_INIT]
                         + [float(np.asarray(d['output_scale'])) ** 2
                            for d in history])
    return dict(
        mp=mp, nat_p=nat_p, dp=dp_final, ip=ip_final, op=op_final,
        ie=ie_final, sparse=sparse, gp_post=gp_post, sd=sparse,
        ls_traj=ls_traj, var_traj=var_traj, wall=wall,
    )


# ----------------------------------------------------------------------------
# EFGP collapsed loss landscape at a frozen smoother
# ----------------------------------------------------------------------------
def build_efgp_state(mp, t_grid, sigma, ls_pin, var_pin):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r)
    z_r = h0 / ws_safe
    return grid, top, z_r


def efgp_loss_at(grid, top, z_r, log_ls, log_var):
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


# ----------------------------------------------------------------------------
# Ground-truth landscape: -log p(x_true; θ) on the true SDE realization
# (smoother-free; usable in synthetic experiments where xs are known).
# ----------------------------------------------------------------------------
def gt_landscape(xs, sigma_drift, t_grid, LOG_LS, LOG_VAR):
    """Compute -log p(x_true; θ) on a (log_ls, log_var) grid.

    Marginalising the GP drift f from the SDE under known x gives a joint
    Gaussian on velocities y_t = (x_{t+1} − x_t)/Δt with covariance
    K_θ + (σ²_drift/Δt)·I, where K_θ is the dense SE Gram on xs[:-1].

    log p(y; θ) per output dim: -½ y^T A^{-1} y − ½ log|A| − ½ (T-1) log(2π).
    Sum over D output dims.

    Returns L = -log p (so minimization → MLE).
    """
    import scipy.linalg
    xs_np = np.asarray(xs)
    T = xs_np.shape[0]
    inputs = xs_np[:-1]                         # (T-1, D)
    dt = float(np.asarray(t_grid[1] - t_grid[0]))
    velocities = (xs_np[1:] - xs_np[:-1]) / dt  # (T-1, D)
    n_obs = T - 1
    D_out = velocities.shape[1]

    # Pairwise squared distances on the cloud
    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs ** 2).sum(-1)             # (T-1, T-1)
    noise_var = sigma_drift ** 2 / dt
    eye_n = np.eye(n_obs)

    L_grid = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        ell = math.exp(float(ll))
        K_unscaled = np.exp(-0.5 * sq_dists / (ell ** 2))
        for j, lv in enumerate(LOG_VAR):
            var_ = math.exp(float(lv))
            A_ = var_ * K_unscaled + noise_var * eye_n
            try:
                L_chol = np.linalg.cholesky(A_)
            except np.linalg.LinAlgError:
                L_grid[i, j] = np.nan
                continue
            logdet = 2.0 * np.log(np.diag(L_chol)).sum()
            quad_total = 0.0
            for d in range(D_out):
                y = velocities[:, d]
                z = scipy.linalg.solve_triangular(L_chol, y, lower=True)
                quad_total += float(z @ z)
            # -log p (minimisation form), drop θ-independent log(2π) const
            L_grid[i, j] = 0.5 * quad_total + 0.5 * D_out * logdet
    return L_grid


def gt_loss_at(xs, sigma_drift, t_grid, log_ls, log_var):
    """Single-point eval of -log p(x_true; θ)."""
    return float(gt_landscape(xs, sigma_drift, t_grid,
                               np.array([log_ls]), np.array([log_var]))[0, 0])


# ----------------------------------------------------------------------------
# SparseGP −ELBO landscape at frozen q(x) + inducing points
# ----------------------------------------------------------------------------
def build_sparse_neg_elbo_fn(s, lik, t_grid, sigma):
    """Return a JIT'd fn (log_ls, log_var) -> -ELBO using frozen q(x) +
    inducing points from the SparseGP final state."""
    nat_p = s['nat_p']
    mp = s['mp']
    op = s['op']
    ip = s['ip']
    ie = s['ie']
    sparse = s['sparse']
    # inducing_points may not be in dp_final if they weren't included in the
    # initial drift_params (M-step only updates keys present in drift_params).
    # Fall back to sparse.zs (the class attr — initial inducing points).
    inducing_frozen = s['dp'].get('inducing_points', sparse.zs)

    T = lik.ys_obs.shape[1]
    trial_mask = jnp.ones((1, T), dtype=bool)
    _, ap = vmap(natural_to_marginal_params)(nat_p, trial_mask)
    inputs = jnp.zeros((1, T, 1))

    def neg_elbo(log_ls, log_var):
        drift_params = dict(
            length_scales=jnp.full((D,), jnp.exp(log_ls)),
            output_scale=jnp.exp(0.5 * log_var),
            inducing_points=inducing_frozen,
        )
        return -compute_elbo_over_batch(
            jr.PRNGKey(0), lik.ys_obs, lik.t_mask, trial_mask,
            sparse, lik, t_grid, drift_params,
            ip, op, nat_p, mp, ap, inputs, ie, sigma)[0]

    return jax.jit(neg_elbo)


def _eval_efgp_drift(mp, ls_pin, var_pin, t_grid, sigma, grid_pts,
                       eps_predict=1e-6):
    """Evaluate EFGP posterior drift mean on a 2D point cloud.

    Uses ``spectral_grid_se`` with a tight ``eps_predict=1e-6`` to rebuild
    the grid once at the converged hypers — auto-picking both K and h to
    achieve the requested spectral truncation tolerance.  This is OK at
    predict time because we're not jit-cached on a fixed K_per_dim like in
    the M-step; we get the natural eps-tolerant grid for the final ℓ.
    """
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(
        jnp.asarray(ls_pin, dtype=jnp.float32),
        jnp.asarray(var_pin, dtype=jnp.float32),
        X_template, eps=eps_predict)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r, _, _ = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    return np.asarray(Ef)


def _eval_sp_drift(s, grid_pts):
    """Evaluate SparseGP posterior drift mean on a 2D point cloud."""
    return np.asarray(s['sd'].get_posterior_f_mean(
        s['gp_post'] if 'gp_post' in s else s['nat_p'],
        s['dp'], jnp.asarray(grid_pts, dtype=jnp.float32)))


def run_one_benchmark(cfg, *, oracle_emissions=False):
    print(f"\n[bench] === {cfg['name']} ===", flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(
        cfg, oracle_emissions=oracle_emissions)
    sigma = cfg['sigma']
    xs_np = np.asarray(xs)
    learn_em = not oracle_emissions
    print(f"  emissions: {'ORACLE C frozen' if oracle_emissions else 'SVD init, learned'}",
          flush=True)

    # ---- Fit both methods ----
    print(f"  Running EFGP (Cholesky) at mstep_lr=0.01...", flush=True)
    e = fit_efgp(lik, op, ip, t_grid, sigma, mstep_lr=0.01,
                  learn_emissions=learn_em)
    e_lat_rmse = compute_lat_rmse(e['mp'], xs_np)
    print(f"    final ℓ={e['ls_traj'][-1]:.3f}, σ²={e['var_traj'][-1]:.3f}, "
          f"wall={e['wall']:.1f}s, lat_rmse={e_lat_rmse:.4f}", flush=True)

    import os as _os
    sp_lr_env = _os.environ.get('SPARSEGP_LR', 'auto')
    if sp_lr_env == 'auto':
        # Fully matched: same lr as EFGP, same n_iters_m=4, same rho_sched.
        # The Cholesky-based gradient in sde.py makes this stable.
        sp_lr = 1e-2
        print(f"  SparseGP matched LR: lr={sp_lr:.0e}, n_iters_m=4",
              flush=True)
    else:
        sp_lr = float(sp_lr_env)
    print(f"  Running SparseGP @ lr={sp_lr:.0e}...", flush=True)
    s = fit_sparsegp(lik, op, ip, t_grid, sigma, lr=sp_lr,
                      learn_emissions=learn_em)
    s_lat_rmse = compute_lat_rmse(s['mp'], xs_np)
    print(f"    final ℓ={s['ls_traj'][-1]:.3f}, σ²={s['var_traj'][-1]:.3f}, "
          f"wall={s['wall']:.1f}s, lat_rmse={s_lat_rmse:.4f}", flush=True)

    LOG_LS = np.linspace(*LOG_LS_RANGE, N_GRID_LL)
    LOG_VAR = np.linspace(*LOG_VAR_RANGE, N_GRID_LV)

    # ---- EFGP collapsed loss landscape (at EFGP-converged smoother) ----
    print(f"  Building EFGP loss landscape (at EFGP-converged smoother)...",
          flush=True)
    e_grid, e_top, e_zr = build_efgp_state(
        e['mp'], t_grid, sigma, e['ls_traj'][-1], e['var_traj'][-1])
    L_efgp = np.zeros((N_GRID_LL, N_GRID_LV))
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            L_efgp[i, j] = efgp_loss_at(e_grid, e_top, e_zr,
                                          float(ll), float(lv))
    eb = np.unravel_index(np.nanargmin(L_efgp), L_efgp.shape)
    eb_ll = float(LOG_LS[eb[0]]); eb_lv = float(LOG_VAR[eb[1]])
    print(f"    EFGP loss min: ℓ={math.exp(eb_ll):.3f}, "
          f"σ²={math.exp(eb_lv):.3f}", flush=True)

    # ---- SparseGP -ELBO landscape (at SparseGP-converged state) ----
    print(f"  Building SparseGP -ELBO landscape (at SparseGP-converged "
          f"state, frozen q(x)+inducing)...", flush=True)
    neg_elbo = build_sparse_neg_elbo_fn(s, lik, t_grid, sigma)
    L_sp = np.zeros((N_GRID_LL, N_GRID_LV))
    t_landscape = time.perf_counter()
    for i, ll in enumerate(LOG_LS):
        for j, lv in enumerate(LOG_VAR):
            L_sp[i, j] = float(neg_elbo(
                jnp.asarray(ll, dtype=jnp.float32),
                jnp.asarray(lv, dtype=jnp.float32)))
    sb = np.unravel_index(np.nanargmin(L_sp), L_sp.shape)
    sb_ll = float(LOG_LS[sb[0]]); sb_lv = float(LOG_VAR[sb[1]])
    print(f"    SparseGP -ELBO min: ℓ={math.exp(sb_ll):.3f}, "
          f"σ²={math.exp(sb_lv):.3f}  ({time.perf_counter() - t_landscape:.1f}s)",
          flush=True)

    # ---- Ground-truth landscape (smoother-free, uses true xs) ----
    print(f"  Building ground-truth landscape (-log p(x_true; θ); "
          f"smoother-free dense GP marginal)...", flush=True)
    t_gt = time.perf_counter()
    L_gt = gt_landscape(xs, sigma, t_grid, LOG_LS, LOG_VAR)
    gb = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    gb_ll = float(LOG_LS[gb[0]]); gb_lv = float(LOG_VAR[gb[1]])
    print(f"    GT MLE: ℓ={math.exp(gb_ll):.3f}, σ²={math.exp(gb_lv):.3f}  "
          f"({time.perf_counter() - t_gt:.1f}s)", flush=True)

    # Eval each method's endpoint on the GT surface for direct comparison
    e_gt_endpt = gt_loss_at(xs, sigma, t_grid,
                              math.log(float(e['ls_traj'][-1])),
                              math.log(float(e['var_traj'][-1])))
    s_gt_endpt = gt_loss_at(xs, sigma, t_grid,
                              math.log(float(s['ls_traj'][-1])),
                              math.log(float(s['var_traj'][-1])))
    gt_min = float(L_gt[gb])
    print(f"    EFGP endpoint -log p(x_true) = {e_gt_endpt:.2f}  "
          f"(min={gt_min:.2f}, gap={e_gt_endpt - gt_min:.2f})", flush=True)
    print(f"    SparseGP endpoint -log p(x_true) = {s_gt_endpt:.2f}  "
          f"(gap={s_gt_endpt - gt_min:.2f})", flush=True)

    # ---- Learned drift evaluation on a grid ----
    print(f"  Evaluating learned drift fields on grid...", flush=True)
    lo = xs_np.min(0) - 0.4
    hi = xs_np.max(0) + 0.4
    g0 = np.linspace(lo[0], hi[0], 14)
    g1 = np.linspace(lo[1], hi[1], 14)
    GX, GY = np.meshgrid(g0, g1, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)
    f_true = np.array([np.asarray(cfg['drift_fn'](jnp.asarray(p), 0.))
                        for p in grid_pts])
    zero_rmse = float(np.sqrt(np.mean(f_true ** 2)))
    # Procrustes alignment: in oracle frame this is ~identity, but generally
    # the smoother frame may be rotated/scaled vs ground truth.
    A_e, b_e = procrustes_align(np.asarray(e['mp']['m'][0]), xs_np)
    A_s, b_s = procrustes_align(np.asarray(s['mp']['m'][0]), xs_np)
    grid_e_inferred = (grid_pts - b_e) @ np.linalg.inv(A_e).T
    grid_s_inferred = (grid_pts - b_s) @ np.linalg.inv(A_s).T
    f_efgp = (_eval_efgp_drift(e['mp'], float(e['ls_traj'][-1]),
                                float(e['var_traj'][-1]), t_grid, sigma,
                                grid_e_inferred) @ A_e.T)
    f_sp = (_eval_sp_drift(s, grid_s_inferred) @ A_s.T)
    e_drift_rmse = float(np.sqrt(np.mean((f_efgp - f_true) ** 2)))
    s_drift_rmse = float(np.sqrt(np.mean((f_sp - f_true) ** 2)))
    e_drift_rel = e_drift_rmse / zero_rmse
    s_drift_rel = s_drift_rmse / zero_rmse
    print(f"    EFGP drift RMSE = {e_drift_rmse:.4f} ({e_drift_rel:.1%}), "
          f"SparseGP drift RMSE = {s_drift_rmse:.4f} ({s_drift_rel:.1%})",
          flush=True)

    # ---- Per-iter own-objective loss curves (option A) ----
    # EFGP: at each θ_t, eval EFGP loss using EFGP-converged smoother
    #       (frozen smoother is a steady-state proxy; the actual per-iter
    #        smoother differs but this is the closest cheap eval).
    print(f"  Computing per-iter own-objective loss curves...", flush=True)
    e_loss_curve = np.array([
        efgp_loss_at(e_grid, e_top, e_zr,
                      math.log(float(e['ls_traj'][t])),
                      math.log(float(e['var_traj'][t])))
        for t in range(len(e['ls_traj']))])
    s_loss_curve = np.array([
        float(neg_elbo(jnp.asarray(math.log(float(s['ls_traj'][t])),
                                     dtype=jnp.float32),
                       jnp.asarray(math.log(float(s['var_traj'][t])),
                                     dtype=jnp.float32)))
        for t in range(len(s['ls_traj']))])

    out = dict(
        name=cfg['name'], short=cfg['short'],
        LOG_LS_GRID=LOG_LS, LOG_VAR_GRID=LOG_VAR,
        L_efgp=L_efgp, L_efgp_min=float(L_efgp[eb]),
        eb_ll=eb_ll, eb_lv=eb_lv,
        L_sp=L_sp, L_sp_min=float(L_sp[sb]),
        sb_ll=sb_ll, sb_lv=sb_lv,
        L_gt=L_gt, L_gt_min=gt_min,
        gb_ll=gb_ll, gb_lv=gb_lv,
        e_gt_endpt=e_gt_endpt, s_gt_endpt=s_gt_endpt,
        e_ls=e['ls_traj'], e_var=e['var_traj'], e_wall=e['wall'],
        s_ls=s['ls_traj'], s_var=s['var_traj'], s_wall=s['wall'],
        e_loss_curve=e_loss_curve, s_loss_curve=s_loss_curve,
        e_lat_rmse=e_lat_rmse, s_lat_rmse=s_lat_rmse,
        # Drift fields (for flow-field plot + RMSE summary)
        GX=GX, GY=GY, grid_pts=grid_pts,
        f_true=f_true, f_efgp=f_efgp, f_sp=f_sp,
        e_drift_rmse=e_drift_rmse, s_drift_rmse=s_drift_rmse,
        e_drift_rel=e_drift_rel, s_drift_rel=s_drift_rel,
        zero_rmse=zero_rmse, xs_np=xs_np,
    )
    suffix = '_oracle_em' if oracle_emissions else ''
    np.savez(f"/tmp/bench_two_obj_{cfg['short']}{suffix}.npz", **out)
    return out


def _adaptive_levels(L_norm, n=10):
    """Pick n contour levels matching the data's actual range."""
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    if finite.size == 0:
        return [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    lo, hi = float(finite.min()), float(finite.max())
    # Use logspace from a small floor up to ~max; keep ≥6 levels
    lo_use = max(lo, hi / 1000.0)
    return list(np.logspace(np.log10(lo_use), np.log10(hi), n))


def _render_landscape_panel(ax, LOG_LS, LOG_VAR, L_norm, e_ls, e_var,
                             s_ls, s_var, eb_ll, eb_lv, title,
                             min_label, e_wall=None, s_wall=None,
                             e_lat=None, s_lat=None, e_endpt=None,
                             s_endpt=None):
    levels = _adaptive_levels(L_norm)
    cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                      levels=[0] + levels, cmap='viridis_r', extend='max')
    cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                    levels=levels, colors='k', linewidths=0.4)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
    for log_a in [-2, -1, 0, 1, 2]:
        ax.plot(LOG_LS, LOG_LS*2 + log_a,
                color='white', linestyle=':', linewidth=0.5, alpha=0.5)
    e_label = 'EFGP traj'
    if e_wall is not None:
        e_label += f"  [{e_wall:.0f}s"
        if e_lat is not None:
            e_label += f", lat={e_lat:.3f}"
        if e_endpt is not None:
            e_label += f", Δ={e_endpt:.1f}"
        e_label += "]"
    s_label = 'SparseGP traj'
    if s_wall is not None:
        s_label += f"  [{s_wall:.0f}s"
        if s_lat is not None:
            s_label += f", lat={s_lat:.3f}"
        if s_endpt is not None:
            s_label += f", Δ={s_endpt:.1f}"
        s_label += "]"
    ax.plot(np.log(e_ls), np.log(e_var), '-o', color='C0',
            markersize=3, linewidth=1.6, label=e_label)
    ax.plot(np.log(s_ls), np.log(s_var), '-s', color='C3',
            markersize=3, linewidth=1.4, label=s_label)
    ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                marker='+', s=200, color='black', zorder=6, label='init')
    ax.scatter([eb_ll], [eb_lv], marker='*', s=240,
                color='gold', edgecolor='k', zorder=7, label=min_label)
    ax.set_xlim(LOG_LS.min(), LOG_LS.max())
    ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    ax.set_xlabel('log ℓ', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower right', fontsize=7)


def render(results, out_png='/tmp/bench_two_obj.png', oracle_emissions=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(20, 22))

    for col, r in enumerate(results):
        LOG_LS = r['LOG_LS_GRID']; LOG_VAR = r['LOG_VAR_GRID']

        # Row 0: ground-truth -log p(x_true; θ) — smoother-free, gold standard
        ax = axes[0, col]
        L_norm_gt = r['L_gt'] - r['L_gt_min']
        e_gap = r['e_gt_endpt'] - r['L_gt_min']
        s_gap = r['s_gt_endpt'] - r['L_gt_min']
        _render_landscape_panel(
            ax, LOG_LS, LOG_VAR, L_norm_gt,
            r['e_ls'], r['e_var'], r['s_ls'], r['s_var'],
            r['gb_ll'], r['gb_lv'],
            title=f"{r['name']}\nGround-truth −log p(x_true; θ)  "
                  "[smoother-free oracle]",
            min_label=f"GT MLE: ℓ={math.exp(r['gb_ll']):.2f}, "
                      f"σ²={math.exp(r['gb_lv']):.2f}",
            e_wall=r['e_wall'], s_wall=r['s_wall'],
            e_lat=r['e_lat_rmse'], s_lat=r['s_lat_rmse'],
            e_endpt=e_gap, s_endpt=s_gap)
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)

        # Row 1: EFGP collapsed loss surface
        ax = axes[1, col]
        L_norm_e = r['L_efgp'] - r['L_efgp_min']
        _render_landscape_panel(
            ax, LOG_LS, LOG_VAR, L_norm_e,
            r['e_ls'], r['e_var'], r['s_ls'], r['s_var'],
            r['eb_ll'], r['eb_lv'],
            title='EFGP collapsed loss surface',
            min_label=f"EFGP min: ℓ={math.exp(r['eb_ll']):.2f}, "
                      f"σ²={math.exp(r['eb_lv']):.2f}")
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)

        # Row 2: SparseGP -ELBO surface
        ax = axes[2, col]
        L_norm_s = r['L_sp'] - r['L_sp_min']
        _render_landscape_panel(
            ax, LOG_LS, LOG_VAR, L_norm_s,
            r['e_ls'], r['e_var'], r['s_ls'], r['s_var'],
            r['sb_ll'], r['sb_lv'],
            title='SparseGP −ELBO surface',
            min_label=f"−ELBO min: ℓ={math.exp(r['sb_ll']):.2f}, "
                      f"σ²={math.exp(r['sb_lv']):.2f}")
        if col == 0:
            ax.set_ylabel('log σ²', fontsize=10)

        # Row 3: per-iter own-objective curves
        ax = axes[3, col]
        iters_e = np.arange(len(r['e_loss_curve']))
        iters_s = np.arange(len(r['s_loss_curve']))
        ax.plot(iters_e, r['e_loss_curve'], '-o', color='C0',
                markersize=3, linewidth=1.5,
                label='EFGP: collapsed loss along EFGP traj')
        ax2 = ax.twinx()
        ax2.plot(iters_s, r['s_loss_curve'], '-s', color='C3',
                  markersize=3, linewidth=1.5,
                  label='SparseGP: −ELBO along SparseGP traj')
        ax.set_xlabel('EM iter', fontsize=9)
        if col == 0:
            ax.set_ylabel('EFGP collapsed loss', fontsize=9, color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='C3')
        if col == 2:
            ax2.set_ylabel('SparseGP −ELBO', fontsize=9, color='C3')
        ax.set_title('per-iter own-objective loss', fontsize=10)
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc='best', fontsize=7)
        ax.grid(alpha=0.3)

    em_tag = ('emissions ORACLE C=C_true frozen'
              if oracle_emissions
              else 'emissions SVD init, learned (default)')
    fig.suptitle(f'Hyper trajectories vs each candidate "objective surface"  '
                  f'[{em_tag}]\n'
                  'row 0 (oracle): −log p(x_true; θ)   |   '
                  'row 1: EFGP collapsed loss   |   '
                  'row 2: SparseGP −ELBO   |   '
                  'row 3: per-iter own-loss',
                  fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=120)
    print(f"\n[bench] Saved {out_png}", flush=True)


def render_flow_fields(results, out_png='/tmp/bench_flow_fields.png',
                         oracle_emissions=False):
    """3 rows × 3 cols quiver: truth | EFGP | SparseGP for each benchmark."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nB = len(results)
    fig, axes = plt.subplots(nB, 3, figsize=(13, 4.2 * nB))
    if nB == 1:
        axes = axes[None, :]
    titles_col = ['Ground truth', 'EFGP learned', 'SparseGP learned']
    for row, r in enumerate(results):
        GX, GY = r['GX'], r['GY']
        f_true, f_efgp, f_sp = r['f_true'], r['f_efgp'], r['f_sp']
        xs_np = r['xs_np']
        for col, (F, color, title) in enumerate(zip(
                [f_true, f_efgp, f_sp],
                ['black', 'C0', 'C3'],
                titles_col)):
            ax = axes[row, col]
            mag = np.linalg.norm(F, axis=1, keepdims=True).clip(1e-8)
            Fn = F / mag
            ax.quiver(GX, GY, Fn[:, 0].reshape(GX.shape),
                       Fn[:, 1].reshape(GX.shape), color=color,
                       angles='xy', scale_units='xy', scale=3.5, alpha=0.85,
                       width=0.012, headwidth=3)
            ax.plot(xs_np[:, 0], xs_np[:, 1], lw=0.5, color='red', alpha=0.3)
            extra = ''
            if col == 1:
                extra = (f"   RMSE={r['e_drift_rmse']:.3f} "
                         f"({r['e_drift_rel']:.0%})")
            elif col == 2:
                extra = (f"   RMSE={r['s_drift_rmse']:.3f} "
                         f"({r['s_drift_rel']:.0%})")
            if col == 0:
                ax.set_ylabel(f"{r['name']}\n$x_2$", fontsize=10)
            else:
                ax.set_ylabel('$x_2$', fontsize=8)
            ax.set_xlabel('$x_1$', fontsize=8)
            ax.set_title(title + extra, fontsize=10)
            ax.set_aspect('equal')
    em_tag = ('ORACLE C frozen' if oracle_emissions
              else 'SVD init, learned')
    fig.suptitle(f"Learned drift fields (Procrustes-aligned to truth)  "
                  f"[emissions {em_tag}]\n"
                  f"RMSE shown is field RMSE / zero-baseline RMSE", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=120)
    print(f"[bench] Saved {out_png}", flush=True)


def render_rmse_summary(results, out_png='/tmp/bench_rmse_summary.png',
                          oracle_emissions=False):
    """Bar chart of drift RMSE + latent RMSE per method per benchmark."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nB = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [r['name'] for r in results]
    x = np.arange(nB)
    w = 0.35

    e_drift = [r['e_drift_rmse'] for r in results]
    s_drift = [r['s_drift_rmse'] for r in results]
    e_drift_rel = [r['e_drift_rel'] * 100 for r in results]
    s_drift_rel = [r['s_drift_rel'] * 100 for r in results]
    e_lat = [r['e_lat_rmse'] for r in results]
    s_lat = [r['s_lat_rmse'] for r in results]

    ax = axes[0]
    b1 = ax.bar(x - w/2, e_drift, w, label='EFGP', color='C0')
    b2 = ax.bar(x + w/2, s_drift, w, label='SparseGP', color='C3')
    for b, rel in zip(b1, e_drift_rel):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{rel:.0f}%", ha='center', va='bottom', fontsize=8)
    for b, rel in zip(b2, s_drift_rel):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{rel:.0f}%", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Drift field RMSE')
    ax.set_title('Drift RMSE on Procrustes-aligned grid\n'
                 '(% = drift RMSE / zero-baseline)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - w/2, e_lat, w, label='EFGP', color='C0')
    ax.bar(x + w/2, s_lat, w, label='SparseGP', color='C3')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Latent RMSE (Procrustes-aligned)')
    ax.set_title('Latent state RMSE')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')

    em_tag = ('ORACLE C frozen' if oracle_emissions
              else 'SVD init, learned')
    fig.suptitle(f"Recovery summary [emissions {em_tag}]", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=120)
    print(f"[bench] Saved {out_png}", flush=True)


def main():
    import os
    oracle = os.environ.get('ORACLE_EMISSIONS', '0') == '1'
    sp_lr_tag = os.environ.get('SPARSEGP_LR', '3e-3')
    T_override = os.environ.get('T_OVERRIDE', '')
    if T_override:
        T_new = int(T_override)
        for b in BENCHMARKS:
            scale = T_new / b['T']
            b['T'] = T_new
            # scale t_max proportionally to keep observations spaced the same
            b['t_max'] = b['t_max'] * scale
    suffix = (('_oracle_em' if oracle else '')
              + (f'_splr{sp_lr_tag}' if sp_lr_tag != '3e-3' else '')
              + (f'_T{T_override}' if T_override else ''))
    out_png = f'/tmp/bench_two_obj{suffix}.png'
    print(f"[bench] Two-objective comparison: EFGP vs SparseGP", flush=True)
    print(f"[bench] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), n_em={N_EM}", flush=True)
    print(f"[bench] emissions: "
          f"{'ORACLE C=C_true frozen' if oracle else 'SVD init, learned (default)'}",
          flush=True)
    results = [run_one_benchmark(cfg, oracle_emissions=oracle)
                for cfg in BENCHMARKS]
    render(results, out_png=out_png, oracle_emissions=oracle)
    flow_png = out_png.replace('bench_two_obj', 'bench_flow_fields')
    rmse_png = out_png.replace('bench_two_obj', 'bench_rmse_summary')
    render_flow_fields(results, out_png=flow_png, oracle_emissions=oracle)
    render_rmse_summary(results, out_png=rmse_png, oracle_emissions=oracle)

    # Summary table
    print(f"\n[bench] Summary (RMSE = Procrustes-aligned vs ground-truth):",
          flush=True)
    print(f"  {'benchmark':25s}  "
          f"{'EFGP wall':>9s}  {'SP wall':>9s}  {'speedup':>8s}  "
          f"{'EFGP lat':>9s}  {'SP lat':>9s}  "
          f"{'EFGP drift':>11s}  {'SP drift':>11s}", flush=True)
    for r in results:
        e_w = r['e_wall']; s_w = r['s_wall']
        e_lr = r['e_lat_rmse']; s_lr = r['s_lat_rmse']
        e_dr = r['e_drift_rmse']; s_dr = r['s_drift_rmse']
        e_drel = r['e_drift_rel']; s_drel = r['s_drift_rel']
        print(f"  {r['name']:25s}  {e_w:>8.1f}s  {s_w:>8.1f}s  "
              f"{s_w/e_w:>7.2f}x  {e_lr:>9.4f}  {s_lr:>9.4f}  "
              f"{e_dr:>5.3f}({e_drel:>3.0%})  {s_dr:>5.3f}({s_drel:>3.0%})",
              flush=True)


if __name__ == '__main__':
    main()
