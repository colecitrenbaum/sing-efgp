"""EFGP (gmix) vs SparseGP, K=4 trials × T=500 from a known-hyper GP drift.

Setup
-----
* Drift: one fixed sample from the SE-kernel GP prior at known
  ``ℓ_true=0.6``, ``σ²_true=1.0``.
* K=4 trials, T=500 each, diverse initial conditions ``x0 ~ U([-2, 2]²)``.
* Gaussian emissions FIXED at truth (clean comparison: no Procrustes
  drift between methods, latent basis stays in truth basis).
* Both methods init at deliberately-wrong ``ℓ_init=1.5``.
* SparseGP uses 25 inducing points (5×5 grid in ``[-2.5, 2.5]²``).
* EFGP uses the default closed-form gmix spreader.

Outputs
-------
1. Learning-curve plot (PNG): per-EM-iter ℓ, σ², and latent RMSE for
   both methods, with truth horizontal lines.
2. Wall-clock timings.
3. Drift-RMSE along each trial's true latent path (per-method, per-trial),
   plus pooled mean.

Usage:  python demos/bench_efgp_vs_sparsegp_4trials.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.initialization import initialize_zs
from sing.sing import fit_variational_em

# --------- problem constants ---------
D = 2
SIGMA = 0.4
LS_TRUE = 0.6
VAR_TRUE = 1.0
LS_INIT = 1.5            # deliberately ~2.5x too large
VAR_INIT = 1.0

K = 4
T = 500
T_MAX = 8.0              # per trial

N_OBS = 6
OUT_DIR = Path(__file__).resolve().parent / "_bench_efgp_vs_sparsegp_4trials_out"
OUT_DIR.mkdir(exist_ok=True)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


# ---------------------------------------------------------------------------
# GP drift sampler — same machinery as the recovery test
# ---------------------------------------------------------------------------
def gp_drift_factory(ls, var, key, extent=4.0, eps_grid=1e-2):
    X_template = jnp.linspace(-extent, extent, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(ls, var, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
        / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps                  # (D, M)

    def drift(x, t):
        x_b = x[None, :]
        return jnp.stack([
            jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real[0]
            for r in range(D)
        ])

    def drift_grid(X_eval):
        return jnp.stack([
            jp.nufft2(X_eval, fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real
            for r in range(D)
        ], axis=-1)

    return drift, drift_grid


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------
def simulate_data(seed=42):
    drift_fn, drift_grid_fn = gp_drift_factory(
        LS_TRUE, VAR_TRUE, jr.PRNGKey(seed))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)

    rng = np.random.default_rng(seed + 1)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, size=(K, D)).astype(np.float32))

    xs_list = []
    for k in range(K):
        xs_k = simulate_sde(jr.PRNGKey(seed + 100 + k), x0=x0_K[k],
                             f=drift_fn, t_max=T_MAX, n_timesteps=T,
                             sigma=sigma_fn)
        xs_list.append(jnp.clip(xs_k, -3.5, 3.5))
    xs_K = jnp.stack(xs_list, axis=0)                # (K, T, D)

    C_true = rng.standard_normal((N_OBS, D)).astype(np.float32) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys_list = [simulate_gaussian_obs(jr.PRNGKey(seed + 200 + k),
                                       xs_K[k], out_true)
               for k in range(K)]
    ys_K = jnp.stack(ys_list, axis=0)                # (K, T, N_OBS)

    return drift_fn, drift_grid_fn, xs_K, ys_K, x0_K, out_true


# ---------------------------------------------------------------------------
# EFGP fit
# ---------------------------------------------------------------------------
def fit_efgp(ys_K, xs_K, x0_K, out_true, t_grid, n_em, n_estep):
    K_, T_, N_ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K_, 1, 1)))

    # Settings inherited from ``bench_gpdrift_x64.py`` (most recent
    # EFGP/SparseGP head-to-head): rho ramp 0.05->0.7, mstep_lr=0.01,
    # n_mstep_iters=4, n_hutchinson=4, kernel_warmup=8, eps_grid=1e-3.
    rho_efgp = jnp.linspace(0.05, 0.7, n_em)
    print(f"  [EFGP] init ℓ={LS_INIT}, σ²={VAR_INIT}; ℓ_true={LS_TRUE}, "
          f"σ²_true={VAR_TRUE}")
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=out_true, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
        estep_method='gmix', S_marginal=2,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho_efgp,
        # Emissions are at truth — keep them fixed so the latent basis
        # stays in truth coordinates and drift comparisons are clean.
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        pin_grid=True, pin_grid_lengthscale=LS_TRUE * 0.75,
        verbose=False,
        true_xs=np.asarray(xs_K),       # for per-iter latent RMSE
    )
    wall = time.time() - t0
    return mp, hist, wall


def _eval_efgp_drift_at(mp, hist, t_grid, X_eval, trial_mask):
    """Build q(f) at the FINAL (ℓ, σ²) on the same pinned grid the EM used,
    then evaluate the posterior drift mean at ``X_eval`` (M_pts, D)."""
    ls, var = float(hist.lengthscale[-1]), float(hist.variance[-1])
    ms = jnp.asarray(np.asarray(mp['m']))
    Ss = jnp.asarray(np.asarray(mp['S']))
    SSs = jnp.asarray(np.asarray(mp['SS']))
    del_t = t_grid[1:] - t_grid[:-1]

    X_template = jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D))
    pin_ls = LS_TRUE * 0.75
    grid = jp.spectral_grid_se(pin_ls, var, X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(ls), jnp.log(var),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)

    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval, dtype=jnp.float32)[None],
        D_lat=D, D_out=D)
    return np.asarray(Ef[0])                         # (M_pts, D)


# ---------------------------------------------------------------------------
# SparseGP fit (with per-iter hyper history hook)
# ---------------------------------------------------------------------------
def _data_aware_zs(num_per_dim, xs_np, pad=0.4):
    """Per-dim inducing grid in ``[min(x_d)-pad, max(x_d)+pad]``, matched
    to the actual trajectory bbox.  Same heuristic as
    ``bench_gpdrift_x64._data_aware_zs``."""
    lo = xs_np.min(axis=(0, 1)) - pad
    hi = xs_np.max(axis=(0, 1)) + pad
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim)
                for d in range(xs_np.shape[-1])]
    return jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'),
                      axis=-1).reshape(-1, xs_np.shape[-1])


def fit_sparsegp(ys_K, xs_K, x0_K, out_true, t_grid, n_em, n_estep,
                  num_per_dim=5):
    K_, T_, N_ = ys_K.shape
    trial_mask = jnp.ones((K_, T_), dtype=bool)
    lik = GLik(ys_K, trial_mask)
    ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D), (K_, 1, 1)))

    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    # Data-aware inducing grid (matched to trial bbox).
    zs = _data_aware_zs(num_per_dim=num_per_dim, xs_np=np.asarray(xs_K))
    print(f"  [SparseGP] {zs.shape[0]} inducing points "
          f"(data-aware {num_per_dim}x{num_per_dim} grid; bbox-matched + 0.4 pad)")
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    # SparseGP's RBF stores ``output_scale`` such that var = output_scale**2,
    # so initialise with ``sqrt(VAR_INIT)``.
    sparse_drift_params = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                                output_scale=jnp.asarray(math.sqrt(VAR_INIT)))

    # Settings from bench_gpdrift_x64: rho ramp 0.05->0.7, lr=0.01, n_iters_m=4.
    sing_rho = jnp.linspace(0.05, 0.7, n_em)
    sing_lr = jnp.full((n_em,), 0.01)

    drift_history = []
    print(f"  [SparseGP] init ℓ={LS_INIT}, σ²={VAR_INIT}")
    t0 = time.time()
    mp, _, _, sp_drift_params, _, op_sp, _, elbos = fit_variational_em(
        key=jr.PRNGKey(33),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid,
        drift_params=sparse_drift_params,
        init_params=ip, output_params=dict(out_true),
        sigma=SIGMA,
        rho_sched=sing_rho,
        n_iters=n_em, n_iters_e=n_estep, n_iters_m=4,
        perform_m_step=True, learn_output_params=False,    # emissions fixed
        learning_rate=sing_lr,
        print_interval=999,
        drift_params_history=drift_history,
    )
    wall = time.time() - t0

    # Extract ℓ, σ² histories — output_scale is stored as sqrt(σ²),
    # so square to convert back to σ².
    ls_hist = [float(jnp.exp(jnp.mean(jnp.log(d['length_scales']))))
               for d in drift_history]
    var_hist = [float(d['output_scale']) ** 2 for d in drift_history]

    return mp, sp_drift_params, sparse_drift, ls_hist, var_hist, elbos, wall


def _eval_sparsegp_drift_at(sparse_drift, sp_drift_params, mp, t_grid,
                             trial_mask, X_eval):
    gp_post = sparse_drift.update_dynamics_params(
        jr.PRNGKey(0), t_grid, mp, trial_mask,
        sp_drift_params,
        jnp.zeros((trial_mask.shape[0], trial_mask.shape[1], 1)),
        jnp.zeros((D, 1)), SIGMA)
    return np.asarray(sparse_drift.get_posterior_f_mean(
        gp_post, sp_drift_params, jnp.asarray(X_eval)))


# ---------------------------------------------------------------------------
# Oracle GT log-marginal landscape — pseudo-velocity GP NLL on the TRUE
# latent paths.  Multi-trial: stack (x, v) pairs across trials.  Same
# math as ``bench_gpdrift_x64.gt_landscape`` / ``bench_duffing_lsinit_x64``,
# extended to K trials.
# ---------------------------------------------------------------------------
def gt_log_marginal_landscape(xs_K, sigma_drift, t_grid,
                                LOG_LS, LOG_VAR):
    """Negative log marginal likelihood (NLL) of the pseudo-velocity model

        v_t | x_t  ~  N(f(x_t),  σ²/dt I)
        f          ~  GP(0,  σ²_f K_RBF(·; ℓ))

    on the TRUE multi-trial latent paths.  Returns ``L_grid`` of shape
    ``(len(LOG_LS), len(LOG_VAR))``.  Argmin = oracle MLE.
    """
    import scipy.linalg
    xs_np = np.asarray(xs_K)                        # (K, T, D)
    K_, T_, D_ = xs_np.shape
    dt = float(np.asarray(t_grid[1] - t_grid[0]))

    # Pool sources/velocities across trials.
    inputs = xs_np[:, :-1, :].reshape(-1, D_)        # (K*(T-1), D)
    velocities = ((xs_np[:, 1:, :] - xs_np[:, :-1, :]) / dt
                  ).reshape(-1, D_)                  # (K*(T-1), D)
    n_obs = inputs.shape[0]

    diffs = inputs[:, None, :] - inputs[None, :, :]
    sq_dists = (diffs ** 2).sum(-1)
    noise_var = sigma_drift ** 2 / dt
    eye_n = np.eye(n_obs)
    print(f"  GT landscape: {len(LOG_LS)}x{len(LOG_VAR)} sweep over "
          f"{n_obs} pooled sources (Cholesky on {n_obs}x{n_obs})...",
          flush=True)
    L_grid = np.zeros((len(LOG_LS), len(LOG_VAR)))
    for i, ll in enumerate(LOG_LS):
        ell = math.exp(float(ll))
        K_unsc = np.exp(-0.5 * sq_dists / (ell ** 2))
        for j, lv in enumerate(LOG_VAR):
            var_ = math.exp(float(lv))
            A_ = var_ * K_unsc + noise_var * eye_n
            try:
                L_chol = np.linalg.cholesky(A_)
            except np.linalg.LinAlgError:
                L_grid[i, j] = np.nan
                continue
            logdet = 2.0 * np.log(np.diag(L_chol)).sum()
            quad = 0.0
            for d in range(D_):
                z = scipy.linalg.solve_triangular(
                    L_chol, velocities[:, d], lower=True)
                quad += float(z @ z)
            L_grid[i, j] = 0.5 * quad + 0.5 * D_ * logdet
    return L_grid


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(n_em=50, n_estep=10, sparsegp_num_per_dim=5,
          log_ls_range=(-2.0, 1.5), log_var_range=(-2.5, 2.0), n_landscape=21):
    print(f"\n=== EFGP-gmix vs SparseGP-25  (K={K}, T={T}, n_em={n_em}) ===")
    drift_fn, drift_grid_fn, xs_K, ys_K, x0_K, out_true = simulate_data()
    t_grid = jnp.linspace(0., T_MAX, T)
    trial_mask = jnp.ones((K, T), dtype=bool)

    # --- EFGP ---
    print(f"\n  fitting EFGP-SING (gmix, multi-trial)...")
    mp_efgp, hist_efgp, wall_efgp = fit_efgp(
        ys_K, xs_K, x0_K, out_true, t_grid, n_em, n_estep)
    ls_hist_efgp = list(hist_efgp.lengthscale)
    var_hist_efgp = list(hist_efgp.variance)
    lat_hist_efgp = list(hist_efgp.latent_rmse)
    print(f"    wall {wall_efgp:.1f}s  "
          f"final ℓ={ls_hist_efgp[-1]:.3f}  σ²={var_hist_efgp[-1]:.3f}")

    # --- SparseGP ---
    print(f"\n  fitting SING-SparseGP (25 inducing, multi-trial)...")
    mp_sp, sp_dp, sp_drift_obj, ls_hist_sp, var_hist_sp, elbos_sp, wall_sp = \
        fit_sparsegp(ys_K, xs_K, x0_K, out_true, t_grid, n_em, n_estep,
                      num_per_dim=sparsegp_num_per_dim)
    print(f"    wall {wall_sp:.1f}s  "
          f"final ℓ={ls_hist_sp[-1]:.3f}  σ²={var_hist_sp[-1]:.3f}")

    # --- Drift accuracy along trajectories ---
    print(f"\n  evaluating drift accuracy along true latent paths...")
    xs_flat = np.asarray(xs_K).reshape(-1, D)        # (K*T, D)
    f_true_flat = np.asarray(drift_grid_fn(jnp.asarray(xs_flat,
                                                          dtype=jnp.float32)))
    f_efgp = _eval_efgp_drift_at(mp_efgp, hist_efgp, t_grid,
                                   xs_flat, trial_mask)
    f_sp = _eval_sparsegp_drift_at(sp_drift_obj, sp_dp, mp_sp, t_grid,
                                    trial_mask, xs_flat)

    f_efgp_per = f_efgp.reshape(K, T, D)
    f_sp_per = f_sp.reshape(K, T, D)
    f_true_per = f_true_flat.reshape(K, T, D)
    drift_rmse_efgp_per = np.sqrt(np.mean((f_efgp_per - f_true_per) ** 2,
                                            axis=(1, 2)))
    drift_rmse_sp_per = np.sqrt(np.mean((f_sp_per - f_true_per) ** 2,
                                          axis=(1, 2)))
    drift_scale = float(np.sqrt(np.mean(f_true_flat ** 2)))
    drift_rmse_efgp = float(np.sqrt(np.mean((f_efgp - f_true_flat) ** 2)))
    drift_rmse_sp = float(np.sqrt(np.mean((f_sp - f_true_flat) ** 2)))

    print(f"\n  ===== drift RMSE along trajectories  (drift scale = {drift_scale:.3f}) =====")
    print(f"  EFGP-gmix      pooled = {drift_rmse_efgp:.4f}  "
          f"({100*drift_rmse_efgp/drift_scale:.1f}% of scale)")
    print(f"                 per-trial = {np.array2string(drift_rmse_efgp_per, precision=4)}")
    print(f"  SparseGP-25    pooled = {drift_rmse_sp:.4f}  "
          f"({100*drift_rmse_sp/drift_scale:.1f}% of scale)")
    print(f"                 per-trial = {np.array2string(drift_rmse_sp_per, precision=4)}")

    # --- GT log-marginal landscape + oracle MLE ---
    LOG_LS = np.linspace(log_ls_range[0], log_ls_range[1], n_landscape)
    LOG_VAR = np.linspace(log_var_range[0], log_var_range[1], n_landscape)
    L_gt = gt_log_marginal_landscape(xs_K, SIGMA, t_grid, LOG_LS, LOG_VAR)
    idx = np.unravel_index(np.nanargmin(L_gt), L_gt.shape)
    ls_mle = float(math.exp(LOG_LS[idx[0]]))
    var_mle = float(math.exp(LOG_VAR[idx[1]]))
    print(f"\n  oracle GT MLE: ℓ={ls_mle:.3f}  σ²={var_mle:.3f}  "
          f"(truth ℓ={LS_TRUE} σ²={VAR_TRUE})")
    print(f"   EFGP final  : ℓ={ls_hist_efgp[-1]:.3f}  σ²={var_hist_efgp[-1]:.3f}")
    print(f"   SparseGP    : ℓ={ls_hist_sp[-1]:.3f}  σ²={var_hist_sp[-1]:.3f}")

    # --- Save ---
    np.savez(OUT_DIR / "bench.npz",
              ls_hist_efgp=ls_hist_efgp, ls_hist_sp=ls_hist_sp,
              var_hist_efgp=var_hist_efgp, var_hist_sp=var_hist_sp,
              lat_hist_efgp=lat_hist_efgp,
              elbos_sp=np.asarray(elbos_sp),
              drift_rmse_efgp_per=drift_rmse_efgp_per,
              drift_rmse_sp_per=drift_rmse_sp_per,
              drift_rmse_efgp=drift_rmse_efgp,
              drift_rmse_sp=drift_rmse_sp,
              drift_scale=drift_scale,
              wall_efgp=wall_efgp, wall_sp=wall_sp,
              ls_true=LS_TRUE, var_true=VAR_TRUE,
              ls_mle=ls_mle, var_mle=var_mle,
              LOG_LS=LOG_LS, LOG_VAR=LOG_VAR, L_gt=L_gt,
              K=K, T=T)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Prepend init point so each trajectory starts at (0, init_value).
    iters_efgp = np.arange(0, len(ls_hist_efgp) + 1)
    iters_sp = np.arange(0, len(ls_hist_sp) + 1)
    ls_traj_efgp = [LS_INIT] + list(ls_hist_efgp)
    var_traj_efgp = [VAR_INIT] + list(var_hist_efgp)
    ls_traj_sp = [LS_INIT] + list(ls_hist_sp)
    var_traj_sp = [VAR_INIT] + list(var_hist_sp)

    ax = axes[0, 0]
    ax.plot(iters_efgp, ls_traj_efgp, '-o', label='EFGP-gmix', color='C0',
             markersize=4)
    ax.plot(iters_sp, ls_traj_sp, '-s', label='SparseGP-25', color='C1',
             markersize=4)
    ax.axhline(LS_TRUE, color='k', linestyle='--', alpha=0.5,
                label=f'truth ℓ={LS_TRUE}')
    ax.axhline(ls_mle, color='r', linestyle=':', alpha=0.7,
                label=f'oracle MLE ℓ={ls_mle:.3f}')
    ax.set_xlabel('EM iter')
    ax.set_ylabel('ℓ')
    ax.set_title('lengthscale trajectory')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters_efgp, var_traj_efgp, '-o', label='EFGP-gmix', color='C0',
             markersize=4)
    ax.plot(iters_sp, var_traj_sp, '-s', label='SparseGP-25', color='C1',
             markersize=4)
    ax.axhline(VAR_TRUE, color='k', linestyle='--', alpha=0.5,
                label=f'truth σ²={VAR_TRUE}')
    ax.axhline(var_mle, color='r', linestyle=':', alpha=0.7,
                label=f'oracle MLE σ²={var_mle:.3f}')
    ax.set_xlabel('EM iter')
    ax.set_ylabel('σ²')
    ax.set_yscale('log')
    ax.set_title('output-scale trajectory  (log y)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1, 0]
    # GT log-marginal landscape (oracle), with FULL hyper trajectories
    # for both methods.  Normalize NLL - min(NLL) for a positive "regret".
    L_norm = L_gt - np.nanmin(L_gt)
    finite = L_norm[np.isfinite(L_norm) & (L_norm > 0)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        levels = list(np.logspace(np.log10(max(lo, hi / 1000)),
                                    np.log10(hi), 10))
    else:
        levels = [1, 10, 100]
    cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T, levels=[0] + levels,
                       cmap='viridis_r', extend='max')
    cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T, levels=levels,
                     colors='k', linewidths=0.4)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

    # Full trajectories.  Prepend the (LS_INIT, VAR_INIT) start so the path
    # shows where the optimiser started.
    log_ls_traj_efgp = np.log([LS_INIT] + list(ls_hist_efgp))
    log_var_traj_efgp = np.log([VAR_INIT] + list(var_hist_efgp))
    log_ls_traj_sp = np.log([LS_INIT] + list(ls_hist_sp))
    log_var_traj_sp = np.log([VAR_INIT] + list(var_hist_sp))
    ax.plot(log_ls_traj_efgp, log_var_traj_efgp, '-', color='C0',
             linewidth=1.5, alpha=0.7)
    ax.plot(log_ls_traj_efgp, log_var_traj_efgp, 'o', color='C0',
             markersize=4, markeredgecolor='k', markeredgewidth=0.3,
             alpha=0.85, label=f'EFGP path  ({len(ls_hist_efgp)} EM iters)')
    ax.plot(log_ls_traj_sp, log_var_traj_sp, '-', color='C1',
             linewidth=1.5, alpha=0.7)
    ax.plot(log_ls_traj_sp, log_var_traj_sp, 's', color='C1',
             markersize=4, markeredgecolor='k', markeredgewidth=0.3,
             alpha=0.85, label=f'SparseGP path  ({len(ls_hist_sp)} EM iters)')
    # Larger end-point markers
    ax.plot([log_ls_traj_efgp[-1]], [log_var_traj_efgp[-1]],
             'o', color='C0', markeredgecolor='k', markersize=11,
             label=f'EFGP final (ℓ={ls_hist_efgp[-1]:.2f}, σ²={var_hist_efgp[-1]:.2f})')
    ax.plot([log_ls_traj_sp[-1]], [log_var_traj_sp[-1]],
             's', color='C1', markeredgecolor='k', markersize=11,
             label=f'SparseGP final (ℓ={ls_hist_sp[-1]:.2f}, σ²={var_hist_sp[-1]:.2f})')
    # Init / truth / MLE
    ax.plot([math.log(LS_INIT)], [math.log(VAR_INIT)], 'D',
             color='lightgray', markeredgecolor='k', markersize=10,
             label=f'init (ℓ={LS_INIT}, σ²={VAR_INIT})')
    ax.plot([math.log(LS_TRUE)], [math.log(VAR_TRUE)], 'w*',
             markeredgecolor='k', markersize=15,
             label=f'truth (ℓ={LS_TRUE}, σ²={VAR_TRUE})')
    ax.plot([math.log(ls_mle)], [math.log(var_mle)], 'rP',
             markeredgecolor='k', markersize=12,
             label=f'oracle MLE (ℓ={ls_mle:.2f}, σ²={var_mle:.2f})')
    ax.set_xlabel(r'$\log\,\ell$')
    ax.set_ylabel(r'$\log\,\sigma^2$')
    ax.set_title('GT log-marginal landscape  (oracle on true xs)\nfull (ℓ, σ²) trajectories')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    width = 0.4
    idx = np.arange(K)
    ax.bar(idx - width/2, drift_rmse_efgp_per, width=width,
            label='EFGP-gmix', color='C0')
    ax.bar(idx + width/2, drift_rmse_sp_per, width=width,
            label='SparseGP-25', color='C1')
    ax.axhline(drift_scale, color='k', linestyle=':', alpha=0.4,
                label='drift scale (zero-pred RMSE)')
    ax.set_xlabel('trial')
    ax.set_xticks(idx)
    ax.set_ylabel('drift RMSE along path')
    ax.set_title('drift accuracy along true latent path')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'EFGP-gmix vs SparseGP-25  (K={K} trials × T={T} from GP-drift, '
        f'ℓ_true={LS_TRUE}, σ²_true={VAR_TRUE})\n'
        f'wall: EFGP {wall_efgp:.1f}s   SparseGP {wall_sp:.1f}s',
        fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / 'bench.png'
    fig.savefig(out_png, dpi=130)
    print(f"\n  saved plot:  {out_png}")
    print(f"  saved data:  {OUT_DIR/'bench.npz'}")


if __name__ == "__main__":
    main()
