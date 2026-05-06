"""2-well drift benchmark: EFGP-MC vs EFGP-GMIX at T=20k.

Drift: f(x, y) = (x - x^3, -y)
  * Double-well in x with minima at ±1 (barrier height V(0)-V(±1) = 1/4)
  * Restoring in y
SDE noise σ=0.4 → escape time ~ exp(2·0.25/0.16) ≈ 22 time units; with
T·Δt = 1000, expect ~40 well-switches → many transitions between wells,
nonlinear drift learning is harder than the GP-sampled case.

Emissions FIXED at truth (Gaussian, 8d obs, R=0.05).  Both methods
learn (ℓ, σ²) from a deliberately-wrong init.
"""
from __future__ import annotations

import sys, os, json, time
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

D = 2
SIGMA = 0.4
OUT_DIR = Path(__file__).resolve().parent / "_bench_mc_vs_gmix_out"
OUT_DIR.mkdir(exist_ok=True)


def two_well_drift_jax(x, t):
    return jnp.array([x[0] - x[0] ** 3, -x[1]])


def two_well_drift_np(x_batch):
    x = np.asarray(x_batch)
    return np.column_stack([x[:, 0] - x[:, 0] ** 3, -x[:, 1]])


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _fit(method, lik, t_grid, output_params, init_params, ls_init,
          n_em, n_estep, S_marginal, ls_pin):
    rho = jnp.linspace(0.05, 0.4, n_em)
    t0 = time.time()
    mp, _, op_out, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=output_params, init_params=init_params, latent_dim=D,
        lengthscale=ls_init, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-2,
        estep_method=method, S_marginal=S_marginal,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=5, mstep_lr=0.05,
        n_hutchinson_mstep=16, kernel_warmup_iters=3,
        pin_grid=True, pin_grid_lengthscale=ls_pin,
        verbose=False,
    )
    return mp, hist, time.time() - t0


def evaluate_drift(label, mp, hist, xs, t_grid, ls_pin, method,
                    gmix_fine_N=None, gmix_stencil_r=None):
    ms = np.asarray(mp['m'][0])
    Ss = jnp.asarray(np.asarray(mp['S'][0]))
    SSs = jnp.asarray(np.asarray(mp['SS'][0]))
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_pin, hist.variance[-1], X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(hist.lengthscale[-1]),
                                jnp.log(hist.variance[-1]),
                                grid.xis_flat, grid.h_per_dim[0],
                                D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws_final)
    if method == 'mc':
        mu_r, _, _ = jpd.compute_mu_r_jax(
            jnp.asarray(ms), Ss, SSs, del_t, grid, jr.PRNGKey(99),
            sigma_drift_sq=SIGMA ** 2, S_marginal=2, D_lat=D, D_out=D)
    else:
        mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
            jnp.asarray(ms), Ss, SSs, del_t, grid,
            sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
            fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r))

    Ef_traj, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(xs, dtype=jnp.float32),
        D_lat=D, D_out=D)
    f_inf_traj = np.asarray(Ef_traj)
    f_true_traj = two_well_drift_np(xs)
    err_traj = np.linalg.norm(f_inf_traj - f_true_traj, axis=1)
    drift_rmse_traj = float(np.sqrt(np.mean(err_traj ** 2)))

    grid_lo = np.asarray(xs).min(0) - 0.3
    grid_hi = np.asarray(xs).max(0) + 0.3
    xs_g = np.linspace(grid_lo[0], grid_hi[0], 16)
    ys_g = np.linspace(grid_lo[1], grid_hi[1], 16)
    GX, GY = np.meshgrid(xs_g, ys_g, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true_grid = two_well_drift_np(grid_pts)
    Ef_grid, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    drift_rmse_grid = float(np.sqrt(np.mean(
        (np.asarray(Ef_grid) - f_true_grid) ** 2)))
    drift_rmse_zero = float(np.sqrt(np.mean(f_true_grid ** 2)))
    return dict(
        drift_rmse_traj=drift_rmse_traj,
        drift_rmse_grid=drift_rmse_grid,
        drift_rmse_zero=drift_rmse_zero,
        err_traj=err_traj,
        Ef_grid=np.asarray(Ef_grid),
        f_true_grid=f_true_grid,
        grid_pts=grid_pts,
    )


def main(T=20000, n_em=20, n_estep=10):
    print(f"\n=== 2-well drift, T={T} ===")
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.0, 0.0]),
                       f=two_well_drift_jax, t_max=t_max,
                       n_timesteps=T, sigma=sigma_fn)
    xs_np = np.asarray(xs)
    n_switches = int(np.sum(np.diff(np.sign(xs_np[:, 0])) != 0))
    print(f"  T={T}, t_max={t_max}, well switches in x: {n_switches}")
    print(f"  xs range: x ∈ [{xs_np[:,0].min():.2f}, {xs_np[:,0].max():.2f}], "
          f"y ∈ [{xs_np[:,1].min():.2f}, {xs_np[:,1].max():.2f}]")

    N_obs = 8
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N_obs, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                  dict(mu0=jnp.array([1.0, 0.0]),
                                       V0=jnp.eye(D) * 0.3))
    t_grid = jnp.linspace(0., t_max, T)

    ls_init = 0.3                # half of the natural well separation (~1)
    ls_pin = 0.5

    print(f"  init ℓ={ls_init} (truth not GP, but ~1 in scale);  "
          f"σ²_init=1.0;  emissions FIXED at truth.")

    print("  fitting EFGP-SING (MC, S_marginal=2)...")
    mp_mc, hist_mc, t_mc = _fit('mc', lik, t_grid, out_true, ip,
                                  ls_init, n_em, n_estep, S_marginal=2,
                                  ls_pin=ls_pin)
    print(f"    wall {t_mc:.1f}s  ℓ={hist_mc.lengthscale[-1]:.3f}  "
          f"σ²={hist_mc.variance[-1]:.3f}")

    print("  fitting EFGP-SING (GMIX, closed-form spreader)...")
    mp_gx, hist_gx, t_gx = _fit('gmix', lik, t_grid, out_true, ip,
                                  ls_init, n_em, n_estep, S_marginal=2,
                                  ls_pin=ls_pin)
    print(f"    wall {t_gx:.1f}s  ℓ={hist_gx.lengthscale[-1]:.3f}  "
          f"σ²={hist_gx.variance[-1]:.3f}")

    from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    g_eval = jp.spectral_grid_se(ls_pin, 1.0, X_template, eps=1e-2)
    h_spec0 = float(g_eval.h_per_dim[0])
    sigma0_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(ip['V0'][0]))))
    fine_N = pick_grid_size(h_spec=h_spec0, m_extent=6.0, sigma_max=sigma0_max)
    h_g = 1.0 / (fine_N * h_spec0)
    s_r = stencil_radius_for(jnp.asarray(ip['V0'][0])[None], h_g, n_sigma=1.5)

    diag_mc = evaluate_drift('mc', mp_mc, hist_mc, xs, t_grid, ls_pin, 'mc')
    diag_gx = evaluate_drift('gmix', mp_gx, hist_gx, xs, t_grid, ls_pin,
                               'gmix', gmix_fine_N=fine_N, gmix_stencil_r=s_r)

    print(f"\n  EFGP-MC    wall {t_mc:.1f}s  drift_rmse_traj {diag_mc['drift_rmse_traj']:.4f}  "
          f"drift_rmse_grid {diag_mc['drift_rmse_grid']:.4f}")
    print(f"  EFGP-GMIX  wall {t_gx:.1f}s  drift_rmse_traj {diag_gx['drift_rmse_traj']:.4f}  "
          f"drift_rmse_grid {diag_gx['drift_rmse_grid']:.4f}")
    print(f"  zero-drift baseline:  {diag_gx['drift_rmse_zero']:.4f}")

    # Plots
    iters = np.arange(1, n_em + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    axes[0].plot(iters, hist_mc.lengthscale, 'o-', label='MC',
                  color='tab:blue')
    axes[0].plot(iters, hist_gx.lengthscale, 's-', label='GMIX',
                  color='tab:orange')
    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel(r'$\ell$')
    axes[0].set_title(f'Lengthscale path  (T={T}, 2-well)')
    axes[0].legend()
    axes[1].plot(iters, hist_mc.variance, 'o-', label='MC', color='tab:blue')
    axes[1].plot(iters, hist_gx.variance, 's-', label='GMIX',
                  color='tab:orange')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel(r'$\sigma^2$ (log)')
    axes[1].set_title('Variance path')
    axes[1].legend()
    times = np.linspace(0, t_max, T)
    axes[2].plot(times, diag_mc['err_traj'], color='tab:blue', alpha=0.7,
                  label=f'MC RMSE {diag_mc["drift_rmse_traj"]:.3f}')
    axes[2].plot(times, diag_gx['err_traj'], color='tab:orange', alpha=0.7,
                  label=f'GMIX RMSE {diag_gx["drift_rmse_traj"]:.3f}')
    axes[2].set_xlabel('time'); axes[2].set_ylabel('|f_inferred − f_true|')
    axes[2].set_title('Drift error along true trajectory')
    axes[2].legend()
    plt.tight_layout()
    out = OUT_DIR / f'2well_compare_T{T}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # Quiver
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharex=True, sharey=True)
    pts = diag_gx['grid_pts']
    for ax, (lab, f_field) in zip(axes, [
            ('truth', diag_gx['f_true_grid']),
            ('MC',    diag_mc['Ef_grid']),
            ('GMIX',  diag_gx['Ef_grid'])]):
        ax.quiver(pts[:, 0], pts[:, 1], f_field[:, 0], f_field[:, 1],
                  scale=20, alpha=0.7)
        ax.plot(xs_np[:, 0], xs_np[:, 1], color='red', lw=0.4, alpha=0.25)
        ax.scatter([-1, 1], [0, 0], marker='o', s=80, color='green',
                    edgecolor='k', zorder=5)
        ax.set_title(lab); ax.set_aspect('equal')
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    plt.tight_layout()
    out = OUT_DIR / f'2well_quiver_T{T}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    summary = dict(
        T=T, drift='2-well', n_switches=n_switches,
        wall_mc=t_mc, wall_gmix=t_gx,
        ls_mc=list(map(float, hist_mc.lengthscale)),
        var_mc=list(map(float, hist_mc.variance)),
        ls_gmix=list(map(float, hist_gx.lengthscale)),
        var_gmix=list(map(float, hist_gx.variance)),
        drift_rmse_traj_mc=diag_mc['drift_rmse_traj'],
        drift_rmse_traj_gmix=diag_gx['drift_rmse_traj'],
        drift_rmse_grid_mc=diag_mc['drift_rmse_grid'],
        drift_rmse_grid_gmix=diag_gx['drift_rmse_grid'],
        drift_rmse_zero=diag_gx['drift_rmse_zero'],
    )
    with open(OUT_DIR / f'summary_2well_T{T}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main(T=20000)
