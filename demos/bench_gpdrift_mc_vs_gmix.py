"""GP-drift recovery: EFGP-MC vs EFGP-GMIX (closed-form spreader).

Both methods solve the SAME EFGP q(f) problem; they differ only in how
the q(x)-expectations are computed:
  * MC:    pseudo-cloud sampling, S_marginal=2 by default
  * GMIX:  closed-form Gaussian-mixture spreader (variant A from
           ``efgp_estep_charfun.tex``)

Emissions are FIXED at the truth (``learn_emissions=False``) to isolate
the q(f) regression block.  Reports:

  * ℓ and σ² learning paths per outer EM iter
  * Drift error along the true latent trajectory at converged hypers
  * Drift RMSE on a grid

Outputs ``demos/_bench_mc_vs_gmix_out/`` with PDF plots + JSON summary.
"""
from __future__ import annotations

import sys, os, json, time, math
from pathlib import Path
from dataclasses import asdict

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
SIGMA = 0.3
ALPHA_RESTORE = 0.5
OUT_DIR = Path(__file__).resolve().parent / "_bench_mc_vs_gmix_out"
OUT_DIR.mkdir(exist_ok=True)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                  * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    keys = jr.split(key, D)
    fk_draw = []
    for k in keys:
        k1, k2 = jr.split(k)
        ee = (jr.normal(k1, (grid.M,))
              + 1j * jr.normal(k2, (grid.M,))).astype(grid.ws.dtype)
        fk_draw.append(grid.ws * ee / math.sqrt(2))
    fk_draw = jnp.stack(fk_draw, axis=0)                     # (D, M)
    OUT = grid.mtot_per_dim
    xcen, h_per_dim = grid.xcen, grid.h_per_dim

    def drift_jax(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*OUT),
                           xcen, h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out) - ALPHA_RESTORE * x

    def drift_np(x_batch):
        xt = jnp.asarray(x_batch, dtype=jnp.float32)
        out = np.zeros_like(np.asarray(x_batch))
        for r in range(D):
            fr = jp.nufft2(xt, fk_draw[r].reshape(*OUT),
                           xcen, h_per_dim, eps=6e-8).real
            out[:, r] = np.asarray(fr) - ALPHA_RESTORE * np.asarray(x_batch)[:, r]
        return out

    return drift_jax, drift_np


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _fit_one(estep_method, ls_true, var_true, lik, t_grid, output_params,
              init_params, ls_init, n_em, n_estep, S_marginal):
    rho = jnp.linspace(0.05, 0.4, n_em)
    t0 = time.time()
    mp, _, op_out, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=output_params, init_params=init_params, latent_dim=D,
        lengthscale=ls_init, variance=1.0, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-2,
        estep_method=estep_method,
        S_marginal=S_marginal,
        n_em_iters=n_em, n_estep_iters=n_estep, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=5, mstep_lr=0.05,
        n_hutchinson_mstep=16, kernel_warmup_iters=3,
        pin_grid=True, pin_grid_lengthscale=ls_true * 0.75,
        verbose=False,
    )
    return mp, op_out, hist, time.time() - t0


def evaluate(label, mp, hist, drift_np, xs, t_grid, ls_true, var_true,
              estep_method, gmix_fine_N=None, gmix_stencil_r=None):
    ms = np.asarray(mp['m'][0])
    Ss = jnp.asarray(np.asarray(mp['S'][0]))
    SSs = jnp.asarray(np.asarray(mp['SS'][0]))
    del_t = t_grid[1:] - t_grid[:-1]

    # Re-build the same pinned-grid as the EM run used
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    pin_ls = ls_true * 0.75
    grid_e = jp.spectral_grid_se(pin_ls, hist.variance[-1], X_template, eps=1e-2)
    ws_final = jpd._ws_real_se(jnp.log(hist.lengthscale[-1]),
                                jnp.log(hist.variance[-1]),
                                grid_e.xis_flat, grid_e.h_per_dim[0],
                                D).astype(grid_e.ws.dtype)
    grid_e = grid_e._replace(ws=ws_final)

    if estep_method == 'mc':
        mu_r, _, _ = jpd.compute_mu_r_jax(
            jnp.asarray(ms), Ss, SSs, del_t, grid_e, jr.PRNGKey(99),
            sigma_drift_sq=SIGMA ** 2, S_marginal=2, D_lat=D, D_out=D)
    else:
        mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
            jnp.asarray(ms), Ss, SSs, del_t, grid_e,
            sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
            fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r))

    # 1) Drift along true trajectory
    Ef_traj, _, _ = jpd.drift_moments_jax(
        mu_r, grid_e, jnp.asarray(xs, dtype=jnp.float32),
        D_lat=D, D_out=D)
    f_inferred_traj = np.asarray(Ef_traj)
    f_true_traj = drift_np(np.asarray(xs))
    err_traj = np.linalg.norm(f_inferred_traj - f_true_traj, axis=1)
    drift_rmse_traj = float(np.sqrt(np.mean(err_traj ** 2)))

    # 2) Drift on a grid
    grid_lo = np.asarray(xs).min(0) - 0.3
    grid_hi = np.asarray(xs).max(0) + 0.3
    xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
    ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
    GX, GY = np.meshgrid(xs_g, ys_g, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true_grid = drift_np(grid_pts)
    Ef_grid, _, _ = jpd.drift_moments_jax(
        mu_r, grid_e, jnp.asarray(grid_pts, dtype=jnp.float32),
        D_lat=D, D_out=D)
    drift_rmse_grid = float(np.sqrt(np.mean(
        (np.asarray(Ef_grid) - f_true_grid) ** 2)))
    drift_rmse_zero = float(np.sqrt(np.mean(f_true_grid ** 2)))

    return dict(
        drift_rmse_traj=drift_rmse_traj,
        drift_rmse_grid=drift_rmse_grid,
        drift_rmse_zero=drift_rmse_zero,
        err_traj=err_traj,
        f_inferred_traj=f_inferred_traj,
        f_true_traj=f_true_traj,
        Ef_grid=np.asarray(Ef_grid),
        f_true_grid=f_true_grid,
        grid_pts=grid_pts,
    )


def run(T, ls_true, var_true, n_em=25, n_estep=10):
    print(f"\n{'='*72}\n  GP-drift recovery: T={T}, ls_true={ls_true}, "
          f"var_true={var_true}\n{'='*72}")
    drift_jax, drift_np = make_gp_drift(ls_true, var_true, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([0.0, 0.0]),
                       f=drift_jax, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    print(f"  xs range: [{float(xs.min()):.2f}, {float(xs.max()):.2f}]")

    # Truth-FIXED emissions
    N_obs = 8
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N_obs, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    ip = jax.tree_util.tree_map(
        lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)

    ls_init = 0.3 if ls_true >= 0.6 else 1.5

    print(f"  init ℓ={ls_init} (truth={ls_true});  "
          f"σ²_init=1.0 (truth={var_true});  emissions FIXED at truth.")

    # ---- Fit MC ----
    print("  fitting EFGP-SING (MC, S_marginal=2)...")
    mp_mc, op_mc, hist_mc, t_mc = _fit_one(
        'mc', ls_true, var_true, lik, t_grid, out_true, ip,
        ls_init, n_em, n_estep, S_marginal=2)

    # ---- Fit GMIX ----
    print("  fitting EFGP-SING (GMIX, closed-form spreader)...")
    mp_gx, op_gx, hist_gx, t_gx = _fit_one(
        'gmix', ls_true, var_true, lik, t_grid, out_true, ip,
        ls_init, n_em, n_estep, S_marginal=2)

    # Recover the gmix knobs from a dummy fit init to evaluate
    from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    pin_ls = ls_true * 0.75
    grid_e = jp.spectral_grid_se(pin_ls, 1.0, X_template, eps=1e-2)
    h_spec0 = float(grid_e.h_per_dim[0])
    sigma0_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(ip['V0'][0]))))
    fine_N = pick_grid_size(h_spec=h_spec0, m_extent=6.0, sigma_max=sigma0_max)
    h_grid_eval = 1.0 / (fine_N * h_spec0)
    stencil_r = stencil_radius_for(jnp.asarray(ip['V0'][0])[None], h_grid_eval)

    # Diagnostics
    diag_mc = evaluate('mc', mp_mc, hist_mc, drift_np, xs, t_grid,
                        ls_true, var_true, 'mc')
    diag_gx = evaluate('gmix', mp_gx, hist_gx, drift_np, xs, t_grid,
                        ls_true, var_true, 'gmix',
                        gmix_fine_N=fine_N, gmix_stencil_r=stencil_r)

    print(f"\n  EFGP-MC    wall {t_mc:.1f}s  ℓ {hist_mc.lengthscale[-1]:.3f}  "
          f"σ² {hist_mc.variance[-1]:.3f}  drift_rmse_traj {diag_mc['drift_rmse_traj']:.4f}")
    print(f"  EFGP-GMIX  wall {t_gx:.1f}s  ℓ {hist_gx.lengthscale[-1]:.3f}  "
          f"σ² {hist_gx.variance[-1]:.3f}  drift_rmse_traj {diag_gx['drift_rmse_traj']:.4f}")
    print(f"  zero-drift baseline (grid):  {diag_gx['drift_rmse_zero']:.4f}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Hyperparameter learning paths
    iters = np.arange(1, n_em + 1)
    axes[0].plot(iters, hist_mc.lengthscale, 'o-', label='MC', color='tab:blue')
    axes[0].plot(iters, hist_gx.lengthscale, 's-', label='GMIX', color='tab:orange')
    axes[0].axhline(ls_true, color='k', ls='--', alpha=0.6, label='truth')
    axes[0].set_xlabel('EM iter'); axes[0].set_ylabel('ℓ')
    axes[0].set_title(f'Lengthscale path  (T={T}, ℓ_true={ls_true})')
    axes[0].legend()

    axes[1].plot(iters, hist_mc.variance, 'o-', label='MC', color='tab:blue')
    axes[1].plot(iters, hist_gx.variance, 's-', label='GMIX', color='tab:orange')
    axes[1].axhline(var_true, color='k', ls='--', alpha=0.6, label='truth')
    axes[1].set_xlabel('EM iter'); axes[1].set_ylabel('σ²')
    axes[1].set_title(f'Variance path  (σ²_true={var_true})')
    axes[1].legend()

    # Drift error along trajectory
    times = np.linspace(0, t_max, T)
    axes[2].plot(times, diag_mc['err_traj'], color='tab:blue', alpha=0.85,
                  label=f'MC  (RMSE {diag_mc["drift_rmse_traj"]:.3f})')
    axes[2].plot(times, diag_gx['err_traj'], color='tab:orange', alpha=0.85,
                  label=f'GMIX  (RMSE {diag_gx["drift_rmse_traj"]:.3f})')
    axes[2].set_xlabel('time'); axes[2].set_ylabel('||f_inferred − f_true||')
    axes[2].set_title('Drift error along true trajectory')
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"compare_T{T}_ls{ls_true}.pdf")
    plt.close(fig)

    # Drift quivers
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    pts = diag_gx['grid_pts']
    for ax, (lab, f_field) in zip(axes, [
            ('truth', diag_gx['f_true_grid']),
            ('MC',    diag_mc['Ef_grid']),
            ('GMIX',  diag_gx['Ef_grid'])]):
        ax.quiver(pts[:, 0], pts[:, 1], f_field[:, 0], f_field[:, 1],
                  scale=20, alpha=0.7)
        ax.plot(np.asarray(xs)[:, 0], np.asarray(xs)[:, 1],
                color='red', lw=0.5, alpha=0.4)
        ax.set_title(lab); ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"quiver_T{T}_ls{ls_true}.pdf")
    plt.close(fig)

    summary = dict(
        T=T, ls_true=ls_true, var_true=var_true,
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
    return summary


def main():
    rows = []
    for T in [500, 2000]:
        rows.append(run(T, ls_true=0.8, var_true=1.0, n_em=25))

    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(rows, f, indent=2)
    print("\nDone.  See:", OUT_DIR)
    print("\n" + "=" * 90)
    print(f"{'T':>5}  {'method':>6}  {'wall':>6}  {'ℓ ratio':>8}  "
          f"{'σ² ratio':>9}  {'drift_traj':>11}  {'drift_grid':>11}")
    print("-" * 90)
    for r in rows:
        for tag, t_, lsf, varf, drmt, drmg in [
            ('MC',   r['wall_mc'],   r['ls_mc'][-1], r['var_mc'][-1],
             r['drift_rmse_traj_mc'], r['drift_rmse_grid_mc']),
            ('GMIX', r['wall_gmix'], r['ls_gmix'][-1], r['var_gmix'][-1],
             r['drift_rmse_traj_gmix'], r['drift_rmse_grid_gmix']),
        ]:
            print(f"{r['T']:>5}  {tag:>6}  {t_:>5.1f}s  "
                  f"{lsf/r['ls_true']:>7.2f}x  {varf/r['var_true']:>8.2f}x  "
                  f"{drmt:>11.4f}  {drmg:>11.4f}")
        print(f"{r['T']:>5}  {'(zero)':>6}                                   "
              f"{'':>11}  {r['drift_rmse_zero']:>11.4f}")


if __name__ == "__main__":
    main()
