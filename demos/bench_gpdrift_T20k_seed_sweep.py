"""GP-drift recovery T=20k: verify the catastrophic-MC finding on a new seed,
and check whether more MC samples (S=8, S=16) rescue it.
"""
from __future__ import annotations

import sys, json, time, math
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
SIGMA = 0.3
ALPHA_RESTORE = 0.5
OUT_DIR = Path(__file__).resolve().parent / "_bench_mc_vs_gmix_out"
OUT_DIR.mkdir(exist_ok=True)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                  * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    keys = jr.split(key, D)
    fk = []
    for k in keys:
        k1, k2 = jr.split(k)
        ee = (jr.normal(k1, (grid.M,))
              + 1j * jr.normal(k2, (grid.M,))).astype(grid.ws.dtype)
        fk.append(grid.ws * ee / math.sqrt(2))
    fk = jnp.stack(fk, axis=0)
    OUT, xcen, h_per_dim = grid.mtot_per_dim, grid.xcen, grid.h_per_dim

    def drift_jax(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk[r].reshape(*OUT),
                           xcen, h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out) - ALPHA_RESTORE * x

    def drift_np(x_batch):
        xt = jnp.asarray(x_batch, dtype=jnp.float32)
        out = np.zeros_like(np.asarray(x_batch))
        for r in range(D):
            fr = jp.nufft2(xt, fk[r].reshape(*OUT),
                           xcen, h_per_dim, eps=6e-8).real
            out[:, r] = (np.asarray(fr)
                          - ALPHA_RESTORE * np.asarray(x_batch)[:, r])
        return out
    return drift_jax, drift_np


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def _fit(method, lik, t_grid, output_params, init_params, ls_init, n_em,
          n_estep, S_marginal, ls_pin):
    rho = jnp.linspace(0.05, 0.4, n_em)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
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


def evaluate_drift(mp, hist, drift_np, xs, t_grid, ls_pin, method,
                    gmix_fine_N=None, gmix_stencil_r=None):
    ms = np.asarray(mp['m'][0])
    Ss = jnp.asarray(np.asarray(mp['S'][0]))
    SSs = jnp.asarray(np.asarray(mp['SS'][0]))
    del_t = t_grid[1:] - t_grid[:-1]
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_pin, hist.variance[-1], X_template, eps=1e-2)
    ws = jpd._ws_real_se(jnp.log(hist.lengthscale[-1]),
                          jnp.log(hist.variance[-1]),
                          grid.xis_flat, grid.h_per_dim[0],
                          D).astype(grid.ws.dtype)
    grid = grid._replace(ws=ws)
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
    f_true_traj = drift_np(np.asarray(xs))
    err_traj = np.linalg.norm(f_inf_traj - f_true_traj, axis=1)
    drift_rmse_traj = float(np.sqrt(np.mean(err_traj ** 2)))

    grid_lo = np.asarray(xs).min(0) - 0.3
    grid_hi = np.asarray(xs).max(0) + 0.3
    xs_g = np.linspace(grid_lo[0], grid_hi[0], 12)
    ys_g = np.linspace(grid_lo[1], grid_hi[1], 12)
    GX, GY = np.meshgrid(xs_g, ys_g, indexing='ij')
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    f_true_grid = drift_np(grid_pts)
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
    )


def run(T, ls_true, var_true, seed, n_em, n_estep, S_list):
    print(f"\n=== T={T}, ls_true={ls_true}, seed={seed} ===")
    drift_jax, drift_np = make_gp_drift(ls_true, var_true,
                                         jr.PRNGKey(seed * 100 + 23))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = T * 0.05
    xs = simulate_sde(jr.PRNGKey(seed * 100 + 7), x0=jnp.array([0., 0.]),
                       f=drift_jax, t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    print(f"  xs range: [{float(xs.min()):.2f}, {float(xs.max()):.2f}]")

    N_obs = 8
    rng = np.random.default_rng(seed)
    C_true = rng.standard_normal((N_obs, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_obs),
                    R=jnp.full((N_obs,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(seed * 100 + 1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    ip = jax.tree_util.tree_map(lambda x: x[None],
        dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    ls_init = 0.3
    ls_pin = ls_true * 0.75

    out = {}
    for S in S_list:
        print(f"  fitting MC, S_marginal={S}...")
        mp_mc, hist_mc, t_mc = _fit('mc', lik, t_grid, out_true, ip, ls_init,
                                      n_em, n_estep, S_marginal=S, ls_pin=ls_pin)
        diag = evaluate_drift(mp_mc, hist_mc, drift_np, xs, t_grid, ls_pin, 'mc')
        print(f"    wall {t_mc:.1f}s  ℓ {hist_mc.lengthscale[-1]:.3f}  "
              f"σ² {hist_mc.variance[-1]:.3f}  drift_traj {diag['drift_rmse_traj']:.4f}")
        out[f'mc_S{S}'] = dict(
            wall=t_mc, ls=list(map(float, hist_mc.lengthscale)),
            var=list(map(float, hist_mc.variance)), **diag)

    print("  fitting GMIX...")
    mp_gx, hist_gx, t_gx = _fit('gmix', lik, t_grid, out_true, ip, ls_init,
                                  n_em, n_estep, S_marginal=2, ls_pin=ls_pin)
    from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    g_eval = jp.spectral_grid_se(ls_pin, 1.0, X_template, eps=1e-2)
    h_spec0 = float(g_eval.h_per_dim[0])
    sigma0_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(ip['V0'][0]))))
    fine_N = pick_grid_size(h_spec=h_spec0, m_extent=6.0, sigma_max=sigma0_max)
    h_g = 1.0 / (fine_N * h_spec0)
    s_r = stencil_radius_for(jnp.asarray(ip['V0'][0])[None], h_g, n_sigma=1.5)
    diag = evaluate_drift(mp_gx, hist_gx, drift_np, xs, t_grid, ls_pin,
                           'gmix', gmix_fine_N=fine_N, gmix_stencil_r=s_r)
    print(f"    wall {t_gx:.1f}s  ℓ {hist_gx.lengthscale[-1]:.3f}  "
          f"σ² {hist_gx.variance[-1]:.3f}  drift_traj {diag['drift_rmse_traj']:.4f}")
    out['gmix'] = dict(wall=t_gx, ls=list(map(float, hist_gx.lengthscale)),
                        var=list(map(float, hist_gx.variance)), **diag)

    out['T'] = T; out['ls_true'] = ls_true; out['var_true'] = var_true
    out['seed'] = seed
    out['drift_rmse_zero'] = diag['drift_rmse_zero']
    return out


def plot(rows):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1: axes = axes[None]
    for ax_row, r in zip(axes, rows):
        seed = r['seed']
        n_em = len(r['gmix']['ls'])
        iters = np.arange(1, n_em + 1)
        ax_row[0].axhline(r['ls_true'], color='k', ls='--', alpha=0.5,
                            label='truth')
        ax_row[1].axhline(r['var_true'], color='k', ls='--', alpha=0.5,
                            label='truth')
        for label, color in [('mc_S2', 'tab:blue'),
                              ('mc_S8', 'tab:cyan'),
                              ('mc_S16', 'tab:purple'),
                              ('gmix', 'tab:orange')]:
            if label in r:
                d = r[label]
                ax_row[0].plot(iters, d['ls'], 'o-', label=label, color=color, ms=3)
                ax_row[1].plot(iters, d['var'], 'o-', label=label, color=color, ms=3)
        ax_row[0].set_xlabel('EM iter'); ax_row[0].set_ylabel(r'$\ell$')
        ax_row[0].set_title(f"T={r['T']}, seed={seed}")
        ax_row[0].legend(fontsize=8)
        ax_row[1].set_xlabel('EM iter'); ax_row[1].set_ylabel(r'$\sigma^2$')
        ax_row[1].set_yscale('log')
        ax_row[1].set_title(f"σ² path (T={r['T']}, seed={seed})")
        ax_row[1].legend(fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "T20k_seed_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    rows = []
    # Two seeds; for each: MC at S=2, 8, 16, plus GMIX
    for seed in [1, 2]:
        rows.append(run(T=20000, ls_true=0.8, var_true=1.0, seed=seed,
                         n_em=20, n_estep=10, S_list=[2, 8, 16]))
        with open(OUT_DIR / f"summary_T20k_seed{seed}.json", 'w') as f:
            json.dump(rows[-1], f, indent=2)

    plot(rows)
    print("\n\n=== SUMMARY ===")
    print(f"{'seed':<6}{'method':<10}{'wall':>7}{'ℓ ratio':>10}{'σ² ratio':>11}"
          f"{'drift_traj':>12}{'drift_grid':>12}")
    for r in rows:
        for label in ['mc_S2', 'mc_S8', 'mc_S16', 'gmix']:
            if label not in r: continue
            d = r[label]
            ls_r = d['ls'][-1] / r['ls_true']
            var_r = d['var'][-1] / r['var_true']
            print(f"{r['seed']:<6}{label:<10}{d['wall']:>5.0f}s "
                  f"{ls_r:>9.2f}x {var_r:>10.2f}x "
                  f"{d['drift_rmse_traj']:>12.4f}{d['drift_rmse_grid']:>12.4f}")
        print(f"{'':<6}{'(zero)':<10}{'':<28}"
              f"{'':<12}{r['drift_rmse_zero']:>12.4f}")


if __name__ == "__main__":
    main()
