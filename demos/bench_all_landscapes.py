"""
Hyper learning curves on log-marginal contours, for all 3 benchmarks
(linear / Duffing / anharmonic).  Mirrors the earlier T7/T10/T18 plots
but covers all three regimes and uses the new exact Cholesky M-step.

For each benchmark:
  - compute EFGP collapsed log-marginal landscape on a grid
  - run EFGP (Cholesky M-step) at mstep_lr=0.01
  - run SparseGP at lr=1e-3 and 3e-3
  - overlay trajectories + init + loss-min on contours
  - record wall-clock per method

Saves /tmp/bench_all_landscapes.png + per-bench .npz files.

Run:
  ~/myenv/bin/python demos/bench_all_landscapes.py
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

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
N_EM = 50
LS_INIT = 0.7
VAR_INIT = 1.0
K_FD = 8
N_GRID_LL = 21
N_GRID_LV = 21
LOG_LS_RANGE = (-2.0, 1.5)
LOG_VAR_RANGE = (-2.5, 2.0)


# ----------------------------------------------------------------------------
# Benchmark configurations (same drifts as efgp_vs_sparsegp_benchmarks.ipynb)
# ----------------------------------------------------------------------------
_A_rot = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])

BENCHMARKS = [
    dict(
        name='Damped rotation (linear)',
        short='linear',
        drift_fn=lambda x, t: _A_rot @ x,
        x0=jnp.array([2.0, 0.0]),
        T=400, t_max=8.0, sigma=0.3, N_obs=8, seed=7,
    ),
    dict(
        name='Duffing double-well',
        short='duffing',
        drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
        x0=jnp.array([1.2, 0.0]),
        T=400, t_max=15.0, sigma=0.2, N_obs=8, seed=13,
    ),
    dict(
        name='Anharmonic oscillator',
        short='anharmonic',
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


def converge_smoother_fixed(lik, op, ip, t_grid, ls_pin, var_pin, sigma):
    """Run SparseGP fixed-hyper EM at given hypers to get reference q(x)."""
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(ls_pin)),
                 output_scale=jnp.asarray(math.sqrt(var_pin)))
    rho_sched = jnp.linspace(0.05, 0.4, 12)
    mp, *_ = fit_variational_em(
        key=jr.PRNGKey(11), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=sp_dp, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=12, n_iters_e=10,
        n_iters_m=1, perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((12,), 0.05), print_interval=999)
    return mp


def build_landscape_state(mp, t_grid, sigma, ls_pin, var_pin):
    """Build (grid, top, z_r) needed to evaluate the EFGP loss at any θ."""
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
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=sigma ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r_init)
    z_r = h0 / ws_safe
    return grid, top, z_r


def loss_at(grid, top, z_r, log_ls, log_var):
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                          jnp.asarray(log_var, dtype=jnp.float32),
                          grid.xis_flat, grid.h_per_dim[0], D)
    ws_c = ws_real.astype(cdtype)
    # T_mat via direct fill
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


def fit_efgp(lik, op, ip, t_grid, sigma, mstep_lr=0.01):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj, wall


def fit_sparsegp(lik, op, ip, t_grid, sigma, lr):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    t0 = time.perf_counter()
    fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=4, perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((N_EM,), lr),
        print_interval=999, drift_params_history=history)
    wall = time.perf_counter() - t0
    ls_traj = np.array([LS_INIT]
                        + [float(np.mean(np.asarray(d['length_scales'])))
                           for d in history])
    var_traj = np.array([VAR_INIT]
                         + [float(np.asarray(d['output_scale'])) ** 2
                            for d in history])
    return ls_traj, var_traj, wall


def run_one_benchmark(cfg):
    print(f"\n[bench] === {cfg['name']} ===", flush=True)
    print(f"  T={cfg['T']}, sigma={cfg['sigma']}, n_em={N_EM}", flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(cfg)
    sigma = cfg['sigma']

    print(f"  Computing landscape (smoother at ℓ_init={LS_INIT}, "
          f"σ²_init={VAR_INIT})...", flush=True)
    mp_ref = converge_smoother_fixed(lik, op, ip, t_grid,
                                       LS_INIT, VAR_INIT, sigma)
    grid, top, z_r = build_landscape_state(
        mp_ref, t_grid, sigma, LS_INIT, VAR_INIT)
    LOG_LS_GRID = np.linspace(*LOG_LS_RANGE, N_GRID_LL)
    LOG_VAR_GRID = np.linspace(*LOG_VAR_RANGE, N_GRID_LV)
    L_grid = np.zeros((N_GRID_LL, N_GRID_LV))
    for i, ll in enumerate(LOG_LS_GRID):
        for j, lv in enumerate(LOG_VAR_GRID):
            L_grid[i, j] = loss_at(grid, top, z_r, float(ll), float(lv))
    idx_min = np.unravel_index(np.argmin(L_grid), L_grid.shape)
    ll_min = float(LOG_LS_GRID[idx_min[0]])
    lv_min = float(LOG_VAR_GRID[idx_min[1]])
    L_min = float(L_grid[idx_min])
    print(f"  Loss min: ℓ={math.exp(ll_min):.3f}, σ²={math.exp(lv_min):.3f}, "
          f"α={math.exp(lv_min-2*ll_min):.3f}", flush=True)

    print(f"  Running EFGP (Cholesky M-step) at mstep_lr=0.01...", flush=True)
    e_ls, e_var, e_wall = fit_efgp(lik, op, ip, t_grid, sigma, mstep_lr=0.01)
    print(f"    final ℓ={e_ls[-1]:.3f}, σ²={e_var[-1]:.3f}, wall={e_wall:.1f}s",
          flush=True)

    print(f"  Running SparseGP @ lr=3e-3 (best from our experiments)...",
          flush=True)
    s_ls, s_var, s_wall = fit_sparsegp(lik, op, ip, t_grid, sigma, lr=3e-3)
    print(f"    final ℓ={s_ls[-1]:.3f}, σ²={s_var[-1]:.3f}, wall={s_wall:.1f}s",
          flush=True)

    out = dict(
        name=cfg['name'], short=cfg['short'],
        LOG_LS_GRID=LOG_LS_GRID, LOG_VAR_GRID=LOG_VAR_GRID,
        L_grid=L_grid, L_min=L_min, ll_min=ll_min, lv_min=lv_min,
        e_ls=e_ls, e_var=e_var, e_wall=e_wall,
        s_ls=s_ls, s_var=s_var, s_wall=s_wall,
    )
    np.savez(f"/tmp/bench_landscape_{cfg['short']}.npz", **out)
    return out


def render_panel(ax, data, fig=None, show_cbar=False):
    LOG_LS = data['LOG_LS_GRID']; LOG_VAR = data['LOG_VAR_GRID']
    L_grid = data['L_grid']; L_min = data['L_min']
    ll_min = data['ll_min']; lv_min = data['lv_min']

    L_norm = L_grid - L_min
    levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    cf = ax.contourf(LOG_LS, LOG_VAR, L_norm.T,
                      levels=[0] + levels, cmap='viridis_r', extend='max')
    if show_cbar and fig is not None:
        fig.colorbar(cf, ax=ax, label='L − L_min')
    cs = ax.contour(LOG_LS, LOG_VAR, L_norm.T,
                    levels=levels, colors='k', linewidths=0.4)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
    for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        ax.plot(LOG_LS, LOG_LS * 2 + log_alpha_show,
                color='white', linestyle=':', linewidth=0.6, alpha=0.5)

    # Trajectories
    ax.plot(np.log(data['e_ls']), np.log(data['e_var']),
            '-o', color='C0', markersize=3, linewidth=1.6,
            label=f"EFGP (Cholesky), lr=0.01  [{data['e_wall']:.0f}s]")
    ax.plot(np.log(data['s_ls']), np.log(data['s_var']),
            '-s', color='C3', markersize=3, linewidth=1.4,
            label=f"SparseGP, lr=3e-3  [{data['s_wall']:.0f}s]")

    # init / loss-min markers
    ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                marker='+', s=200, color='black', zorder=6,
                label=f'init (ℓ={LS_INIT}, σ²={VAR_INIT})')
    ax.scatter([ll_min], [lv_min], marker='*', s=240,
                color='gold', edgecolor='k', zorder=7,
                label=f"loss min: ℓ={math.exp(ll_min):.2f}, "
                      f"σ²={math.exp(lv_min):.2f}")
    ax.set_xlabel('log ℓ')
    ax.set_ylabel('log σ²')
    ax.set_xlim(LOG_LS.min(), LOG_LS.max())
    ax.set_ylim(LOG_VAR.min(), LOG_VAR.max())
    ax.set_title(data['name'], fontsize=10)
    ax.legend(loc='lower right', fontsize=7)


def main():
    print(f"[bench] Hyper learning curves on log-marginal contours, "
          f"all benchmarks", flush=True)
    print(f"[bench] init=(ℓ={LS_INIT}, σ²={VAR_INIT}), n_em={N_EM}",
          flush=True)
    results = []
    for cfg in BENCHMARKS:
        r = run_one_benchmark(cfg)
        results.append(r)

    # Multi-panel plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
        for ax, r in zip(axes, results):
            render_panel(ax, r, fig=fig, show_cbar=False)
        # Single colorbar at the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        L_norm0 = results[0]['L_grid'] - results[0]['L_min']
        cf = axes[0].contourf(results[0]['LOG_LS_GRID'],
                                results[0]['LOG_VAR_GRID'],
                                L_norm0.T,
                                levels=[0] + levels, cmap='viridis_r',
                                extend='max', alpha=0)
        fig.colorbar(cf, cax=cbar_ax, label='L − L_min')
        fig.suptitle('EFGP (new Cholesky M-step) vs SparseGP — '
                     'hyper trajectories on EFGP collapsed log-marginal contours\n'
                     '(white dotted = constant α=σ²/ℓ²)',
                     fontsize=11)
        plt.tight_layout(rect=[0, 0, 0.91, 0.96])
        plt.savefig('/tmp/bench_all_landscapes.png', dpi=140)
        print(f"\n[bench] Saved /tmp/bench_all_landscapes.png", flush=True)
    except Exception as e:
        print(f"[bench] Plot failed: {e}", flush=True)

    # Wall-time summary
    print(f"\n[bench] Wall-time summary:", flush=True)
    print(f"  {'benchmark':25s}  {'EFGP':>10s}  {'SparseGP@3e-3':>14s}  "
          f"{'speedup':>9s}", flush=True)
    for r in results:
        e = r['e_wall']; s = r['s_wall']
        print(f"  {r['name']:25s}  {e:>9.1f}s  {s:>13.1f}s  "
              f"{s/e:>8.2f}x", flush=True)


if __name__ == '__main__':
    main()
