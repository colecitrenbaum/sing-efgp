"""
T10: Identifiability test on a *true* GP-drifted dataset.

User insight (correct): on Duffing the marginal likelihood is fundamentally
ridge-shaped because the true drift is deterministic and not a GP draw —
there's no "true" (ℓ, σ²) for the data to identify. To test whether EFGP and
SparseGP recover individual hypers when they ARE identifiable, generate
from a known SE-GP drift and run the same landscape + trajectory diagnostic.

Plus: vary S_marginal to test whether the trajectory wandering is driven by
the Stein-cloud MC noise (different pseudo-points sampled each E-step iter
→ moving surrogate objective).

Setup:
  - Sample a 2D drift from SE-GP at (ℓ_true=0.8, σ²_true=1.0)
  - Simulate SDE for T=400 → that's enough data to be ID-able
  - Compute landscape at SparseGP-converged smoother (as before)
  - Capture EFGP trajectories at S_marginal ∈ {2, 8, 16}
  - Capture SparseGP@1e-3 trajectory
  - Mark (ℓ_true, σ²_true) on the plot

Run:
  ~/myenv/bin/python demos/diag_gp_drift_landscape.py
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
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
N_EM = 25
LS_TRUE = 0.8
VAR_TRUE = 1.0
# init deliberately on a *different α-ridge* (α_init=0.075 vs α_true=1.56) so
# the methods must move across-ridge to recover truth, not just along-ridge.
LS_INIT = 2.0          # ℓ_init/ℓ_true = 2.5
VAR_INIT = 0.3         # σ²_init/σ²_true = 0.3 (opposite direction from ℓ)
S_MARGINAL_LIST = [2, 8, 16]
K_FD = 8


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
    """Sample a 2D drift from SE-GP with known (ls, var)."""
    X_template = (jnp.linspace(-extent, extent, 16)[:, None]
                   * jnp.ones((1, D)))
    grid = jp.spectral_grid_se(ls_true, var_true, X_template, eps=eps_grid)
    M = grid.M
    keys = jr.split(key, D)
    eps = jnp.stack([
        (jax.random.normal(jr.split(k)[0], (M,))
         + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)
        for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps   # (D, M)

    def drift(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)

    return drift, fk_draw, grid


def setup():
    drift, fk_draw, grid_truth = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    # Clip to avoid blow-up if the GP-drift sample takes the trajectory far
    xs = jnp.clip(xs, -3.0, 3.0)
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


# ---------------------------------------------------------------------------
def converge_smoother(lik, op, ip, t_grid):
    """Run SparseGP fixed-hyper EM at (ℓ_true, σ²_true) to get a clean shared
    smoother for the landscape evaluation."""
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(LS_TRUE)),
                 output_scale=jnp.asarray(math.sqrt(VAR_TRUE)))
    n_em_local = 12
    rho_sched = jnp.linspace(0.05, 0.4, n_em_local)
    mp, *_ = fit_variational_em(
        key=jr.PRNGKey(11), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=sp_dp, init_params=ip, output_params=op,
        sigma=SIGMA, rho_sched=rho_sched, n_iters=n_em_local, n_iters_e=6,
        n_iters_m=1, perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((n_em_local,), 0.05), print_interval=999)
    return mp


def build_landscape_state(mp, t_grid, eps_grid=1e-3):
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(LS_TRUE, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(VAR_TRUE, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=eps_grid)
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=16,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h0 = jax.vmap(A_init)(mu_r_init)
    z_r = h0 / ws_safe
    return grid, top, z_r


def loss_at(log_ls, log_var, *, grid, top, z_r):
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                          jnp.asarray(log_var, dtype=jnp.float32),
                          grid.xis_flat, grid.h_per_dim[0], D)
    ws_c = ws_real.astype(cdtype)
    eye_c = jnp.eye(M, dtype=cdtype)
    T_cols = jax.vmap(lambda e: jp.toeplitz_apply(top, e))(eye_c)
    T_mat = T_cols.T
    D_w = ws_c[:, None] * eye_c
    A = eye_c + D_w @ T_mat @ D_w
    h = ws_c[None, :] * z_r
    mu = jax.vmap(lambda b: jnp.linalg.solve(A, b))(h)

    def per_r(r):
        t1 = jnp.vdot(h[r], mu[r]).real
        t2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
        return -(t1 - t2)
    det_loss = jax.vmap(per_r)(jnp.arange(D)).sum()
    sign, logdet = jnp.linalg.slogdet(A)
    return float(det_loss + 0.5 * D * logdet.real)


def fit_efgp(lik, op, ip, t_grid, *, S_marginal):
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=S_marginal,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.05,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        verbose=False)
    ls_traj = np.array([LS_INIT] + list(hist.lengthscale))
    var_traj = np.array([VAR_INIT] + list(hist.variance))
    return ls_traj, var_traj


def fit_sparsegp(lik, op, ip, t_grid, lr=1e-3):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    fit_variational_em(
        key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params0, init_params=ip, output_params=op,
        sigma=SIGMA, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=4, perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((N_EM,), lr),
        print_interval=999, drift_params_history=history)
    ls_traj = np.array([LS_INIT] + [float(np.mean(np.asarray(d['length_scales'])))
                                      for d in history])
    var_traj = np.array([VAR_INIT] + [float(np.asarray(d['output_scale'])) ** 2
                                       for d in history])
    return ls_traj, var_traj


def main():
    print(f"[T10] JAX devices: {jax.devices()}", flush=True)
    print(f"[T10] GP-drift identifiability test", flush=True)
    print(f"[T10] T={T}, sigma={SIGMA}, ls_true={LS_TRUE}, var_true={VAR_TRUE}",
          flush=True)
    print(f"[T10]   init=(ℓ={LS_INIT}, σ²={VAR_INIT})  far from truth", flush=True)
    print(f"[T10]   sweeping S_marginal ∈ {S_MARGINAL_LIST}", flush=True)

    xs, ys, lik, op, ip, t_grid = setup()

    print("\n[T10] Computing shared smoother at TRUE hypers...", flush=True)
    mp = converge_smoother(lik, op, ip, t_grid)
    grid, top, z_r = build_landscape_state(mp, t_grid)
    print(f"[T10] Landscape grid: K={K_FD}, M={int(grid.M)}", flush=True)

    LOG_LS_GRID = np.linspace(-2.0, 1.5, 25)
    LOG_VAR_GRID = np.linspace(-3.0, 2.5, 25)
    L_grid = np.zeros((len(LOG_LS_GRID), len(LOG_VAR_GRID)))
    print(f"[T10] Computing loss landscape...", flush=True)
    for i, ll in enumerate(LOG_LS_GRID):
        for j, lv in enumerate(LOG_VAR_GRID):
            L_grid[i, j] = loss_at(float(ll), float(lv),
                                     grid=grid, top=top, z_r=z_r)
    idx_min = np.unravel_index(np.argmin(L_grid), L_grid.shape)
    ll_min = float(LOG_LS_GRID[idx_min[0]])
    lv_min = float(LOG_VAR_GRID[idx_min[1]])
    L_min = float(L_grid[idx_min])

    log_ls_true = math.log(LS_TRUE)
    log_var_true = math.log(VAR_TRUE)
    L_at_true = loss_at(log_ls_true, log_var_true,
                         grid=grid, top=top, z_r=z_r)

    print(f"\n[T10] Loss landscape summary:", flush=True)
    print(f"  Loss min on grid:  ℓ={math.exp(ll_min):.3f}, σ²={math.exp(lv_min):.3f}, "
          f"α={math.exp(lv_min - 2*ll_min):.3f}", flush=True)
    print(f"  Loss at TRUE:      ℓ={LS_TRUE:.3f}, σ²={VAR_TRUE:.3f}, "
          f"α={VAR_TRUE/LS_TRUE**2:.3f}", flush=True)
    print(f"  Δ(L_at_true - L_min) = {L_at_true - L_min:.4f}", flush=True)

    # Identifiability check via curvatures
    delta = 0.1
    L0 = loss_at(ll_min, lv_min, grid=grid, top=top, z_r=z_r)
    along = np.array([1.0, 2.0]) / math.sqrt(5)
    across = np.array([2.0, -1.0]) / math.sqrt(5)
    L_p_a = loss_at(ll_min + delta*along[0], lv_min + delta*along[1],
                     grid=grid, top=top, z_r=z_r)
    L_m_a = loss_at(ll_min - delta*along[0], lv_min - delta*along[1],
                     grid=grid, top=top, z_r=z_r)
    L_p_x = loss_at(ll_min + delta*across[0], lv_min + delta*across[1],
                     grid=grid, top=top, z_r=z_r)
    L_m_x = loss_at(ll_min - delta*across[0], lv_min - delta*across[1],
                     grid=grid, top=top, z_r=z_r)
    curv_along = (L_p_a + L_m_a - 2*L0) / (delta**2)
    curv_across = (L_p_x + L_m_x - 2*L0) / (delta**2)
    if abs(curv_along) > 0:
        ratio = curv_across / curv_along
        print(f"  Curvature ratio (across/along) = {ratio:.2f}", flush=True)
        if ratio > 5:
            print(f"  → Still strongly ridge-shaped — only α identifiable.",
                  flush=True)
        else:
            print(f"  → Less ridge-shaped — individual hypers MORE identifiable.",
                  flush=True)

    # Run EFGP at multiple S_marginal
    efgp_trajs = {}
    for S in S_MARGINAL_LIST:
        print(f"\n[T10] Running EFGP, S_marginal={S}...", flush=True)
        ls_t, var_t = fit_efgp(lik, op, ip, t_grid, S_marginal=S)
        efgp_trajs[S] = (ls_t, var_t)
        print(f"  Final: ℓ={ls_t[-1]:.3f}, σ²={var_t[-1]:.3f}, "
              f"α={var_t[-1]/ls_t[-1]**2:.3f}  "
              f"(truth: ℓ={LS_TRUE}, σ²={VAR_TRUE}, α={VAR_TRUE/LS_TRUE**2:.3f})",
              flush=True)

    print(f"\n[T10] Running SparseGP @ lr=1e-3...", flush=True)
    sp1_ls, sp1_var = fit_sparsegp(lik, op, ip, t_grid, lr=1e-3)
    print(f"  Final: ℓ={sp1_ls[-1]:.3f}, σ²={sp1_var[-1]:.3f}, "
          f"α={sp1_var[-1]/sp1_ls[-1]**2:.3f}", flush=True)

    np.savez('/tmp/T10_gp_drift_landscape.npz',
              log_ls_grid=LOG_LS_GRID, log_var_grid=LOG_VAR_GRID,
              L_grid=L_grid, L_min=L_min, ll_min=ll_min, lv_min=lv_min,
              ls_true=LS_TRUE, var_true=VAR_TRUE,
              sp1_ls=sp1_ls, sp1_var=sp1_var,
              **{f'efgp_S{S}_ls': v[0] for S, v in efgp_trajs.items()},
              **{f'efgp_S{S}_var': v[1] for S, v in efgp_trajs.items()})
    print('\n[T10] Saved /tmp/T10_gp_drift_landscape.npz', flush=True)

    # Render
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7))
        L_norm = L_grid - L_min
        levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        cf = ax.contourf(LOG_LS_GRID, LOG_VAR_GRID, L_norm.T,
                          levels=[0] + levels, cmap='viridis_r', extend='max')
        fig.colorbar(cf, ax=ax, label='L − L_min')
        cs = ax.contour(LOG_LS_GRID, LOG_VAR_GRID, L_norm.T,
                        levels=levels, colors='k', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%g')
        for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ax.plot(LOG_LS_GRID, LOG_LS_GRID * 2 + log_alpha_show,
                    color='white', linestyle=':', linewidth=0.7, alpha=0.5)

        # EFGP trajectories at varying S_marginal
        colors_efgp = ['C0', 'C2', 'C4']
        for color, S in zip(colors_efgp, S_MARGINAL_LIST):
            ls_t, var_t = efgp_trajs[S]
            ax.plot(np.log(ls_t), np.log(var_t), '-o', color=color,
                    markersize=4, linewidth=1.5,
                    label=f'EFGP S_marg={S}', zorder=4)
        ax.plot(np.log(sp1_ls), np.log(sp1_var), '-s', color='C1',
                markersize=4, linewidth=1.5,
                label='SparseGP @ lr=1e-3', zorder=4)

        # Init and truth
        ax.scatter([math.log(LS_INIT)], [math.log(VAR_INIT)],
                    marker='+', s=200, color='black', zorder=6,
                    label=f'init (ℓ={LS_INIT}, σ²={VAR_INIT})')
        ax.scatter([log_ls_true], [log_var_true],
                    marker='X', s=240, color='red', edgecolor='k', zorder=6,
                    label=f'TRUTH (ℓ={LS_TRUE}, σ²={VAR_TRUE})')
        ax.scatter([ll_min], [lv_min], marker='*', s=240,
                    color='gold', edgecolor='k', zorder=5,
                    label=f'loss min: ℓ={math.exp(ll_min):.2f}, σ²={math.exp(lv_min):.2f}')

        ax.set_xlabel('log ℓ')
        ax.set_ylabel('log σ²')
        ax.set_xlim(LOG_LS_GRID.min(), LOG_LS_GRID.max())
        ax.set_ylim(LOG_VAR_GRID.min(), LOG_VAR_GRID.max())
        ax.set_title(f'GP-drifted data (truth ℓ={LS_TRUE}, σ²={VAR_TRUE}): '
                     f'EFGP S_marg sweep + SparseGP\n'
                     f'(red X = truth, gold ★ = loss min on landscape)')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig('/tmp/T10_gp_drift_landscape_with_trajectories.png', dpi=150)
        print('[T10] Saved /tmp/T10_gp_drift_landscape_with_trajectories.png',
              flush=True)
    except Exception as e:
        print(f'[T10] Plot rendering failed: {e}', flush=True)


if __name__ == '__main__':
    main()
