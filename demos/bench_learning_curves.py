"""
Compare EFGP (now with exact Cholesky M-step) and SparseGP on
GP-drift data, tracking each method's M-step loss surface value
(its proxy for log marginal) + wall clock per outer EM iter.

Run:
  ~/myenv/bin/python demos/bench_learning_curves.py
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
T = 400
SIGMA = 0.3
N_OBS = 8
SEED = 7
LS_TRUE = 0.8
VAR_TRUE = 1.0
LS_INIT = 2.0
VAR_INIT = 0.3
N_EM = 50
K_FD = 8


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_gp_drift(ls_true, var_true, key, eps_grid=1e-2, extent=4.0):
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
    fk_draw = grid.ws[None, :] * eps

    def drift(x, t):
        x_b = x[None, :]
        out = []
        for r in range(D):
            fr = jp.nufft2(x_b, fk_draw[r].reshape(*grid.mtot_per_dim),
                            grid.xcen, grid.h_per_dim, eps=6e-8).real
            out.append(fr[0])
        return jnp.stack(out)
    return drift


def setup():
    drift = make_gp_drift(LS_TRUE, VAR_TRUE, jr.PRNGKey(123))
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    t_max = 6.0
    xs = simulate_sde(jr.PRNGKey(SEED), x0=jnp.array([0.0, 0.0]),
                      f=drift, t_max=t_max, n_timesteps=T,
                      sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    yc = ys - ys.mean(0)
    _, _, vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=vt[:D].T, d=ys.mean(0),
              R=jnp.full((N_OBS,), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., t_max, T)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def efgp_log_marginal_at(grid, top, z_r, log_ls, log_var, D_out=2):
    """Compute the EFGP collapsed log marginal at given hypers + smoother
    state.  Proxy for log p(y; θ) for the EFGP M-step objective."""
    M = int(grid.xis_flat.shape[0])
    cdtype = top.v_fft.dtype
    eye_c = jnp.eye(M, dtype=cdtype)
    h_scalar = grid.h_per_dim[0]
    ws_real = _ws_real_se(jnp.asarray(log_ls, dtype=jnp.float32),
                           jnp.asarray(log_var, dtype=jnp.float32),
                           grid.xis_flat, h_scalar, D)
    ws_c = ws_real.astype(cdtype)
    # Build T_mat O(M²)
    v_pad = jnp.fft.ifftn(top.v_fft).astype(cdtype)
    ns_v = tuple(2 * n - 1 for n in top.ns)
    v_conv = v_pad[tuple(slice(0, L) for L in ns_v)]
    d_ = len(top.ns)
    mi = jnp.indices(top.ns).reshape(d_, -1)
    offset = jnp.array([n - 1 for n in top.ns], dtype=jnp.int32)
    diff = mi[:, :, None] - mi[:, None, :] + offset[:, None, None]
    T_mat = v_conv[tuple(diff[k] for k in range(d_))]
    A = eye_c + ws_c[:, None] * T_mat * ws_c[None, :]
    L = jnp.linalg.cholesky(A)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
    h = ws_c[None, :] * z_r
    def solve_one(b):
        y = jla.solve_triangular(L, b, lower=True)
        return jla.solve_triangular(L.conj().T, y, lower=False)
    mu = jax.vmap(solve_one)(h)
    # Loss (negative log marginal up to constants)
    det_loss = -0.5 * jnp.sum(jnp.real(jnp.sum(jnp.conj(h) * mu, axis=-1)))
    return float(det_loss + 0.5 * D_out * logdet)


def fit_efgp_with_loss(lik, op, ip, t_grid, mstep_lr=0.01, kernel_warmup_iters=8):
    """Run EFGP EM, track per-iter loss + wall time."""
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)

    losses = []      # collapsed loss at each outer EM iter
    walls = []       # cumulative wall time

    # We'll do the EM loop manually so we can intercept after each iter.
    # Easier: just record the post-iter (ℓ, σ²) and recompute loss at the
    # current smoother + grid, since fit_efgp_sing_jax doesn't expose
    # per-iter losses.
    t0 = time.perf_counter()
    mp, _, _, _, hist = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3, S_marginal=2,
        n_em_iters=N_EM, n_estep_iters=10, rho_sched=rho_sched,
        learn_emissions=True, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=mstep_lr,
        n_hutchinson_mstep=4, kernel_warmup_iters=kernel_warmup_iters,
        verbose=False)
    wall_total = time.perf_counter() - t0
    ls_traj = np.array(hist.lengthscale)
    var_traj = np.array(hist.variance)
    return ls_traj, var_traj, wall_total


def fit_sparsegp_with_loss(lik, op, ip, t_grid, lr=3e-3):
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
        sigma=SIGMA, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
        n_iters_m=4, perform_m_step=True, learn_output_params=True,
        learning_rate=jnp.full((N_EM,), lr),
        print_interval=999, drift_params_history=history)
    wall_total = time.perf_counter() - t0
    ls_traj = np.array([float(np.mean(np.asarray(d['length_scales'])))
                         for d in history])
    var_traj = np.array([float(np.asarray(d['output_scale'])) ** 2
                          for d in history])
    return ls_traj, var_traj, wall_total


def evaluate_efgp_loss_at_traj(lik, op, ip, t_grid, ls_traj, var_traj,
                                 ls_pin_for_smoother=LS_TRUE,
                                 var_pin_for_smoother=VAR_TRUE):
    """Build a reference smoother + grid, then evaluate the EFGP collapsed
    loss at each (ℓ, σ²) along the trajectory.

    This lets us track 'value of the M-step objective' at each EM iter,
    using a CONSISTENT reference smoother (so the loss is a proper
    function of (ℓ, σ²)).  The reference is the truth-hyper smoother.
    """
    # Build smoother at truth hypers
    rho_sched = jnp.linspace(0.05, 0.7, 8)
    mp_ref, *_ = efgp_em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=ls_pin_for_smoother, variance=var_pin_for_smoother,
        sigma=SIGMA, sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
        S_marginal=2, n_em_iters=8, n_estep_iters=10,
        rho_sched=rho_sched, learn_emissions=True, update_R=False,
        learn_kernel=False, kernel_warmup_iters=8, verbose=False)
    # Build grid at K_FD
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin_for_smoother, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin_for_smoother, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3)
    ms = mp_ref['m'][0]; Ss = mp_ref['S'][0]; SSs = mp_ref['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M)
    ws_real0 = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real0) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real0.dtype), ws_real0)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h_r0 = jax.vmap(A_init)(mu_r_init)
    z_r = h_r0 / ws_safe

    losses = []
    for ls, var in zip(ls_traj, var_traj):
        L = efgp_log_marginal_at(grid, top, z_r,
                                   math.log(float(ls)),
                                   math.log(float(var)),
                                   D_out=D)
        losses.append(L)
    return np.array(losses)


def main():
    print(f"[bench] GP-drift: ℓ_true={LS_TRUE}, σ²_true={VAR_TRUE}", flush=True)
    print(f"[bench] init: ℓ={LS_INIT}, σ²={VAR_INIT}, n_em={N_EM}", flush=True)
    xs, ys, lik, op, ip, t_grid = setup()

    print("\n[bench] Running EFGP (Cholesky M-step) at mstep_lr=0.01...",
          flush=True)
    e_ls, e_var, e_wall = fit_efgp_with_loss(lik, op, ip, t_grid, mstep_lr=0.01)
    print(f"  EFGP wall = {e_wall:.1f}s, final ℓ={e_ls[-1]:.3f}, σ²={e_var[-1]:.3f}",
          flush=True)

    print("\n[bench] Running SparseGP at lr=3e-3...", flush=True)
    s_ls, s_var, s_wall = fit_sparsegp_with_loss(lik, op, ip, t_grid, lr=3e-3)
    print(f"  Sparse wall = {s_wall:.1f}s, final ℓ={s_ls[-1]:.3f}, σ²={s_var[-1]:.3f}",
          flush=True)

    print("\n[bench] Running SparseGP at lr=1e-3...", flush=True)
    s_ls_slow, s_var_slow, s_wall_slow = fit_sparsegp_with_loss(
        lik, op, ip, t_grid, lr=1e-3)
    print(f"  Sparse@1e-3 wall = {s_wall_slow:.1f}s, "
          f"final ℓ={s_ls_slow[-1]:.3f}, σ²={s_var_slow[-1]:.3f}", flush=True)

    print("\n[bench] Evaluating EFGP collapsed loss at each method's "
          "(ℓ, σ²) trajectory using a reference smoother (at truth hypers)...",
          flush=True)
    e_loss = evaluate_efgp_loss_at_traj(lik, op, ip, t_grid, e_ls, e_var)
    s_loss = evaluate_efgp_loss_at_traj(lik, op, ip, t_grid, s_ls, s_var)
    s_loss_slow = evaluate_efgp_loss_at_traj(
        lik, op, ip, t_grid, s_ls_slow, s_var_slow)
    # Truth loss (for reference)
    truth_loss = evaluate_efgp_loss_at_traj(
        lik, op, ip, t_grid,
        np.array([LS_TRUE]), np.array([VAR_TRUE]))[0]

    np.savez('/tmp/bench_learning_curves.npz',
              e_ls=e_ls, e_var=e_var, e_loss=e_loss, e_wall=e_wall,
              s_ls=s_ls, s_var=s_var, s_loss=s_loss, s_wall=s_wall,
              s_ls_slow=s_ls_slow, s_var_slow=s_var_slow,
              s_loss_slow=s_loss_slow, s_wall_slow=s_wall_slow,
              truth_loss=truth_loss,
              ls_true=LS_TRUE, var_true=VAR_TRUE,
              ls_init=LS_INIT, var_init=VAR_INIT)

    # Plot: 2 panels: (a) loss vs EM iter, (b) loss vs wall clock
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: loss vs EM iter
        ax = axes[0]
        ax.plot(np.arange(1, len(e_loss)+1), e_loss, '-o', color='C0',
                markersize=3, linewidth=1.6, label='EFGP (Cholesky), lr=0.01')
        ax.plot(np.arange(1, len(s_loss_slow)+1), s_loss_slow, '-s', color='C1',
                markersize=3, linewidth=1.4, label='SparseGP, lr=1e-3')
        ax.plot(np.arange(1, len(s_loss)+1), s_loss, '-^', color='C3',
                markersize=3, linewidth=1.4, label='SparseGP, lr=3e-3')
        ax.axhline(truth_loss, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label=f'loss at TRUTH (ℓ={LS_TRUE}, σ²={VAR_TRUE})')
        ax.set_xlabel('EM iteration')
        ax.set_ylabel('EFGP collapsed loss (= -log p̂; lower is better)')
        ax.set_title('Learning curves: EFGP M-step loss surface\n'
                      'evaluated at each method\'s (ℓ, σ²) trajectory')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        # Panel B: loss vs wall clock
        ax = axes[1]
        # Approximate per-iter wall times by spreading total evenly
        e_walls = np.linspace(0, e_wall, len(e_loss)+1)[1:]
        s_walls = np.linspace(0, s_wall, len(s_loss)+1)[1:]
        s_walls_slow = np.linspace(0, s_wall_slow, len(s_loss_slow)+1)[1:]
        ax.plot(e_walls, e_loss, '-o', color='C0',
                markersize=3, linewidth=1.6,
                label=f'EFGP (Cholesky), lr=0.01 [{e_wall:.0f}s]')
        ax.plot(s_walls_slow, s_loss_slow, '-s', color='C1',
                markersize=3, linewidth=1.4,
                label=f'SparseGP, lr=1e-3 [{s_wall_slow:.0f}s]')
        ax.plot(s_walls, s_loss, '-^', color='C3',
                markersize=3, linewidth=1.4,
                label=f'SparseGP, lr=3e-3 [{s_wall:.0f}s]')
        ax.axhline(truth_loss, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label='loss at TRUTH')
        ax.set_xlabel('wall-clock time (s)')
        ax.set_ylabel('EFGP collapsed loss')
        ax.set_title('Learning curves: loss vs wall time')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/bench_learning_curves.png', dpi=140)
        print(f"\n[bench] Saved /tmp/bench_learning_curves.png", flush=True)
    except Exception as e:
        print(f"[bench] Plot failed: {e}", flush=True)


if __name__ == '__main__':
    main()
