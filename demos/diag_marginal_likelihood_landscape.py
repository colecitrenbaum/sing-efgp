"""
T7: Is the collapsed marginal likelihood ridge-shaped or peaked?

User pushback (correct): α=σ²/ℓ² agreement is necessary but not
sufficient. If the data identifies (ℓ, σ²) individually, both methods
should converge to similar values. If only α agrees, either one method
is biased, OR the marginal likelihood at this data is genuinely
ridge-shaped (only α is identifiable).

This script computes the EFGP collapsed marginal likelihood
L(log_ℓ, log_σ²) on a fine 2-D grid at a fixed converged q(x), then
plots / reports the contours. We use the EXACT slogdet path (small K, dense
A) — no Hutchinson noise.

  L(θ) = -Re(h_θ * μ_θ) + ½ μ_θ* A_θ μ_θ + ½ D_out * log|A_θ|

This is the actual M-step loss being minimized. If it's peaked at a
single (ℓ*, σ²*), both methods should converge to that point — and the
T2a observation that EFGP and SparseGP land at different (ℓ, σ²) but
same α would mean at least one is biased.

If it's ridge-shaped along σ²/ℓ² = const, then individual hypers are
genuinely under-determined and α-agreement is the strongest test
available.

Run:
  ~/myenv/bin/python demos/diag_marginal_likelihood_landscape.py
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
T = 300
SIGMA = 0.3
N_OBS = 8
SEED = 7
K_FD = 8

# Wide range to show the full landscape, not just the L≈min basin.
# log_ℓ ∈ [-2, 1.5] → ℓ ∈ [0.135, 4.48]
# log_σ² ∈ [-3, 2.5] → σ² ∈ [0.05, 12.18]
LOG_LS_GRID = np.linspace(-2.0, 1.5, 25)
LOG_VAR_GRID = np.linspace(-3.0, 2.5, 25)


def drift_for(regime):
    if regime == "linear":
        A = jnp.array([[-0.6, 1.0], [-1.0, -0.6]])
        return lambda x, t: A @ x, jnp.array([2.0, 0.0]), 6.0
    if regime == "duffing":
        return (lambda x, t: jnp.array(
                    [x[1], 1.0 * x[0] - 1.0 * x[0]**3 - 0.5 * x[1]]),
                jnp.array([1.5, 0.0]), T * 0.05)
    if regime == "anharmonic":
        return (lambda x, t: jnp.array(
                    [x[1], -x[0] - 0.3 * x[1] - 0.5 * x[0]**3]),
                jnp.array([1.0, -0.6]), T * 0.05)
    raise ValueError(regime)


def setup(regime):
    drift, x0, t_max = drift_for(regime)
    sigma_fn = lambda x, t: SIGMA * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(SEED), x0=x0, f=drift,
                      t_max=t_max, n_timesteps=T, sigma=sigma_fn)
    rng = np.random.default_rng(SEED)
    C_true = rng.standard_normal((N_OBS, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N_OBS),
                    R=jnp.full((N_OBS,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(SEED + 1), xs, out_true)
    lik = Gaussian(ys[None], jnp.ones((1, T), dtype=bool))
    return xs, ys, lik, t_max


def converge_smoother(lik, t_max, ls_init=1.0, var_init=1.0, n_em=20):
    yc = lik.ys_obs[0] - lik.ys_obs[0].mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op_init = dict(C=Vt[:D].T, d=lik.ys_obs[0].mean(0),
                   R=jnp.full((lik.ys_obs.shape[2],), 0.1))
    ip_init = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    quad = GaussHermiteQuadrature(D=D, n_quad=7)
    zs = initialize_zs(D=D, zs_lim=3.0, num_per_dim=8)
    sparse_drift = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    sp_dp = dict(length_scales=jnp.full((D,), float(ls_init)),
                 output_scale=jnp.asarray(math.sqrt(var_init)))
    rho_sched = jnp.linspace(0.05, 0.4, n_em)
    t_grid = jnp.linspace(0., t_max, T)
    mp, _, _, _, _, _, _, _ = fit_variational_em(
        key=jr.PRNGKey(11),
        fn=sparse_drift, likelihood=lik, t_grid=t_grid, drift_params=sp_dp,
        init_params=ip_init, output_params=op_init, sigma=SIGMA,
        rho_sched=rho_sched, n_iters=n_em, n_iters_e=6, n_iters_m=1,
        perform_m_step=False, learn_output_params=False,
        learning_rate=jnp.full((n_em,), 0.05), print_interval=999,
    )
    return mp, t_grid


def build_state(mp, t_grid, ls_pin=1.0, var_pin=1.0):
    """Build EFGP grid + Toeplitz at the pinned hyper used for q(f) reference."""
    X_template = (jnp.linspace(-3., 3., 16)[:, None] * jnp.ones((1, D)))
    X_extent = float((X_template.max(axis=0) - X_template.min(axis=0)).max())
    xcen = jnp.asarray(0.5 * (X_template.min(axis=0) + X_template.max(axis=0)),
                       dtype=jnp.float32)
    grid = jp.spectral_grid_se_fixed_K(
        log_ls=jnp.log(jnp.asarray(ls_pin, dtype=jnp.float32)),
        log_var=jnp.log(jnp.asarray(var_pin, dtype=jnp.float32)),
        K_per_dim=K_FD, X_extent=X_extent, xcen=xcen, d=D, eps=1e-3,
    )
    ms = mp['m'][0]; Ss = mp['S'][0]; SSs = mp['SS'][0]
    del_t = t_grid[1:] - t_grid[:-1]
    mu_r_init, _, top = jpd.compute_mu_r_jax(
        ms, Ss, SSs, del_t, grid, jr.PRNGKey(99),
        sigma_drift_sq=SIGMA ** 2, S_marginal=8,
        D_lat=D, D_out=D, cg_tol=1e-9, max_cg_iter=4 * grid.M,
    )
    ws_real_init = grid.ws.real.astype(grid.ws.dtype)
    ws_safe = jnp.where(jnp.abs(ws_real_init) < 1e-30,
                        jnp.array(1e-30, dtype=ws_real_init.dtype),
                        ws_real_init)
    A_init = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    h_r0 = jax.vmap(A_init)(mu_r_init)
    z_r = h_r0 / ws_safe
    return grid, top, z_r


def loss_at(log_ls, log_var, *, grid, top, z_r, D_out):
    """Collapsed M-step loss at refreshed μ via dense linear solve."""
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
        term1 = jnp.vdot(h[r], mu[r]).real
        term2 = jnp.vdot(mu[r], A @ mu[r]).real * 0.5
        return -(term1 - term2)

    det_loss = jax.vmap(per_r)(jnp.arange(D_out)).sum()
    sign, logdet = jnp.linalg.slogdet(A)
    return float(det_loss + 0.5 * D_out * logdet.real)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="duffing",
                         choices=["linear", "duffing", "anharmonic"])
    parser.add_argument("--out", default=None,
                         help="Path to save .npz (default: /tmp/T7_<regime>_landscape.npz)")
    parser.add_argument("--png", default=None,
                         help="Path to save contour PNG (default: /tmp/T7_<regime>_landscape.png)")
    args = parser.parse_args()
    regime = args.regime
    out_npz = args.out or f"/tmp/T7_{regime}_landscape.npz"
    out_png = args.png or f"/tmp/T7_{regime}_landscape.png"

    print(f"[T7] regime: {regime}", flush=True)
    print(f"[T7] JAX devices: {jax.devices()}", flush=True)
    print(f"[T7] Computing collapsed M-step loss surface on a "
          f"{len(LOG_LS_GRID)}×{len(LOG_VAR_GRID)} grid", flush=True)

    xs, ys, lik, t_max = setup(regime)
    print(f"[T7] Converging shared smoother (SparseGP fixed-hyper) for {regime}...",
          flush=True)
    mp, t_grid = converge_smoother(lik, t_max)
    print("[T7] Building EFGP M-step state at ℓ_pin=1.0...", flush=True)
    grid, top, z_r = build_state(mp, t_grid, ls_pin=1.0, var_pin=1.0)
    print(f"[T7] grid: K={K_FD}, M={int(grid.M)}, h={float(grid.h_per_dim[0]):.4f}",
          flush=True)

    L_grid = np.zeros((len(LOG_LS_GRID), len(LOG_VAR_GRID)))
    print("[T7] Computing loss landscape...", flush=True)
    for i, ll in enumerate(LOG_LS_GRID):
        for j, lv in enumerate(LOG_VAR_GRID):
            L_grid[i, j] = loss_at(ll, lv, grid=grid, top=top, z_r=z_r, D_out=D)
        print(f"  row {i+1}/{len(LOG_LS_GRID)}: ll={ll:+.2f}, "
              f"L range [{L_grid[i].min():.2f}, {L_grid[i].max():.2f}]",
              flush=True)

    # Identify minimum
    idx_min = np.unravel_index(np.argmin(L_grid), L_grid.shape)
    ll_min = LOG_LS_GRID[idx_min[0]]
    lv_min = LOG_VAR_GRID[idx_min[1]]
    L_min = L_grid[idx_min]
    print(f"\n[T7] Loss minimum on grid:", flush=True)
    print(f"  log_ℓ={ll_min:+.3f} (ℓ={math.exp(ll_min):.3f}), "
          f"log_σ²={lv_min:+.3f} (σ²={math.exp(lv_min):.3f}), "
          f"α=σ²/ℓ²={math.exp(lv_min - 2*ll_min):.3f}, "
          f"L={L_min:.4f}", flush=True)

    # Examine ridge structure: find points within 1.0 of the minimum
    ridge_mask = (L_grid - L_min) < 1.0
    print(f"\n[T7] Ridge structure (loss within 1.0 of minimum):", flush=True)
    print(f"  total ridge cells: {int(ridge_mask.sum())} / "
          f"{L_grid.size} ({100*ridge_mask.sum()/L_grid.size:.0f}%)", flush=True)

    # If ridge follows constant σ²/ℓ², log_σ² - 2*log_ℓ should be near-constant
    rows, cols = np.where(ridge_mask)
    alphas_in_ridge = LOG_VAR_GRID[cols] - 2 * LOG_LS_GRID[rows]
    print(f"  α=σ²/ℓ² in ridge:  min={math.exp(alphas_in_ridge.min()):.3f}, "
          f"max={math.exp(alphas_in_ridge.max()):.3f}, "
          f"std={alphas_in_ridge.std():.4f}", flush=True)
    print(f"  log_ℓ in ridge: min={LOG_LS_GRID[rows].min():+.2f}, "
          f"max={LOG_LS_GRID[rows].max():+.2f}, "
          f"range={LOG_LS_GRID[rows].max() - LOG_LS_GRID[rows].min():.2f}",
          flush=True)
    print(f"  log_σ² in ridge: min={LOG_VAR_GRID[cols].min():+.2f}, "
          f"max={LOG_VAR_GRID[cols].max():+.2f}, "
          f"range={LOG_VAR_GRID[cols].max() - LOG_VAR_GRID[cols].min():.2f}",
          flush=True)

    # Compare loss at the EFGP and SparseGP n_em=100 endpoints from T2a
    efgp_endpoint = (math.log(1.413), math.log(1.262))
    sp_endpoint = (math.log(0.914), math.log(0.533))
    L_efgp = loss_at(efgp_endpoint[0], efgp_endpoint[1],
                      grid=grid, top=top, z_r=z_r, D_out=D)
    L_sp = loss_at(sp_endpoint[0], sp_endpoint[1],
                    grid=grid, top=top, z_r=z_r, D_out=D)
    print(f"\n[T7] Loss at T2a endpoints (linear, n_em=100):", flush=True)
    print(f"  EFGP (ℓ=1.413, σ²=1.262):    L={L_efgp:.4f}  "
          f"(Δ from min: {L_efgp - L_min:+.4f})", flush=True)
    print(f"  Sparse (ℓ=0.914, σ²=0.533):  L={L_sp:.4f}  "
          f"(Δ from min: {L_sp - L_min:+.4f})", flush=True)
    print(f"  Difference EFGP - Sparse:    {L_efgp - L_sp:+.4f}", flush=True)

    # Save the grid for later plotting
    np.savez(out_npz,
              log_ls_grid=LOG_LS_GRID, log_var_grid=LOG_VAR_GRID,
              L_grid=L_grid, L_min=L_min, ll_min=ll_min, lv_min=lv_min,
              efgp_endpoint=efgp_endpoint, sp_endpoint=sp_endpoint,
              regime=regime)
    print(f"\n[T7] Saved {out_npz}", flush=True)

    # Render contour plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6.5))
        L_norm = L_grid - L_min
        # Log-spaced contour levels to show structure across many orders
        levels = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        # Cap the colormap range so the basin doesn't drown the rest
        cf = ax.contourf(LOG_LS_GRID, LOG_VAR_GRID, L_norm.T,
                          levels=[0] + levels, cmap="viridis_r", extend="max")
        cb = fig.colorbar(cf, ax=ax, label="L − L_min")
        cs = ax.contour(LOG_LS_GRID, LOG_VAR_GRID, L_norm.T,
                        levels=levels, colors="k", linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
        ax.scatter([ll_min], [lv_min], marker="*", s=220,
                    color="gold", edgecolor="k", zorder=5,
                    label=f"loss min: ℓ={math.exp(ll_min):.2f}, σ²={math.exp(lv_min):.2f}, α={math.exp(lv_min - 2*ll_min):.2f}")
        ax.scatter([efgp_endpoint[0]], [efgp_endpoint[1]],
                    marker="o", s=90, color="C0", edgecolor="k", zorder=5,
                    label=f"EFGP n_em=100 (linear): ℓ={math.exp(efgp_endpoint[0]):.2f}, σ²={math.exp(efgp_endpoint[1]):.2f}")
        ax.scatter([sp_endpoint[0]], [sp_endpoint[1]],
                    marker="s", s=90, color="C1", edgecolor="k", zorder=5,
                    label=f"SparseGP n_em=100 (linear): ℓ={math.exp(sp_endpoint[0]):.2f}, σ²={math.exp(sp_endpoint[1]):.2f}")
        # constant-α lines: log_σ² = 2 log_ℓ + log α
        for log_alpha_show in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            ax.plot(LOG_LS_GRID, LOG_LS_GRID * 2 + log_alpha_show,
                    color="white", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("log ℓ")
        ax.set_ylabel("log σ²")
        ax.set_xlim(LOG_LS_GRID.min(), LOG_LS_GRID.max())
        ax.set_ylim(LOG_VAR_GRID.min(), LOG_VAR_GRID.max())
        ax.set_title(f"EFGP collapsed M-step loss landscape ({regime})\n"
                      f"contours = L − L_min  (white dotted: constant α=σ²/ℓ²)")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"[T7] Saved {out_png}", flush=True)
    except Exception as e:
        print(f"[T7] Plot rendering failed: {e}", flush=True)

    # Print loss along constant-α slice through the minimum
    target_alpha = math.exp(lv_min - 2 * ll_min)
    print(f"\n[T7] Loss along constant-α={target_alpha:.3f} slice:", flush=True)
    print(f"  {'log_ℓ':>7s}  {'log_σ²':>7s}  {'L':>10s}  {'L - L_min':>10s}",
          flush=True)
    for ll in LOG_LS_GRID:
        lv = ll * 2 + math.log(target_alpha)
        if LOG_VAR_GRID.min() <= lv <= LOG_VAR_GRID.max():
            L_at = loss_at(ll, lv, grid=grid, top=top, z_r=z_r, D_out=D)
            print(f"  {ll:>+7.3f}  {lv:>+7.3f}  {L_at:>10.4f}  "
                  f"{L_at - L_min:>+10.4f}", flush=True)

    # Compare: orthogonal-to-ridge slice (vary log_ℓ at fixed log_σ²)
    print(f"\n[T7] Loss along log_σ²={lv_min:+.3f} slice (orthogonal):", flush=True)
    print(f"  {'log_ℓ':>7s}  {'log_σ²':>7s}  {'L':>10s}  {'L - L_min':>10s}",
          flush=True)
    for ll in LOG_LS_GRID:
        L_at = loss_at(ll, lv_min, grid=grid, top=top, z_r=z_r, D_out=D)
        print(f"  {ll:>+7.3f}  {lv_min:>+7.3f}  {L_at:>10.4f}  "
              f"{L_at - L_min:>+10.4f}", flush=True)

    # Diagnostic: ratio of curvature along ridge vs orthogonal.
    # Sample 5 points around the minimum: along log_ℓ and along α.
    delta = 0.1
    L0 = loss_at(ll_min, lv_min, grid=grid, top=top, z_r=z_r, D_out=D)

    # Curvature along log_ℓ at fixed log_σ²
    L_p_ls = loss_at(ll_min + delta, lv_min,
                      grid=grid, top=top, z_r=z_r, D_out=D)
    L_m_ls = loss_at(ll_min - delta, lv_min,
                      grid=grid, top=top, z_r=z_r, D_out=D)
    curv_ls = (L_p_ls + L_m_ls - 2 * L0) / (delta ** 2)

    # Curvature along log_σ² at fixed log_ℓ
    L_p_var = loss_at(ll_min, lv_min + delta,
                       grid=grid, top=top, z_r=z_r, D_out=D)
    L_m_var = loss_at(ll_min, lv_min - delta,
                       grid=grid, top=top, z_r=z_r, D_out=D)
    curv_var = (L_p_var + L_m_var - 2 * L0) / (delta ** 2)

    # Curvature along α-direction (orthogonal to ridge): increment (log_ℓ, log_σ²) by (-1, -2)/√5 * delta
    # Actually let's use the constant-α direction (1, 2)/√5 for "along ridge",
    # and the orthogonal (2, -1)/√5 for "across ridge"
    along_dir = np.array([1.0, 2.0]) / math.sqrt(5)
    across_dir = np.array([2.0, -1.0]) / math.sqrt(5)

    L_p_along = loss_at(ll_min + delta * along_dir[0],
                         lv_min + delta * along_dir[1],
                         grid=grid, top=top, z_r=z_r, D_out=D)
    L_m_along = loss_at(ll_min - delta * along_dir[0],
                         lv_min - delta * along_dir[1],
                         grid=grid, top=top, z_r=z_r, D_out=D)
    curv_along = (L_p_along + L_m_along - 2 * L0) / (delta ** 2)

    L_p_across = loss_at(ll_min + delta * across_dir[0],
                          lv_min + delta * across_dir[1],
                          grid=grid, top=top, z_r=z_r, D_out=D)
    L_m_across = loss_at(ll_min - delta * across_dir[0],
                          lv_min - delta * across_dir[1],
                          grid=grid, top=top, z_r=z_r, D_out=D)
    curv_across = (L_p_across + L_m_across - 2 * L0) / (delta ** 2)

    print(f"\n[T7] Curvatures around minimum (delta={delta}):", flush=True)
    print(f"  ∂²L/∂(log_ℓ)²       = {curv_ls:.4f}", flush=True)
    print(f"  ∂²L/∂(log_σ²)²      = {curv_var:.4f}", flush=True)
    print(f"  ∂²L along ridge (1,2)/√5    = {curv_along:.4f}", flush=True)
    print(f"  ∂²L across ridge (2,-1)/√5 = {curv_across:.4f}", flush=True)
    if abs(curv_along) > 0:
        ratio = curv_across / curv_along
        print(f"  ratio (across / along) = {ratio:.2f}", flush=True)
        if ratio > 5.0:
            print(f"  → STRONG ridge: across is {ratio:.0f}× steeper than along.",
                  flush=True)
            print(f"  → Individual hypers are NOT identifiable from this data; "
                  f"only α is.", flush=True)
            print(f"  → α-agreement is the appropriate alignment metric.",
                  flush=True)
        elif ratio > 2.0:
            print(f"  → MODERATE ridge: individual hypers are weakly identified.",
                  flush=True)
        else:
            print(f"  → NOT ridge-shaped: individual hypers should be "
                  f"recoverable; methods that disagree are biased.", flush=True)


if __name__ == "__main__":
    main()
