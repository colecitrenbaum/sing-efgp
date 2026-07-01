"""Neural-data line-attractor head-to-head: EFGP-SING vs SparseGP-SING, RBF.

Reproduces the gpSLDS line-attractor analysis of ``neural_data_gpslds.ipynb``
(Vinograd et al. 2024 calcium data; Hu et al. 2024 gpSLDS) but with a generic
**RBF (SE) GP drift** instead of the SSL kernel, fit two ways that share the
SING natural-gradient backbone:

    EFGP-SING     — random-feature GP, closed-form gmix q(f) update
                    (sing.efgp_em.fit_efgp_sing_jax)
    SparseGP-SING — inducing-point GP, Gauss-Hermite expectations
                    (sing.sing.fit_variational_em)

Both get the SAME data, the SAME 2 step-function inputs (intruder entrances)
with a learned input-effect matrix B, and emissions FIXED at the PCA init
(``learn_(output)_emissions=False``) so the two latent posteriors live in one
shared basis and the drift fields are directly comparable.

Scientific question: does a plain RBF GP drift — under either inference method
— still surface the line-attractor structure the SSL kernel was designed for?

Deliverable figure: two side-by-side panels (EFGP | SparseGP), each showing the
posterior drift quiver, the inferred latent trajectory, a speed-based
"slow-point" map (||f|| ~ 0, the approximate line attractor), and the learned
input-effect arrows at the two intruder onsets — mirroring notebook cell 37.

NOTE on the slow-point map: the notebook weights it by the SparseGP posterior
drift *variance*.  Here we use a method-agnostic SPEED proxy ``||f(x)||~0`` for
BOTH methods so the comparison is apples-to-apples (EFGP's posterior drift
variance V(x) needs the heavier restore_qf_variance machinery; out of MVP
scope).

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/bench_neural_efgp_vs_sparsegp.py

First run downloads the DANDI calcium trace via ``lindi`` (install with
``/Users/colecitrenbaum/myenv/bin/pip install lindi``) and caches the
z-scored array; later runs read the cache (no network).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
# fp64 — mandatory for SING at K*T this large (CLAUDE.md): the block-tridiag
# smoother + SparseGP M-step NaN in fp32 above K*T ~ a few thousand.
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.likelihoods import Likelihood, Gaussian
from sing.inputs import InputSignals
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_params_pca


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D = 2                     # latent dim (matches notebook)
RATE = 10                 # Hz
DT = 1.0 / RATE           # 0.1 s per bin
SIGMA = 1.0               # SDE diffusion scale (notebook gpSLDS uses unit diff.)

# T is large (4201) and K=1 -> SparseGP's sequential smoother scan is the
# bottleneck.  Subsample in time for a tractable first pass; set to 1 for the
# full-resolution run.
SUBSAMPLE_T = 2

# Shared SING schedule (CLAUDE.md head-to-head defaults, trimmed n_em for the
# long single trial).
N_EM = 40
N_ESTEP = 10
N_MSTEP_INNER = 4
MSTEP_LR = 0.01

# RBF hyper init (learned during the fit).  ell is set from the latent extent
# in main(); variance starts at 1.
VAR_INIT = 1.0

# Intruder-entrance input windows (seconds), from the notebook.
INPUT1_TIMES = (74.0, 80.0)
INPUT2_TIMES = (207.0, 212.0)

DANDI_URL = ('https://api.dandiarchive.org/api/assets/'
             '1e11c74e-6f25-4604-9216-5b861fec8f1c/download/')

OUT_DIR = Path(__file__).resolve().parent / "_bench_neural_efgp_vs_sparsegp_out"
OUT_DIR.mkdir(exist_ok=True)
CACHE = (Path(__file__).resolve().parent / "_neural_data_cache"
         / "neural_zscored.npz")


# ---------------------------------------------------------------------------
# Gaussian likelihood shim for the EFGP path (closed-form ell)
# ---------------------------------------------------------------------------
class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_neural_data():
    """Return z-scored neural trace (T, N) and the time grid, cached locally."""
    if CACHE.exists():
        d = np.load(CACHE)
        print(f"  [data] loaded cache {CACHE}  shape={d['norm_neural'].shape}")
        return d['norm_neural']
    print("  [data] downloading DANDI calcium trace via lindi ...")
    import lindi
    f = lindi.LindiH5pyFile.from_hdf5_file(DANDI_URL)
    neural = f['/processing/ophys/NeuralTrace']['data'][:]      # (T, N)
    norm_neural = (neural - neural.mean(0)) / neural.std(0)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, norm_neural=norm_neural)
    print(f"  [data] cached {CACHE}  shape={norm_neural.shape}")
    return norm_neural


def build_inputs(n_timesteps):
    inputs = np.zeros((1, n_timesteps, 2))
    i1 = (np.array(INPUT1_TIMES) * RATE / SUBSAMPLE_T).astype(int)
    i2 = (np.array(INPUT2_TIMES) * RATE / SUBSAMPLE_T).astype(int)
    inputs[0, i1[0]:i1[1] + 1, 0] = 1.0
    inputs[0, i2[0]:i2[1] + 1, 1] = 1.0
    onset1 = int(i1[0])
    onset2 = int(i2[0])
    return jnp.asarray(inputs), onset1, onset2


# ---------------------------------------------------------------------------
# EFGP fit + drift-field evaluation
# ---------------------------------------------------------------------------
def fit_efgp(ys, inputs, output_params, x0, t_grid, X_template, ls_init):
    T = ys.shape[1]
    lik = GLik(ys, jnp.ones((1, T), dtype=bool))
    ip = dict(mu0=x0, V0=jnp.eye(D)[None])
    rho = jnp.linspace(0.05, 0.7, N_EM)
    t0 = time.time()
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(output_params), init_params=ip, latent_dim=D,
        lengthscale=ls_init, variance=VAR_INIT, sigma=SIGMA,
        sigma_drift_sq=SIGMA ** 2, eps_grid=1e-2,
        estep_method='gmix',
        n_em_iters=N_EM, n_estep_iters=N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=N_MSTEP_INNER, mstep_lr=MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        input_signals=InputSignals(inputs),
        learn_input_effect=True, input_effect_warmup_iters=8,
        X_template=X_template,
        verbose=True,
    )
    wall = time.time() - t0
    return mp, hist, wall


def efgp_drift_field(mp, ls, var, t_grid, X_template, X_eval):
    """Posterior drift mean of the final EFGP q(f) at points X_eval (M, D)."""
    ms = jnp.asarray(mp['m']); Ss = jnp.asarray(mp['S']); SSs = jnp.asarray(mp['SS'])
    trial_mask = jnp.ones(ms.shape[:2], dtype=bool)
    del_t = t_grid[1:] - t_grid[:-1]
    grid = jp.spectral_grid_se(ls, var, X_template, eps=1e-2)
    m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
        ms, Ss, SSs, del_t, trial_mask)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_src, C_src, w_src, grid,
        sigma_drift_sq=SIGMA ** 2, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(X_eval)[None], D_lat=D, D_out=D)
    return np.asarray(Ef[0])                                  # (M, D)


# ---------------------------------------------------------------------------
# SparseGP fit + drift-field evaluation
# ---------------------------------------------------------------------------
def _data_aware_zs(num_per_dim, lo, hi):
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim) for d in range(D)]
    return jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'),
                     axis=-1).reshape(-1, D)


def fit_sparsegp(ys, inputs, output_params, x0, t_grid, lo, hi, ls_init,
                 num_per_dim=8):
    T = ys.shape[1]
    lik = Gaussian(ys, jnp.ones((1, T), dtype=jnp.float64))
    ip = dict(mu0=x0, V0=jnp.eye(D)[None])
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = _data_aware_zs(num_per_dim, lo, hi)
    print(f"  [SparseGP] {zs.shape[0]} inducing pts "
          f"({num_per_dim}x{num_per_dim} over latent bbox)")
    fn = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    # RBF stores output_scale = sqrt(variance) (CLAUDE.md gotcha).
    drift_params = dict(length_scales=jnp.full((D,), float(ls_init)),
                        output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho = jnp.linspace(0.05, 0.7, N_EM)
    lr = jnp.full((N_EM,), MSTEP_LR)
    hist = []
    t0 = time.time()
    (mp, _, gp_post, dp, _, op, input_effect, elbos) = fit_variational_em(
        key=jr.PRNGKey(33), fn=fn, likelihood=lik, t_grid=t_grid,
        drift_params=drift_params, init_params=ip,
        output_params=dict(output_params),
        input_signals=InputSignals(inputs),
        sigma=SIGMA, rho_sched=rho,
        n_iters=N_EM, n_iters_e=N_ESTEP, n_iters_m=N_MSTEP_INNER,
        perform_m_step=True, learn_output_params=False,
        learning_rate=lr, print_interval=5,
        drift_params_history=hist)
    wall = time.time() - t0
    ls_hist = [float(jnp.exp(jnp.mean(jnp.log(d['length_scales'])))) for d in hist]
    var_hist = [float(d['output_scale']) ** 2 for d in hist]
    return mp, fn, dp, gp_post, np.asarray(input_effect), elbos, ls_hist, var_hist, wall


def sparsegp_drift_field(fn, dp, gp_post, X_eval):
    return np.asarray(fn.get_posterior_f_mean(gp_post, dp, jnp.asarray(X_eval)))


# ---------------------------------------------------------------------------
# Plotting (identical treatment for both methods)
# ---------------------------------------------------------------------------
def _slow_prob(speed, eps):
    """Speed-based slow-point proxy in [0,1]: ~1 where ||f|| << eps."""
    return np.exp(-0.5 * (speed / eps) ** 2)


def attractor_panel(ax, drift_fn, m_latent, B, onsets, lo, hi, eps,
                    n_q=22, n_slow=60, title=""):
    gx = np.linspace(lo[0], hi[0], n_q)
    gy = np.linspace(lo[1], hi[1], n_q)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    Fq = drift_fn(np.stack([GX.ravel(), GY.ravel()], -1)).reshape(n_q, n_q, D)

    sx = np.linspace(lo[0], hi[0], n_slow)
    sy = np.linspace(lo[1], hi[1], n_slow)
    SX, SY = np.meshgrid(sx, sy, indexing='ij')
    Fs = drift_fn(np.stack([SX.ravel(), SY.ravel()], -1))
    speed = np.linalg.norm(Fs, axis=-1).reshape(n_slow, n_slow)

    im = ax.imshow(_slow_prob(speed, eps).T, origin='lower',
                   extent=[lo[0], hi[0], lo[1], hi[1]], cmap='Purples',
                   vmin=0, vmax=1, alpha=0.75, aspect='auto',
                   interpolation='bilinear')
    ax.plot(m_latent[:, 0], m_latent[:, 1], color='tab:blue', lw=0.8, alpha=0.5)
    ax.quiver(GX, GY, Fq[..., 0], Fq[..., 1], angles='xy', color='black',
              alpha=0.85, scale_units='xy')
    # Input-effect arrows at the two intruder onsets.
    for j, t_on in enumerate(onsets):
        x0 = m_latent[t_on]
        ax.quiver(x0[0], x0[1], B[0, j], B[1, j], angles='xy',
                  scale_units='xy', scale=1, color='darkorange', width=0.012,
                  zorder=5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    return im


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"\n=== Neural line-attractor: EFGP vs SparseGP (RBF) "
          f"[subsample T/{SUBSAMPLE_T}, n_em={N_EM}] ===")

    norm_neural = load_neural_data()[::SUBSAMPLE_T]            # (T, N)
    n_timesteps, n_neurons = norm_neural.shape
    t_grid = jnp.arange(n_timesteps) * (DT * SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])                        # (1, T, N)
    print(f"  data: T={n_timesteps}, N={n_neurons}, dt={DT*SUBSAMPLE_T:.2f}s")

    inputs, onset1, onset2 = build_inputs(n_timesteps)
    onsets = (onset1, onset2)

    # PCA-init emissions (FIXED) + initial latent state.
    output_params, x0 = initialize_params_pca(D, ys)
    # Latent bbox from the PCA scores (the convention initialize_params_pca uses).
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])  # (T,D)
    lo = xs_pca.min(0) - 1.0
    hi = xs_pca.max(0) + 1.0
    extent = float(np.max(hi - lo))
    ls_init = extent / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, D)))
    print(f"  latent bbox lo={lo.round(2)} hi={hi.round(2)}  ls_init={ls_init:.2f}")

    # ---- EFGP ----
    print("\n  fitting EFGP-SING (RBF, inputs + learned B)...")
    mp_e, hist_e, wall_e = fit_efgp(ys, inputs, output_params, x0, t_grid,
                                    X_template, ls_init)
    ls_e, var_e = float(hist_e.lengthscale[-1]), float(hist_e.variance[-1])
    B_e = hist_e.input_effect
    print(f"    EFGP wall={wall_e:.1f}s  ell={ls_e:.3f}  var={var_e:.3f}")
    efgp_fn = lambda X: efgp_drift_field(mp_e, ls_e, var_e, t_grid, X_template, X)

    # ---- SparseGP ----
    print("\n  fitting SparseGP-SING (RBF, inputs)...")
    (mp_s, fn_s, dp_s, gp_s, B_s, elbos_s,
     ls_hist_s, var_hist_s, wall_s) = fit_sparsegp(
        ys, inputs, output_params, x0, t_grid, lo, hi, ls_init)
    print(f"    SparseGP wall={wall_s:.1f}s  ell={ls_hist_s[-1]:.3f}  "
          f"var={var_hist_s[-1]:.3f}")
    sparse_fn = lambda X: sparsegp_drift_field(fn_s, dp_s, gp_s, X)

    m_e = np.asarray(mp_e['m'][0])
    m_s = np.asarray(mp_s['m'][0])

    # Slow-point speed threshold: ~3% of the median drift speed scale on the
    # quiver grid, shared by both panels.
    gx = np.linspace(lo[0], hi[0], 30); gy = np.linspace(lo[1], hi[1], 30)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    sp_scale = float(np.median(np.linalg.norm(efgp_fn(pts), axis=-1)))
    eps_slow = max(0.1 * sp_scale, 1e-3)
    print(f"  slow-point eps={eps_slow:.3f} (10% of median drift speed)")

    # ---- Save ----
    np.savez(OUT_DIR / "bench.npz",
             T=n_timesteps, N=n_neurons, subsample=SUBSAMPLE_T,
             wall_efgp=wall_e, wall_sparsegp=wall_s,
             ls_efgp=ls_e, var_efgp=var_e,
             ls_hist_efgp=np.asarray(hist_e.lengthscale),
             var_hist_efgp=np.asarray(hist_e.variance),
             ls_hist_sparsegp=np.asarray(ls_hist_s),
             var_hist_sparsegp=np.asarray(var_hist_s),
             B_efgp=B_e, B_sparsegp=B_s,
             m_efgp=m_e, m_sparsegp=m_s, xs_pca=xs_pca,
             elbos_sparsegp=np.asarray(elbos_s),
             lo=lo, hi=hi, onsets=np.asarray(onsets), eps_slow=eps_slow)

    # ---- Line-attractor figure (the deliverable) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    im0 = attractor_panel(axes[0], efgp_fn, m_e, B_e, onsets, lo, hi, eps_slow,
                          title=f"EFGP-SING (RBF)\nell={ls_e:.2f}, "
                                f"var={var_e:.2f}, {wall_e:.0f}s")
    im1 = attractor_panel(axes[1], sparse_fn, m_s, B_s, onsets, lo, hi, eps_slow,
                          title=f"SparseGP-SING (RBF)\nell={ls_hist_s[-1]:.2f}, "
                                f"var={var_hist_s[-1]:.2f}, {wall_s:.0f}s")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("slow-point proxy  (exp[-||f||^2/2eps^2])")
    fig.suptitle("Neural line-attractor: RBF GP drift, SING E-step\n"
                 "blue = inferred latents,  black = drift,  "
                 "orange = learned input effect B", fontsize=11)
    fig.savefig(OUT_DIR / "line_attractor.png", dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- Secondary: latent overlay + hyper traces ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(xs_pca[:, 0], xs_pca[:, 1], color='gray', lw=0.7, alpha=0.6,
                 label='PCA scores')
    axes[0].plot(m_e[:, 0], m_e[:, 1], color='tab:blue', lw=0.8, label='EFGP m')
    axes[0].plot(m_s[:, 0], m_s[:, 1], color='tab:red', lw=0.8, label='SparseGP m')
    axes[0].set_title("inferred latent trajectories")
    axes[0].set_xlabel(r"$x_1$"); axes[0].set_ylabel(r"$x_2$")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.asarray(hist_e.lengthscale), '-o', ms=3, label='EFGP', color='tab:blue')
    axes[1].plot(np.asarray(ls_hist_s), '-s', ms=3, label='SparseGP', color='tab:red')
    axes[1].set_title("lengthscale ell"); axes[1].set_xlabel("EM iter")
    axes[1].legend(fontsize=8)

    axes[2].plot(np.asarray(hist_e.variance), '-o', ms=3, label='EFGP', color='tab:blue')
    axes[2].plot(np.asarray(var_hist_s), '-s', ms=3, label='SparseGP', color='tab:red')
    axes[2].set_title("variance sigma_f^2"); axes[2].set_xlabel("EM iter")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics.png", dpi=120)
    plt.close(fig)

    print(f"\n  wrote {OUT_DIR}/line_attractor.png, diagnostics.png, bench.npz")
    print(f"  EFGP B=\n{np.asarray(B_e).round(3)}")
    print(f"  SparseGP B=\n{np.asarray(B_s).round(3)}")


if __name__ == "__main__":
    main()
