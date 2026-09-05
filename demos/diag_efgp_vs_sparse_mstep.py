"""
diag_efgp_vs_sparse_mstep.py

Term-by-term M-step comparison at a K=100 divergence point.  On the SAME frozen
q(x) (reconstructed from an M-step dump), compute the length-scale objective for:
  (1) EFGP   : collapsed spectral marginal likelihood  L_M(θ)      [top, z_r]
  (2) SparseGP: inducing-point drift ELBO  -(neg_CE+prior)(θ)      [build_objective]
  (3) GT      : exact pseudo-velocity GP evidence on the q(x) MEANS [gt_landscape]
and report argmin_ℓ + dL/dlogℓ at the oracle θ for each.

If SparseGP (2) stays near the GT MLE while EFGP (1) wants ℓ↓ on the SAME q(x),
the collapse is localized to the M-step OBJECTIVE (spectral vs inducing basis
bandwidth).  If both want ℓ↓, the M-step is not the differentiator -> look at
the E-step (drift moments / precision update).

Run on GPU (SparseGP neg_CE over K*T is heavy): see the sbatch.
"""
from __future__ import annotations
import os, sys, math, argparse
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from functools import partial

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.utils.sing_helpers import compute_neg_CE
import demos.bench_gpdrift_scaling as base   # _rbf helpers, D, gt landscape lives in bench_gpdrift_x64
import demos.bench_gpdrift_x64 as bench       # gt_landscape, _data_aware_zs
from demos.replay_mstep_landscape import build_T_mat
from sing.efgp_jax_drift import _ws_real_se


def reconstruct_mp(z):
    """Reconstruct (K,T,D) marginal_params m,S,SS from a flattened M-step dump.
    The dump stores sources for transitions i=0..T-2 (i.e. ms[:, :-1]); pad the
    final time point by repetition (1 of T points; negligible for the landscape)."""
    D = int(z['D'])
    m_src = np.asarray(z['m_src']); S_src = np.asarray(z['S_src'])
    C_src = np.asarray(z['C_src']); w_src = np.asarray(z['w_src'])
    del_t = float(np.max(w_src))
    N = m_src.shape[0]
    # infer K, T-1 : the dump is K*(T-1). We know T=1000 for the K-sweep.
    Tm1 = 999
    K = N // Tm1
    assert K * Tm1 == N, (N, K, Tm1)
    m = m_src.reshape(K, Tm1, D)
    S = S_src.reshape(K, Tm1, D, D)
    # SSs_T = C + S ; SSs = swapaxes ; (SING conv: SSs[t]=Cov(x_{t+1},x_t))
    SSs_T = C_src.reshape(K, Tm1, D, D) + S
    SSs = np.swapaxes(SSs_T, -1, -2)
    # pad final time point (repeat last) to length T
    m = np.concatenate([m, m[:, -1:]], axis=1)              # (K,T,D)
    S = np.concatenate([S, S[:, -1:]], axis=1)              # (K,T,D,D)
    return (jnp.asarray(m), jnp.asarray(S), jnp.asarray(SSs), K, Tm1 + 1,
            del_t, D)


def efgp_landscape(z, ls_grid):
    """EFGP L_M(ℓ) at fixed σ² from the dump's (top, z_r)."""
    T_mat = build_T_mat(z['top_v_fft'], z['top_ns'])
    z_r = jnp.asarray(z['z_r']); xis = jnp.asarray(z['xis_flat'])
    h = float(np.asarray(z['h_per_dim']).reshape(-1)[0])
    D = int(z['D']); M = T_mat.shape[0]; cdtype = T_mat.dtype
    eye = jnp.eye(M, dtype=cdtype)
    import jax.scipy.linalg as jla
    lv = float(z['log_var'])
    def loss(log_ls):
        ws = _ws_real_se(jnp.asarray(log_ls), jnp.asarray(lv), xis, h, D).astype(cdtype)
        A = eye + ws[:, None] * T_mat * ws[None, :]
        L = jnp.linalg.cholesky(A)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L).real))
        hh = ws[None, :] * z_r
        mu = jax.vmap(lambda b: jla.solve_triangular(
            L.conj().T, jla.solve_triangular(L, b, lower=True), lower=False))(hh)
        return -0.5 * jnp.sum(jnp.real(jnp.sum(jnp.conj(hh) * mu, -1))) + 0.5 * D * logdet
    loss = jax.jit(loss)
    vals = np.array([float(loss(math.log(l))) for l in ls_grid])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--M', type=int, default=49)
    ap.add_argument('--n-grid', type=int, default=21)
    ap.add_argument('--l-lo', type=float, default=0.15)
    ap.add_argument('--l-hi', type=float, default=4.0)
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)
    D = int(z['D'])
    ls_grid = np.exp(np.linspace(math.log(args.l_lo), math.log(args.l_hi), args.n_grid))
    log_var = float(z['log_var']); sig2 = float(z['sigma_drift_sq'])
    sigma = math.sqrt(sig2)
    m, S, SS, K, T, del_t, _ = reconstruct_mp(z)
    print(f"dump={os.path.basename(args.dump)} K={K} T={T} del_t={del_t:.4g} "
          f"sigma={sigma:.3g} log_var={log_var:.3f} (var={math.exp(log_var):.3f})",
          flush=True)
    t_grid = jnp.arange(T) * del_t
    trial_mask = jnp.ones((K, T), dtype=bool)
    inputs = jnp.zeros((K, T, 1)); input_effect = jnp.zeros((D, 1))
    mp = dict(m=m, S=S + 1e-5 * jnp.eye(D), SS=SS)

    # ---- (1) EFGP L_M(ℓ) from the dump ----
    efgp_vals = efgp_landscape(z, ls_grid)
    i_e = int(np.argmin(efgp_vals))
    print(f"\n(1) EFGP   L_M: argmin ℓ={ls_grid[i_e]:.3f}", flush=True)

    # ---- (3) GT exact evidence on q(x) MEANS (concatenate trials) ----
    LOGLS = np.log(ls_grid); LOGV = np.array([log_var])
    # gt on all trials' means concatenated as one big regression (per CLAUDE.md)
    mm = np.asarray(m)                                   # (K,T,D)
    # build per-trial gt and sum (independent trials share θ)
    gt_tot = np.zeros(len(ls_grid))
    for k in range(K):
        Lk = bench.gt_landscape(jnp.asarray(mm[k]), sigma, t_grid, LOGLS, list(LOGV))
        gt_tot += np.asarray(Lk).reshape(-1)
    i_g = int(np.argmin(gt_tot))
    print(f"(3) GT     evidence on q(x) means: argmin ℓ={ls_grid[i_g]:.3f}", flush=True)

    # ---- (3b) GT evidence on the TRUE latent paths (honest oracle MLE) ----
    try:
        import demos.bench_duffing_kscaling as dk
        xs_true = np.asarray(dk.make_data_K(K, T, 0)[0])            # (K,T,D)
        gt_true = np.zeros(len(ls_grid))
        for k in range(K):
            Lk = bench.gt_landscape(jnp.asarray(xs_true[k]), sigma, t_grid,
                                    LOGLS, list(LOGV))
            gt_true += np.asarray(Lk).reshape(-1)
        i_gt = int(np.argmin(gt_true))
        print(f"(3b) GT   evidence on TRUE paths (oracle MLE): argmin ℓ={ls_grid[i_gt]:.3f}",
              flush=True)
    except Exception as ex:
        print(f"(3b) GT-true skipped: {ex}", flush=True)
        gt_true = None

    # ---- (2) SparseGP inducing M-step objective on the SAME q(x) ----
    n_per = int(round(math.sqrt(args.M)))
    mm2 = np.asarray(m).reshape(-1, D)
    lo = mm2.min(0) - 0.4; hi = mm2.max(0) + 0.4
    per = [np.linspace(lo[d], hi[d], n_per) for d in range(D)]
    zs = jnp.asarray(np.stack(np.meshgrid(*per, indexing='ij'), -1).reshape(-1, D))
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    key0 = jr.PRNGKey(0)
    ip = dict(mu0=m[:, 0], V0=jnp.broadcast_to(jnp.eye(D), (K, D, D)))

    @jax.jit
    def Lsp(log_ls, marginal_params):
        dp = dict(length_scales=jnp.full((D,), jnp.exp(log_ls)),
                  output_scale=jnp.exp(0.5 * log_var))
        gp_post = sparse.update_dynamics_params(
            key0, t_grid, marginal_params, trial_mask, dp, inputs,
            input_effect, sigma)
        neg_CE = jax.vmap(partial(compute_neg_CE, t_grid, sparse, gp_post, dp,
                                  input_effect=input_effect, sigma=sigma))(
            ip, jr.split(key0, K), marginal_params, inputs, trial_mask).sum()
        prior = sparse.prior_term(dp, gp_post)
        return -(neg_CE + prior)

    sp_vals = np.array([float(Lsp(jnp.asarray(math.log(l)), mp)) for l in ls_grid])
    i_s = int(np.argmin(sp_vals))
    print(f"(2) SparseGP M={args.M}: argmin ℓ={ls_grid[i_s]:.3f}", flush=True)

    print("\n=== SUMMARY (argmin ℓ on the SAME K=100 q(x)) ===", flush=True)
    print(f"  EFGP spectral : {ls_grid[i_e]:.3f}", flush=True)
    print(f"  SparseGP M={args.M:<3d}: {ls_grid[i_s]:.3f}", flush=True)
    print(f"  GT exact      : {ls_grid[i_g]:.3f}", flush=True)
    print("  -> if SparseGP≈GT >> EFGP, the M-step OBJECTIVE (spectral basis) is the locus.", flush=True)
    np.savez(args.dump.replace('.npz', '_vs_sparse.npz'),
             ls_grid=ls_grid, efgp=efgp_vals, sparse=sp_vals, gt=gt_tot,
             argmin_efgp=ls_grid[i_e], argmin_sparse=ls_grid[i_s], argmin_gt=ls_grid[i_g])


if __name__ == '__main__':
    main()
