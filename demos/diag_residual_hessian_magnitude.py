"""
diag_residual_hessian_magnitude.py

Rigorous, crash-free measurement of the dropped transition S-gradient term
(the residual Hessian) RELATIVE to the production Gauss-Newton term, on a
FROZEN q(x) (an M-step dump).  No smoother, no EM loop -> cannot crash.

Question: is the missing term big enough to matter (like it must, to explain
the collapse), or negligible (like E[V], which was ~5e-5 and inert)?

The transition ELBO S-gradient (per transition), with Sigma^-1 = sigma^-2 I:
  production (Gauss-Newton, item iii kept):
      dS_GN   = -(Δ/2) σ⁻² E[J]ᵀ E[J]
  dropped (item ii's d·½E[H] + item iii residual Hessian), leading order:
      dS_miss = +½ σ⁻² Σ_r ( d_r − Δ E[f̄_r] ) E[H_r]
Report the per-transition ratio  ‖dS_miss‖ / ‖dS_GN‖  (σ⁻² cancels), its sign
(does the missing term NET REDUCE the precision -> inflate S?), and for contrast
the WRONG f̄-weighted version ‖Σ_r f̄_r E[H_r]‖ that was implemented.

All moments are EXACT Gaussian expectations via the characteristic function
(single spectral sums; no cholesky, no GH, no NUFFT):
  E[f̄_r]      = Re Σ_k μ_{r,k} D_k e^{2πi ξ_k·(m−xc)} e^{−2π² ξ_kᵀ S ξ_k}
  E[∂_j f̄_r]  = Re Σ_k μ ... (2πi ξ_{k,j})  · same envelope
  E[∂²_{jl} f̄_r] = Re Σ_k μ ... (2πi ξ_{k,j})(2πi ξ_{k,l}) · same envelope
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse, math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def reconstruct_mu_r(z, cdtype=jnp.complex128):
    """mu_r = A^{-1} h, A = I + diag(ws) T diag(ws), h = ws ⊙ z_r  (dense)."""
    from demos.replay_mstep_landscape import build_T_mat
    T = build_T_mat(z['top_v_fft'], z['top_ns'])                  # (M,M)
    ws = jnp.asarray(z['ws_real']).astype(cdtype)                 # (M,) = D_theta
    zr = jnp.asarray(z['z_r']).astype(cdtype)                     # (D_out, M)
    M = T.shape[0]
    A = jnp.eye(M, dtype=cdtype) + ws[:, None] * T * ws[None, :]
    h = ws[None, :] * zr                                          # (D_out, M)
    mu = jnp.linalg.solve(A, h.T).T                               # (D_out, M)
    return mu, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--n-sub', type=int, default=1500)
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)
    D = int(z['D'])
    xis = jnp.asarray(z['xis_flat'])                              # (M, D)
    xcen = jnp.asarray(z['xcen'])
    sig2 = float(z['sigma_drift_sq'])
    mu_r, ws = reconstruct_mu_r(z)                                # (D_out,M),(M,)
    fk = ws[None, :] * mu_r                                       # (D_out, M) complex
    two_pi = 2.0 * math.pi
    xixi = (xis[:, :, None] * xis[:, None, :])                    # (M,D,D) real

    m_src = jnp.asarray(z['m_src']); S_src = jnp.asarray(z['S_src'])
    d_src = jnp.asarray(z['d_src'])                               # mean increment
    w_src = jnp.asarray(z['w_src'])                               # del_t*mask
    # del_t: recover from weights (nonzero entries all equal del_t)
    del_t = float(jnp.max(w_src))
    N = m_src.shape[0]
    rng = np.random.default_rng(0)
    # only real (unmasked) transitions
    valid = np.where(np.asarray(w_src) > 0)[0]
    idx = rng.choice(valid, size=min(args.n_sub, valid.size), replace=False)

    def moments(m, S):
        env = jnp.exp(-2.0 * (math.pi ** 2)
                      * jnp.einsum('kj,jl,kl->k', xis, S, xis))   # (M,)
        ph = jnp.exp(2j * math.pi * (xis @ (m - xcen)))           # (M,)
        base = fk * (ph * env)[None, :]                           # (D_out,M)
        Ef = jnp.real(base.sum(1))                                # (D_out,)
        EJ = jnp.real(base @ (1j * two_pi * xis))                 # (D_out,D)
        EH = -(two_pi ** 2) * jnp.real(
            jnp.einsum('rk,kjl->rjl', base, xixi.astype(base.dtype)))  # (D_out,D,D)
        return Ef, EJ, EH

    Ef, EJ, EH = jax.vmap(moments)(m_src[idx], S_src[idx])        # batched
    dsub = d_src[idx]                                             # (n,D)

    # dS_GN  = (Δ/2) E[J]ᵀE[J]   (magnitude; drop σ⁻², sign)
    GN = jnp.einsum('nrj,nrl->njl', EJ, EJ) * (del_t / 2.0)      # (n,D,D)
    # dS_miss = ½ Σ_r (d_r − Δ Ef_r) E[H_r]      (leading-order, fit-residual wtd)
    resid = dsub - del_t * Ef                                    # (n,D_out)
    MISS = 0.5 * jnp.einsum('nr,nrjl->njl', resid, EH)           # (n,D,D)
    # WRONG version I implemented: ½ Σ_r f̄_r E[H_r] (f̄-weighted)  [for contrast]
    WRONG = 0.5 * del_t * jnp.einsum('nr,nrjl->njl', Ef, EH)

    def frob(A): return jnp.sqrt(jnp.sum(A * A, axis=(-1, -2)))
    def mineig(A):
        As = 0.5 * (A + jnp.swapaxes(A, -1, -2))
        return jnp.linalg.eigvalsh(As)[..., 0]

    r_miss = np.asarray(frob(MISS) / (frob(GN) + 1e-30))
    r_wrong = np.asarray(frob(WRONG) / (frob(GN) + 1e-30))
    # net effect on precision: precision_missing = -dS_miss (the -2*∂_S...).
    # If MISS (∂_S) is POSITIVE-def -> reduces precision -> inflates S. Check
    # the sign of the trace of MISS relative to GN's ∂_S sign.
    # ∂_S_GN = -(Δ/2)E[J]ᵀE[J] is NEGATIVE-def. ∂_S_miss = MISS (any sign).
    # Net ∂_S = ∂_S_GN + MISS. Precision ∝ -∂_S. S inflates where MISS is
    # POSITIVE-def (makes ∂_S less negative -> precision smaller).
    miss_min = np.asarray(mineig(MISS)); miss_max = np.asarray(mineig(-MISS))  # -min(-M)=maxeig
    frac_posdef = float(np.mean(miss_min > 0))
    frac_inflating = float(np.mean(np.asarray(jnp.trace(MISS, axis1=-1, axis2=-2)) > 0))

    print(f"=== residual-Hessian magnitude on {os.path.basename(args.dump)} ===")
    print(f"  n_sub={len(idx)}  del_t={del_t:.4g}  sig2={sig2:.4g}")
    print(f"  CORRECT fit-residual term ‖dS_miss‖/‖dS_GN‖:")
    print(f"     median={np.median(r_miss):.3g}  mean={np.mean(r_miss):.3g}  "
          f"p90={np.percentile(r_miss,90):.3g}  max={np.max(r_miss):.3g}")
    print(f"  WRONG f̄-weighted term (what was coded) ‖·‖/‖dS_GN‖:")
    print(f"     median={np.median(r_wrong):.3g}  mean={np.mean(r_wrong):.3g}  "
          f"p90={np.percentile(r_wrong,90):.3g}  max={np.max(r_wrong):.3g}")
    print(f"  fraction of transitions where MISS is PSD (would inflate S in all dirs): {frac_posdef:.2f}")
    print(f"  fraction with tr(MISS)>0 (net S-inflating): {frac_inflating:.2f}")
    print(f"  interpretation: ratio ~O(1) => term matters; ~1e-3 => inert like E[V].")


if __name__ == '__main__':
    main()
