"""EXPERIMENT: Matheron/pathwise restoration of the drift-variance V in the
keep-all autodiff E-step. Draw S posterior weight samples w_s ~ CN(mu, A^-1)
via Matheron (prior sample at latent draws + A^-1 solve); the sample
autocorrelation of (mu+dw) gives rho+omega = coeff of E_q[fbar^2 + V].
Drop-in replacement for the rho used by gmix_E_full_Eff. (Ignores nothing;
folds V in exactly in expectation.)
"""
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jr
import sing.efgp_jax_primitives as jp
from sing.efgp_gmix_qx_moments import _autocorr_per_r, GmixQxAux, _delta_grid_for_autocorr

def _jacobi(ws, top):
    N = 1.0
    for n in top.fft_shape: N *= n
    Tdiag = (jnp.sum(top.v_fft) / N).real
    Minv = 1.0 / (1.0 + ws.real**2 * Tdiag)
    return Minv

def sample_dw(mu_r, grid, top, m_src, S_src, weights, key, S,
              cg_tol=1e-6, max_iter=2000):
    """δw ~ CN(0, A^{-1}), A = I + D T D, via Matheron prior samples. (S, M)."""
    cdtype = grid.ws.dtype; ws = grid.ws.real; xcen = grid.xcen; h = grid.h_per_dim
    mtot = tuple(int(n) for n in grid.mtot_per_dim); M = grid.M; Ns = m_src.shape[0]
    A_apply = jp.make_A_apply(grid.ws, top, sigmasq=1.0)
    Minv_diag = _jacobi(ws, top).astype(cdtype)
    L = jnp.linalg.cholesky(S_src + 1e-9 * jnp.eye(S_src.shape[-1]))
    sqrt_w = jnp.sqrt(weights).astype(cdtype)
    def one(k):
        k1, k2, k3, k4 = jr.split(k, 4)
        xs = m_src + jnp.einsum('nij,nj->ni', L, jr.normal(k1, m_src.shape))
        g = (jr.normal(k2, (Ns,)) + 1j * jr.normal(k3, (Ns,))) / jnp.sqrt(2.)
        d = jp.nufft1(xs, sqrt_w * g, xcen, h, out_shape=mtot, eps=6e-8).reshape(-1)
        xi = (jr.normal(k4, (M,)) + 1j * jr.normal(jr.fold_in(k4, 1), (M,))) / jnp.sqrt(2.)
        b = xi + ws.astype(cdtype) * d
        return jp.cg_solve(A_apply, b, tol=cg_tol, max_iter=max_iter,
                           M_inv_apply=lambda v: Minv_diag * v)
    return jax.vmap(one)(jr.split(key, S))

def rho_plus_omega_aux(mu_r, delta_w, grid):
    """GmixQxAux whose rho_summed = (1/S)Σ_s Σ_r autocorr((mu_r+δw_s)·ws)
    = coefficient block of E_q[fbar^T fbar + V]."""
    ws = grid.ws.real; D_out = mu_r.shape[0]; S = delta_w.shape[0]
    w = mu_r[:, None, :] + delta_w[None, :, :]              # (D_out, S, M)
    wf = w.reshape(D_out * S, grid.M)
    ac = _autocorr_per_r(wf, ws, grid.mtot_per_dim)         # (D_out*S, *pad)
    ac = ac.reshape(D_out, S, *ac.shape[1:]).mean(1).sum(0) # (*pad)
    return GmixQxAux(rho_summed=ac.reshape(-1),
                     delta_flat=_delta_grid_for_autocorr(grid))
