"""EXPERIMENTAL (worktree only): differentiable exact gmix drift moments,
no custom_vjp -- so jax.grad through compute_neg_CE_single keeps ALL
Price terms (Hessians + covariances), i.e. no statistical-linearisation
drops. Used to test 'keep all terms + PSD projection' vs the production
drop path. NOT for main.
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from jax import Array
import sing.efgp_jax_primitives as jp
from sing.sde import SDE
from sing.efgp_gmix_qx_moments import gmix_E_full_Eff, precompute_aux, GmixQxAux


def _coeff(m, S, grid):
    xis = grid.xis_flat                                  # (M,d) real
    ws = grid.ws.real                                    # (M,)
    phase = jnp.exp(2j * jnp.pi * (xis @ (m - grid.xcen)))  # (M,) xcen frame
    quad = jnp.einsum('md,de,me->m', xis, S, xis)        # ξ'Sξ
    env = jnp.exp(-2 * jnp.pi**2 * quad)                 # (M,)
    return (ws * env).astype(phase.dtype) * phase        # (M,) complex


def Ef_diff(m, S, mu_r, grid):
    """E_{q}[bar_f]  (D_out,), differentiable in (m,S)."""
    c = _coeff(m, S, grid)
    return (mu_r * c[None, :]).sum(-1).real              # (D_out,)


def Edfdx_diff(m, S, mu_r, grid):
    """E_{q}[J_bar_f]  (D_out, d), differentiable in (m,S)."""
    c = _coeff(m, S, grid)
    xis = grid.xis_flat
    fac = (2j * jnp.pi * xis)                            # (M,d)
    # (D_out, M, d) -> sum over M
    return jnp.einsum('rm,md->rd', mu_r * c[None, :], fac).real


@jax.tree_util.register_pytree_node_class
class FullGmixDrift(SDE):
    """f/ff/dfdx are plain differentiable closed forms -> autodiff keeps
    all terms (no linearisation)."""
    def __init__(self, *, latent_dim, grid, mu_r, aux):
        super().__init__(expectation=None, latent_dim=latent_dim)
        self._grid = grid; self._mu_r = mu_r; self._aux = aux
    def tree_flatten(self):
        return (self._mu_r, self._aux), (self.latent_dim, self._grid)
    @classmethod
    def tree_unflatten(cls, aux_, children):
        latent_dim, grid = aux_; mu_r, aux = children
        return cls(latent_dim=latent_dim, grid=grid, mu_r=mu_r, aux=aux)
    def drift(self, drift_params, x, t):
        return Ef_diff(x, jnp.zeros((self.latent_dim, self.latent_dim)), self._mu_r, self._grid)
    def f(self, drift_params, key, t, m, S, gp_post=None, *a, **k):
        return Ef_diff(m, S, self._mu_r, self._grid)
    def ff(self, drift_params, key, t, m, S, gp_post=None, *a, **k):
        import os as _os
        if _os.environ.get('KEEPALL_FF', 'full') == 'sq':
            ef = Ef_diff(m, S, self._mu_r, self._grid)      # single gmix contraction
            return (ef * ef).sum()                          # ||E[fbar]||^2, no autocorr
        return gmix_E_full_Eff(m - self._grid.xcen, S, self._mu_r, self._grid, self._aux)
    def dfdx(self, drift_params, key, t, m, S, gp_post=None, *a, **k):
        return Edfdx_diff(m, S, self._mu_r, self._grid)
