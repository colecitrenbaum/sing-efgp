"""Gmix-closed-form q(x)-side drift moments for SING-EFGP.

Replaces the linearised + custom-VJP shim path (FrozenEFGPDrift +
ef_with_jac_grad/eff_with_grads) with explicit Gaussian averages
under q(x_i) = N(m_i, S_i):

    E[bar_f]       = sum_k mu_{r,k} D_k * exp(2π i m^T ξ_k − 2π² ξ_k^T S ξ_k)
    E[J_bar_f]_{j} = sum_k mu_{r,k} (2π i ξ_{k,j}) D_k * exp(...)
    E[bar_f^T bar_f] = sum_δ ρ(δ) * exp(2π i m^T δ − 2π² δ^T S δ)

with ρ(δ) = sum_r sum_k (D_k μ_{r,k}) (D_{k+δ} μ_{r,k+δ})  -- spectral
autocorr, precomputed once per outer iter via FFT.

All three are smooth functions of (m, S); plain jax.grad gives
unbiased Bonnet/Price gradients of the Gaussian average. No custom
VJPs needed.
"""
from __future__ import annotations

# IMPORTANT: import jax_finufft FIRST via efgp_jax_primitives (load-order)
import sing.efgp_jax_primitives as jp

import math
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from sing.sde import SDE


class GmixQxAux(NamedTuple):
    """Per-outer-iter precomputed structures for live gmix moments."""
    rho_summed: Array   # (P,) real — sum over r of spectral autocorr, flat
    delta_flat: Array   # (P, d) real — corresponding freq-difference grid


def _autocorr_per_r(mu_r: Array, ws_real: Array,
                     mtot_per_dim: Tuple[int, ...]) -> Array:
    """Autocorr ρ_r(δ) = Σ_k tilde_μ_{r,k} tilde_μ_{r,k+δ} via padded FFT.

    Returns shape ``(D_out, *pad_shape)`` real, where
    ``pad_shape = tuple(2*n for n in mtot_per_dim)`` and lag 0 is at the
    centre after fftshift.
    """
    tilde = mu_r * ws_real[None, :]                 # (D_out, M)
    grid_shape = tuple(int(n) for n in mtot_per_dim)
    pad_shape  = tuple(2 * n for n in grid_shape)

    def one(u_flat):
        u_grid = u_flat.reshape(grid_shape)
        ft = jnp.fft.fftn(u_grid, s=pad_shape)
        ac = jnp.fft.ifftn(jnp.abs(ft) ** 2).real
        return jnp.fft.fftshift(ac)

    return jax.vmap(one)(tilde)


def _delta_grid_for_autocorr(grid: jp.JaxGridState) -> Array:
    """Difference-frequency lag grid matching fftshifted autocorr output.

    Per dim: lags = (arange(2N) − N) * h_per_dim[d]. Returns shape
    ``(prod(2*mtot_per_dim), d)``.
    """
    grids = []
    for d, n in enumerate(grid.mtot_per_dim):
        lags = (jnp.arange(2 * n, dtype=grid.h_per_dim.dtype) - n) \
                * grid.h_per_dim[d]
        grids.append(lags)
    delta = jnp.stack(jnp.meshgrid(*grids, indexing='ij'), axis=-1)
    return delta.reshape(-1, grid.d)


def precompute_aux(mu_r: Array, grid: jp.JaxGridState) -> GmixQxAux:
    """Precompute autocorr + difference grid for live gmix moments.

    Call once per outer iter (when ``mu_r`` updates); pass the result into
    every per-(m, S) moment evaluation.
    """
    ws_real = grid.ws.real
    rho_per_r  = _autocorr_per_r(mu_r, ws_real, grid.mtot_per_dim)
    rho_summed = rho_per_r.sum(0).reshape(-1)        # (P,)
    delta_flat = _delta_grid_for_autocorr(grid)      # (P, d)
    return GmixQxAux(rho_summed=rho_summed, delta_flat=delta_flat)


def gmix_E_full_Eff(m: Array, S: Array, mu_r: Array,
                     grid: jp.JaxGridState, aux: GmixQxAux) -> Array:
    """Full Gaussian-averaged second moment ``E_{q(x_i)}[bar_f^T bar_f]`` via
    spectral autocorrelation. For ELBO reporting only — \\emph{not} used
    in the gradient that drives the q(x) update; the gradient path uses
    the linearised quadratic ``||Ef||² + tr[J^T J S]`` via the legacy
    ``eff_with_grads`` custom VJP. See ``efgp_estep.tex`` §7."""
    cdtype = grid.ws.dtype
    rdtype = grid.xcen.dtype
    rho     = aux.rho_summed.astype(cdtype)
    delta   = aux.delta_flat.astype(rdtype)
    d_phase = jnp.exp(2j * jnp.pi * (delta @ m.astype(rdtype)).astype(cdtype))
    d_quad  = jnp.einsum('pd,de,pe->p', delta, S.astype(rdtype), delta)
    d_env   = jnp.exp(-2 * jnp.pi**2 * d_quad).astype(cdtype)
    return (rho * d_phase * d_env).real.sum()
