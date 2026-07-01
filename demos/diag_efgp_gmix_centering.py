"""Localize the EFGP centering bug: gmix q(f) build vs MC reference, off-origin.

Synthetic Stein sources centered at xcen ~ [5,5] (off origin).  Compares the
posterior drift field from:
  - compute_mu_r_jax      (MONTE CARLO; standard nufft1/nufft2, the reference)
  - compute_mu_r_gmix_jax (gmix spreader; the suspect)

and checks each path's TRANSLATION-EQUIVARIANCE (evaluate at center [5,5] vs
the same problem shifted to [0,0]).

Expectation if the bug is the extra exp(2pi i xi.xcen) phase in _gmix_fft:
  - MC: equivariant, and matches itself at [0,0] and [5,5].
  - gmix: MATCHES MC at [0,0] but DIVERGES at [5,5] and is NOT equivariant.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_gmix_centering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd

D = 2
SIG2 = 1.0
LS = 0.8
VAR = 1.0


def make_sources(center, seed=0, n=400):
    rng = np.random.default_rng(seed)
    m = rng.normal(center, 0.6, size=(n, D))
    S = np.tile(0.02 * np.eye(D), (n, 1, 1))
    d = rng.normal(0, 0.1, size=(n, D))                 # displacement targets
    C = np.tile(0.0 * np.eye(D), (n, 1, 1))             # no Stein for clarity
    w = np.full(n, 0.05)
    return (jnp.asarray(m), jnp.asarray(S), jnp.asarray(d),
            jnp.asarray(C), jnp.asarray(w))


def field(compute, m, S, d, C, w, eval_pts, gmix):
    X_tmpl = jnp.asarray(np.asarray(m))
    grid = jp.spectral_grid_se(LS, VAR, X_tmpl, eps=1e-2)
    if gmix:
        mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
            m, S, d, C, w, grid, sigma_drift_sq=SIG2, D_lat=D, D_out=D,
            fine_N=256, stencil_r=16, cg_tol=1e-8, max_cg_iter=4000)
    else:
        mu_r, _, _ = jpd.compute_mu_r_jax(
            m, S, d, C, w, grid, jr.PRNGKey(7), sigma_drift_sq=SIG2,
            S_marginal=20, D_lat=D, D_out=D, cg_tol=1e-8, max_cg_iter=4000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(eval_pts)[None], D_lat=D, D_out=D)
    return np.asarray(Ef[0]), float(grid.xcen[0])


def field_gather(m, S, d, C, w, eval_pts):
    """gmix build + gmix GATHER eval (drift_moments_gmix_jax) -- what the EM
    actually uses.  Tiny Ss at eval points => point evaluation."""
    X_tmpl = jnp.asarray(np.asarray(m))
    grid = jp.spectral_grid_se(LS, VAR, X_tmpl, eps=1e-2)
    mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
        m, S, d, C, w, grid, sigma_drift_sq=SIG2, D_lat=D, D_out=D,
        fine_N=256, stencil_r=16, cg_tol=1e-8, max_cg_iter=4000)
    P = jnp.asarray(eval_pts)[None]                       # (1, Mpts, D)
    Ss = jnp.broadcast_to(1e-6 * jnp.eye(D), P.shape[:2] + (D, D))
    Ef, _, _ = jpd.drift_moments_gmix_jax(
        mu_r, grid, P, Ss, D_lat=D, D_out=D, gather_N=256, stencil_r=16)
    return np.asarray(Ef[0])


def run_at(center, tag):
    m, S, d, C, w = make_sources(np.array(center))
    # eval on a grid around the sources
    g = np.linspace(-1.5, 1.5, 25)
    GX, GY = np.meshgrid(g + center[0], g + center[1], indexing='ij')
    P = np.stack([GX.ravel(), GY.ravel()], -1)
    f_mc, xc = field(jpd.compute_mu_r_jax, m, S, d, C, w, P, gmix=False)
    f_gx, _ = field(jpd.compute_mu_r_gmix_jax, m, S, d, C, w, P, gmix=True)
    f_ga = field_gather(m, S, d, C, w, P)
    rel_nufft = np.linalg.norm(f_gx - f_mc) / (np.linalg.norm(f_mc) + 1e-12)
    rel_gather = np.linalg.norm(f_ga - f_mc) / (np.linalg.norm(f_mc) + 1e-12)
    print(f"  [{tag}] xcen~{xc:.2f}  median||f_mc||="
          f"{np.median(np.linalg.norm(f_mc,axis=-1)):.3f}")
    print(f"           gmix(build)+nufft2  vs MC : rel = {rel_nufft:.3f}")
    print(f"           gmix(build)+gather  vs MC : rel = {rel_gather:.3f}")
    return f_mc, f_gx


def main():
    print("  gmix-vs-MC drift agreement at the ORIGIN vs OFF-ORIGIN:\n")
    f_mc0, f_gx0 = run_at([0.0, 0.0], "origin  ")
    f_mc5, f_gx5 = run_at([5.0, 5.0], "off [5,5]")

    # Self-equivariance: same problem at 0 and at 5 -> identical field values.
    eq_mc = np.linalg.norm(f_mc5 - f_mc0) / (np.linalg.norm(f_mc0) + 1e-12)
    eq_gx = np.linalg.norm(f_gx5 - f_gx0) / (np.linalg.norm(f_gx0) + 1e-12)
    print(f"\n  self-equivariance (field at [5,5] vs [0,0], same problem):")
    print(f"    MC   rel change = {eq_mc:.3f}   (should be ~0)")
    print(f"    gmix rel change = {eq_gx:.3f}   (large => gmix centering bug)")


if __name__ == "__main__":
    main()
