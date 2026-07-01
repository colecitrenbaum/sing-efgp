"""Regression test for the gmix q(f) centering/phase bug.

The EFGP posterior drift must be TRANSLATION-EQUIVARIANT: with the same
(shifted) sources, the field shifts by the same constant and its values are
unchanged.  The gmix q(f) build (``compute_mu_r_gmix_jax``) used to return
``mu_r`` in the spreader's ABSOLUTE frame while the evaluator
``drift_moments_jax`` (and the MC build ``compute_mu_r_jax``) work in the
RELATIVE frame, so the evaluated gmix field was shifted by ``grid.xcen`` —
invisible at xcen≈0 (every centered synthetic bench) but a real spatial shift
on off-origin data.

These tests pin the fix:
  1. gmix and MC drift fields agree to within gmix-spreader tolerance at an
     OFF-ORIGIN center (they always agreed at the origin).
  2. The gmix field is translation-equivariant: building+evaluating the same
     problem at center [0,0] vs [6,6] gives identical field values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import jax
import jax.numpy as jnp
import jax.random as jr

D = 2
SIG2 = 1.0
LS = 0.8
VAR = 1.0


def _sources(center, seed=0, n=400):
    rng = np.random.default_rng(seed)
    m = rng.normal(center, 0.6, size=(n, D))
    S = np.tile(0.02 * np.eye(D), (n, 1, 1))
    d = rng.normal(0, 0.1, size=(n, D))
    C = np.zeros((n, D, D))
    w = np.full(n, 0.05)
    return (jnp.asarray(m), jnp.asarray(S), jnp.asarray(d),
            jnp.asarray(C), jnp.asarray(w))


def _field(center, gmix, eval_pts):
    m, S, d, C, w = _sources(np.asarray(center))
    grid = jp.spectral_grid_se(LS, VAR, m, eps=1e-2)
    if gmix:
        mu_r, _, _ = jpd.compute_mu_r_gmix_jax(
            m, S, d, C, w, grid, sigma_drift_sq=SIG2, D_lat=D, D_out=D,
            fine_N=256, stencil_r=16, cg_tol=1e-8, max_cg_iter=4000)
    else:
        mu_r, _, _ = jpd.compute_mu_r_jax(
            m, S, d, C, w, grid, jr.PRNGKey(0), sigma_drift_sq=SIG2,
            S_marginal=20, D_lat=D, D_out=D, cg_tol=1e-8, max_cg_iter=4000)
    Ef, _, _ = jpd.drift_moments_jax(
        mu_r, grid, jnp.asarray(eval_pts)[None], D_lat=D, D_out=D)
    return np.asarray(Ef[0])


def test_gmix_matches_mc_off_origin():
    """gmix drift field ~ MC drift field at an off-origin center."""
    center = [5.0, 5.0]
    g = np.linspace(-1.5, 1.5, 20)
    GX, GY = np.meshgrid(g + center[0], g + center[1], indexing='ij')
    P = np.stack([GX.ravel(), GY.ravel()], -1)
    f_gmix = _field(center, True, P)
    f_mc = _field(center, False, P)
    rel = np.linalg.norm(f_gmix - f_mc) / (np.linalg.norm(f_mc) + 1e-12)
    print(f"\n  off-origin gmix-vs-MC rel = {rel:.3f}")
    # gmix-spreader truncation vs MC sampling: a few percent.  Pre-fix this
    # was ~1.1 (field shifted by xcen).
    assert rel < 0.20, f"gmix drift disagrees with MC off-origin (rel={rel:.3f})"


def test_gmix_translation_equivariant():
    """Same problem at [0,0] vs [6,6] -> identical gmix field values."""
    g = np.linspace(-1.5, 1.5, 20)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    base = np.stack([GX.ravel(), GY.ravel()], -1)
    c = np.array([6.0, 6.0])
    f0 = _field([0.0, 0.0], True, base)
    f6 = _field(list(c), True, base + c)            # same relative geometry
    rel = np.linalg.norm(f6 - f0) / (np.linalg.norm(f0) + 1e-12)
    print(f"\n  gmix self-equivariance rel change = {rel:.4f}")
    assert rel < 1e-2, (
        f"gmix drift not translation-equivariant (rel={rel:.3f}); "
        "centering/phase bug in compute_mu_r_gmix_jax has regressed.")
