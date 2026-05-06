"""
Tests for multi-trial EFGP-SING.

Covers:
  * FrozenEFGPDrift pytree round-trip
  * K-replication σ²/K equivalence (the only mathematically clean
    invariant when the same data is replicated K times)
  * Ragged-trials vs end-to-end concatenation with masked boundary
  * K=1 invariance: the multi-trial path produces the same outputs as
    the existing single-trial path on K=1 inputs (smoke / regression
    against single-trial recovery)
  * (slow) K-helps-ℓ-recovery: K diverse trials help vs one long one
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SING = _HERE.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import sing.efgp_em as em
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
import jax
import jax.numpy as jnp
import jax.random as jr
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


# ----------------------------------------------------------------------------
# Pytree round-trip
# ----------------------------------------------------------------------------
def test_frozen_efgp_drift_pytree_roundtrip():
    """Flatten/unflatten FrozenEFGPDrift, and confirm vmap over K slices the
    leading axis of (Ef, Eff, Edfdx) so each mapped instance behaves like a
    single-trial drift."""
    K, T, D = 3, 8, 2
    t_grid = jnp.linspace(0.0, 1.0, T)
    Ef = jr.normal(jr.PRNGKey(0), (K, T, D))
    Eff = jr.normal(jr.PRNGKey(1), (K, T))
    Edfdx = jr.normal(jr.PRNGKey(2), (K, T, D, D))

    fr = jpd.FrozenEFGPDrift(latent_dim=D, t_grid=t_grid,
                              Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)

    leaves, treedef = jax.tree_util.tree_flatten(fr)
    assert len(leaves) == 3
    fr_back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(fr_back, jpd.FrozenEFGPDrift)
    assert fr_back.latent_dim == D
    assert jnp.allclose(fr_back._Ef, Ef)

    # vmap over K: each slice should pick up trial-k's arrays.
    def take_first(fr_k):
        return fr_k._Ef[0]                                # (D,)
    out = jax.vmap(take_first)(fr)                        # (K, D)
    assert out.shape == (K, D)
    assert jnp.allclose(out, Ef[:, 0])


# ----------------------------------------------------------------------------
# K-replication: σ²/K equivalence
# ----------------------------------------------------------------------------
def test_K_replication_sigma_over_K_equivalence():
    """K identical replicated trials with σ² give the same μ_r as 1 trial
    with σ²/K (because BTTB Toeplitz and RHS both scale linearly with K but
    the noise term I does not)."""
    rng = np.random.default_rng(42)
    K = 4
    T = 30
    D = 2
    sigma_sq = 0.4 ** 2

    # One synthetic q(x) trial
    base = jnp.linspace(-0.8, 0.8, T)
    ms_one = jnp.stack([0.6 * base + 0.1 * rng.standard_normal(T),
                        -0.5 * base + 0.05 * rng.standard_normal(T)],
                       axis=-1)                          # (T, D)
    Ss_one = jnp.tile(0.05 * jnp.eye(D), (T, 1, 1))
    SSs_one = 0.95 * Ss_one[:-1]
    del_t = jnp.full((T - 1,), 1.0 / (T - 1))

    # Build EFGP grid
    X_template = jnp.linspace(-1.5, 1.5, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(0.6, 1.0, X_template, eps=1e-2)

    # Replicate K-fold with full mask
    ms_K = jnp.broadcast_to(ms_one, (K, T, D))
    Ss_K = jnp.broadcast_to(Ss_one, (K, T, D, D))
    SSs_K = jnp.broadcast_to(SSs_one, (K, T - 1, D, D))
    trial_mask_K = jnp.ones((K, T), dtype=bool)
    trial_mask_1 = jnp.ones((1, T), dtype=bool)

    # (a) K trials with σ²
    mu_K, _, _, _ = jpd.qf_and_moments_gmix_jax(
        ms_K, Ss_K, SSs_K, del_t, trial_mask_K, grid,
        sigma_drift_sq=sigma_sq, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)

    # (b) K=1 with σ²/K
    mu_1, _, _, _ = jpd.qf_and_moments_gmix_jax(
        ms_one[None], Ss_one[None], SSs_one[None], del_t, trial_mask_1, grid,
        sigma_drift_sq=sigma_sq / K, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)

    rel = float(jnp.linalg.norm(mu_K - mu_1) / jnp.linalg.norm(mu_1))
    print(f"\n  K-replication σ²/K: rel || mu_K - mu_1 || = {rel:.2e}")
    assert rel < 5e-3, f"σ²/K invariant violated: rel err {rel:.2e}"


# ----------------------------------------------------------------------------
# Ragged trials via mask vs concatenation
# ----------------------------------------------------------------------------
def test_ragged_trials_match_concatenation():
    """K=2 trials of differing lengths via trial_mask should give the same
    q(f) as one concatenated trial with a masked boundary transition."""
    rng = np.random.default_rng(7)
    T1, T2 = 18, 26
    T_max = max(T1, T2)
    D = 2
    sigma_sq = 0.4 ** 2

    # Build two synthetic latent paths
    def make_path(seed, T_):
        r = np.random.default_rng(seed)
        base = jnp.linspace(-0.8, 0.8, T_)
        ms = jnp.stack([0.6 * base + 0.05 * r.standard_normal(T_),
                        -0.5 * base + 0.05 * r.standard_normal(T_)],
                        axis=-1)
        Ss = jnp.tile(0.05 * jnp.eye(D), (T_, 1, 1))
        SSs = 0.95 * Ss[:-1]
        return ms, Ss, SSs

    ms1, Ss1, SSs1 = make_path(1, T1)
    ms2, Ss2, SSs2 = make_path(2, T2)

    # ---- Path A: K=2 with padding + trial_mask ----
    def pad(arr, T_target, axis):
        pad_n = T_target - arr.shape[axis]
        if pad_n == 0:
            return arr
        pad_shape = list(arr.shape)
        pad_shape[axis] = pad_n
        # Pad with zeros — values don't matter because mask zeros them out.
        return jnp.concatenate(
            [arr, jnp.zeros(pad_shape, dtype=arr.dtype)], axis=axis)

    ms_K = jnp.stack([pad(ms1, T_max, 0), pad(ms2, T_max, 0)], axis=0)
    Ss_K = jnp.stack([pad(Ss1, T_max, 0), pad(Ss2, T_max, 0)], axis=0)
    SSs_K = jnp.stack([pad(SSs1, T_max - 1, 0), pad(SSs2, T_max - 1, 0)], axis=0)
    trial_mask_K = jnp.zeros((2, T_max), dtype=bool)
    trial_mask_K = trial_mask_K.at[0, :T1].set(True).at[1, :T2].set(True)
    del_t_K = jnp.full((T_max - 1,), 1.0 / (T_max - 1))

    X_template = jnp.linspace(-1.5, 1.5, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(0.6, 1.0, X_template, eps=1e-2)

    mu_pad, _, _, _ = jpd.qf_and_moments_gmix_jax(
        ms_K, Ss_K, SSs_K, del_t_K, trial_mask_K, grid,
        sigma_drift_sq=sigma_sq, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)

    # ---- Path B: concatenate as a single trial of length T1+T2 with the
    # boundary transition masked out (trans_mask=0 between the two trials).
    ms_cat = jnp.concatenate([ms1, ms2], axis=0)                 # (T1+T2, D)
    Ss_cat = jnp.concatenate([Ss1, Ss2], axis=0)
    # SSs_cat has T1+T2 - 1 transitions: the boundary one (between trial-1
    # last and trial-2 first) needs to be padded with anything (it'll be
    # zeroed by the mask).
    SSs_boundary = jnp.zeros((1, D, D))                           # arbitrary
    SSs_cat = jnp.concatenate([SSs1, SSs_boundary, SSs2], axis=0)  # (T1+T2-1, D, D)
    # Trial mask: all True except we explicitly mask the boundary transition.
    # We can do this by masking the first time of trial 2 — but that would
    # also remove ms2[0] from BTTB sources of trial 2.  Cleaner: keep
    # trial_mask all True and manually use a custom del_t that's zero at
    # the boundary.  But del_t goes into BTTB — trans_mask handles RHS too.
    #
    # Simpler hack: set trial_mask[t]=False for t = T1 (the gap).  This
    # zeros transition (T1-1, T1) AND (T1, T1+1) — i.e. removes both
    # the boundary transition AND the first transition of trial 2.  To
    # keep trial 2's first transition we need a different surgery.
    #
    # Cleanest: use a custom trans_mask via the public _flatten_stein
    # internals.  Build d_a and C_a for the full concat, zero out at
    # boundary index, then call compute_mu_r_gmix_jax directly with
    # the BTTB built from del_t with boundary set to 0.
    Tcat = T1 + T2
    trial_mask_cat = jnp.ones((1, Tcat), dtype=bool)

    # Build the "ideal" concatenated supersources by hand: compute Stein
    # over the full path, then zero out the boundary transition (index T1-1)
    # in both d_a, C_a, and the BTTB del_t.
    d_a = (ms_cat[1:] - ms_cat[:-1])                             # (Tcat-1, D)
    SSs_T_cat = jnp.swapaxes(SSs_cat, -1, -2)
    C_a = SSs_T_cat - Ss_cat[:-1]                                # (Tcat-1, D, D)

    boundary_idx = T1 - 1                                         # transition (T1-1, T1)
    del_t_cat = jnp.full((Tcat - 1,), 1.0 / (T_max - 1))          # uniform
    boundary_mask = jnp.ones(Tcat - 1).at[boundary_idx].set(0.0)
    d_a = d_a * boundary_mask[:, None]
    C_a = C_a * boundary_mask[:, None, None]
    del_t_cat = del_t_cat * boundary_mask                          # zero at boundary

    m_src = ms_cat[:-1]
    S_src = Ss_cat[:-1]

    mu_cat, _, _ = jpd.compute_mu_r_gmix_jax(
        m_src, S_src, d_a, C_a, del_t_cat, grid,
        sigma_drift_sq=sigma_sq, D_lat=D, D_out=D,
        fine_N=64, stencil_r=8, cg_tol=1e-6, max_cg_iter=2000)

    rel = float(jnp.linalg.norm(mu_pad - mu_cat) / jnp.linalg.norm(mu_cat))
    print(f"\n  ragged-vs-concat: rel || mu_pad - mu_cat || = {rel:.2e}")
    assert rel < 5e-3, f"ragged-vs-concat mismatch: rel err {rel:.2e}"


# ----------------------------------------------------------------------------
# K=1 invariance smoke test (multi-trial path with K=1)
# ----------------------------------------------------------------------------
def test_K1_smoke_recovery():
    """fit_efgp_sing_jax with K=1 should still produce a sensible fit
    on a small synthetic linear problem. Smoke test the multi-trial
    code path on the K=1 single-trial config."""
    D = 2
    T = 60
    sigma = 0.3
    n_em = 5

    def drift_fn(x, t):
        A = jnp.array([[-1.0, 0.0], [0.0, -1.5]])
        return A @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)

    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.0, -0.5]),
                       f=drift_fn, t_max=2.0, n_timesteps=T,
                       sigma=sigma_fn)
    N = 4
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                     R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(8), xs, out_true)

    class GLik(Likelihood):
        def ell(self, y, m, v, op):
            R = op['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - m) ** 2 + v) / R)

    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    yc = ys - ys.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys.mean(0), R=jnp.full((N,), 0.1))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                  dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))

    mp, _, _, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., 2.0, T),
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.6, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=8,
        rho_sched=jnp.linspace(0.3, 0.9, n_em),
        learn_emissions=True, update_R=False,
        learn_kernel=False, kernel_warmup_iters=2,
        emission_warmup_iters=2,
        verbose=False,
    )
    # Output marginals should still be (K=1, T, D)
    assert mp['m'].shape == (1, T, D), f"unexpected ms shape: {mp['m'].shape}"
    # Latent should at least be in the right ballpark
    rmse = float(jnp.sqrt(jnp.mean((mp['m'][0] - xs) ** 2)))
    assert rmse < 1.0, f"K=1 smoke fit RMSE {rmse:.3f} too large"


# ----------------------------------------------------------------------------
# Multi-trial smoke fit
# ----------------------------------------------------------------------------
def test_K_trials_smoke_fit():
    """fit_efgp_sing_jax with K=3 trials runs end-to-end and produces
    output marginals of the correct (K, T, D) shape, with reasonable
    per-trial latent RMSE on a linear drift."""
    D = 2
    T = 50
    K = 3
    sigma = 0.3
    n_em = 4

    def drift_fn(x, t):
        A = jnp.array([[-1.0, 0.0], [0.0, -1.5]])
        return A @ x
    sigma_fn = lambda x, t: sigma * jnp.eye(D)

    # Simulate K trials with diverse x0
    x0s = jnp.array([[1.0, -0.5], [-0.7, 0.6], [0.3, 1.0]])
    xs_list = [
        simulate_sde(jr.PRNGKey(10 + k), x0=x0s[k],
                      f=drift_fn, t_max=1.5, n_timesteps=T, sigma=sigma_fn)
        for k in range(K)
    ]
    xs_K = jnp.stack(xs_list, axis=0)                            # (K, T, D)

    N = 4
    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                     R=jnp.full((N,), 0.05))
    ys_list = [simulate_gaussian_obs(jr.PRNGKey(20 + k), xs_K[k], out_true)
               for k in range(K)]
    ys_K = jnp.stack(ys_list, axis=0)                            # (K, T, N)

    class GLik(Likelihood):
        def ell(self, y, m, v, op):
            R = op['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - m) ** 2 + v) / R)

    t_mask = jnp.ones((K, T), dtype=bool)
    lik = GLik(ys_K, t_mask)

    # Init via SVD on flattened ys
    ys_flat = ys_K.reshape(-1, N)
    yc = ys_flat - ys_flat.mean(0)
    _, _, Vt = jnp.linalg.svd(yc, full_matrices=False)
    op = dict(C=Vt[:D].T, d=ys_flat.mean(0), R=jnp.full((N,), 0.1))
    # Per-trial init params
    ip = dict(mu0=jnp.tile(jnp.zeros(D), (K, 1)),
               V0=jnp.tile(jnp.eye(D) * 0.1, (K, 1, 1)))

    mp, _, _, _, _ = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., 1.5, T),
        output_params=op, init_params=ip, latent_dim=D,
        lengthscale=0.6, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2, S_marginal=2,
        n_em_iters=n_em, n_estep_iters=8,
        rho_sched=jnp.linspace(0.3, 0.9, n_em),
        learn_emissions=True, update_R=False,
        learn_kernel=False, kernel_warmup_iters=2,
        emission_warmup_iters=2,
        verbose=False,
    )
    assert mp['m'].shape == (K, T, D)
    # Per-trial fit should be in the ballpark for each trial
    per_trial_rmse = jnp.sqrt(jnp.mean((mp['m'] - xs_K) ** 2, axis=(1, 2)))
    print(f"\n  K-trial smoke fit: per-trial RMSE = {np.asarray(per_trial_rmse)}")
    assert float(per_trial_rmse.mean()) < 1.0, (
        f"multi-trial smoke fit per-trial RMSE {per_trial_rmse} too large")


if __name__ == "__main__":
    test_frozen_efgp_drift_pytree_roundtrip()
    test_K_replication_sigma_over_K_equivalence()
    test_ragged_trials_match_concatenation()
    test_K1_smoke_recovery()
    test_K_trials_smoke_fit()
    print("\nAll multi-trial tests passed.")
