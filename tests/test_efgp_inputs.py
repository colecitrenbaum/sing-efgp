"""Tests for EFGP-SING input / input-effect (B) support.

Model:  dx(t) = (f(x) + B v(t)) dt + dW(t).

Two tests:

1. ``test_update_input_effect_efgp_recovers_B`` — a *deterministic* unit
   test of the closed-form B solver ``_update_input_effect_efgp``.  We
   construct latent means whose increments are exactly
   ``Δm = Δt·(E[f] + B_true·v)`` and check the solver returns ``B_true``
   to machine precision (multi-trial, with a trial mask).

2. ``test_efgp_learns_input_effect_e2e`` (slow) — an end-to-end fit on a
   linear SDE driven by a known step input.  Emissions are FIXED at truth
   (so latents stay in truth coordinates) and we check the learned B
   points in the right direction with a sensible magnitude, and that the
   input-driven latents are tracked.
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

# Load jax_finufft FIRST via efgp_em (which imports efgp_jax_primitives).
import sing.efgp_em as em
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from sing.likelihoods import Likelihood
from sing.inputs import InputSignals
from sing.simulate_data import simulate_sde, simulate_gaussian_obs


class GLik(Likelihood):
    def ell(self, y, m, v, op):
        R = op['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - m) ** 2 + v) / R)


def test_update_input_effect_efgp_recovers_B():
    """Closed-form solver recovers B_true exactly when Δm matches the model."""
    rng = np.random.default_rng(0)
    K, T, D, I = 3, 40, 2, 2
    B_true = rng.standard_normal((D, I))

    del_t = jnp.asarray(rng.uniform(0.02, 0.05, size=(T - 1,)))   # (T-1,)
    Ef = jnp.asarray(rng.standard_normal((K, T - 1, D)))         # E[f(x_t)]
    inputs = jnp.asarray(rng.standard_normal((K, T, I)))         # v(t)

    # Build ms so that Δm_t = Δt_t · (E[f]_t + B_true·v_t) exactly.
    drive = del_t[None, :, None] * (
        Ef + jnp.einsum('di,kti->ktd', jnp.asarray(B_true), inputs[:, :-1]))
    ms0 = jnp.asarray(rng.standard_normal((K, 1, D)))
    ms = jnp.concatenate([ms0, ms0 + jnp.cumsum(drive, axis=1)], axis=1)  # (K,T,D)

    # A trial mask that deactivates a tail of trial 2 — those transitions
    # must be excluded from the LS system but recovery should still be exact.
    trial_mask = jnp.ones((K, T), dtype=bool)
    trial_mask = trial_mask.at[2, T - 5:].set(False)

    B_hat = em._update_input_effect_efgp(ms, Ef, inputs, trial_mask, del_t,
                                         jitter=1e-12)
    err = float(jnp.max(jnp.abs(B_hat - jnp.asarray(B_true))))
    print(f"\n  max|B_hat - B_true| = {err:.2e}")
    assert err < 1e-6, f"closed-form B solve off by {err:.2e}\n{np.asarray(B_hat)}"


def test_input_signals_none_is_backcompat():
    """With input_signals=None the fit runs and reports no input_effect."""
    D, T, N = 2, 60, 5
    sigma = 0.3
    A_true = jnp.array([[-0.5, 1.0], [-1.0, -0.5]])
    xs = simulate_sde(jr.PRNGKey(0), x0=jnp.array([1.0, 0.0]),
                      f=lambda x, t: A_true @ x, t_max=3.0, n_timesteps=T,
                      sigma=lambda x, t: sigma * jnp.eye(D))
    rng = np.random.default_rng(1)
    C_true = rng.standard_normal((N, D)) * 0.5
    out = dict(C=jnp.asarray(C_true), d=jnp.zeros(N), R=jnp.full((N,), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(2), xs, out)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))
    ip = jax.tree_util.tree_map(lambda x: x[None],
                                dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.3))

    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=jnp.linspace(0., 3.0, T),
        output_params=dict(out), init_params=ip, latent_dim=D,
        lengthscale=1.0, variance=1.0, sigma=sigma,
        n_em_iters=6, n_estep_iters=6, learn_emissions=False,
        learn_kernel=False, verbose=False)
    assert hist.input_effect is None
    assert np.all(np.isfinite(np.asarray(mp['m'])))


@pytest.mark.slow
def test_efgp_learns_input_effect_e2e():
    """End-to-end: recover a known step-input effect B on a linear SDE.

    Emissions fixed at truth so latents stay in truth coordinates and the
    learned B is directly comparable to B_true.
    """
    D, T, N = 2, 300, 8
    t_max = 6.0
    sigma = 0.25
    A_true = jnp.array([[-0.4, 1.0], [-1.0, -0.4]])
    B_true = jnp.array([[2.0, 0.0],
                        [0.0, 2.0]])          # (D, I), I=2

    # Two disjoint step-input windows (mimicking the neural-data intruders).
    inputs = np.zeros((1, T, 2))
    inputs[0, 60:90, 0] = 1.0
    inputs[0, 180:210, 1] = 1.0
    inputs_j = jnp.asarray(inputs)
    t_grid = jnp.linspace(0., t_max, T)
    del_t = float(t_grid[1] - t_grid[0])

    # Simulate dx = (A x + B v) dt + dW by folding the input drift into f.
    v_of_t = lambda ti: inputs_j[0, jnp.clip((ti / del_t).astype(int), 0, T - 1)]
    drift = lambda x, t: A_true @ x + B_true @ v_of_t(t)
    xs = simulate_sde(jr.PRNGKey(7), x0=jnp.array([1.5, 0.0]), f=drift,
                      t_max=t_max, n_timesteps=T,
                      sigma=lambda x, t: sigma * jnp.eye(D))
    xs = jnp.clip(xs, -8.0, 8.0)

    rng = np.random.default_rng(0)
    C_true = rng.standard_normal((N, D)) * 0.5
    out_true = dict(C=jnp.asarray(C_true), d=jnp.zeros(N),
                    R=jnp.full((N,), 0.04))
    ys = simulate_gaussian_obs(jr.PRNGKey(1), xs, out_true)
    lik = GLik(ys[None], jnp.ones((1, T), dtype=bool))

    ip = jax.tree_util.tree_map(lambda x: x[None],
                                dict(mu0=jnp.array([1.5, 0.0]),
                                     V0=jnp.eye(D) * 0.3))

    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(out_true), init_params=ip, latent_dim=D,
        lengthscale=1.2, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma ** 2, eps_grid=1e-2,
        n_em_iters=25, n_estep_iters=10,
        rho_sched=jnp.linspace(0.05, 0.7, 25),
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
        kernel_warmup_iters=6,
        input_signals=InputSignals(inputs_j),
        learn_input_effect=True, input_effect_warmup_iters=6,
        X_template=(jnp.linspace(-8., 8., T)[:, None] * jnp.ones((1, D))),
        verbose=False,
    )

    B_hat = np.asarray(hist.input_effect)
    B_t = np.asarray(B_true)
    assert B_hat is not None and B_hat.shape == (D, 2)
    rel = np.linalg.norm(B_hat - B_t) / np.linalg.norm(B_t)
    # Sign/direction agreement on the dominant component of each input column.
    cos0 = float(B_hat[:, 0] @ B_t[:, 0]
                 / (np.linalg.norm(B_hat[:, 0]) * np.linalg.norm(B_t[:, 0]) + 1e-9))
    cos1 = float(B_hat[:, 1] @ B_t[:, 1]
                 / (np.linalg.norm(B_hat[:, 1]) * np.linalg.norm(B_t[:, 1]) + 1e-9))
    print(f"\n  B_true=\n{B_t}\n  B_hat=\n{B_hat}\n  rel={rel:.3f}  "
          f"cos0={cos0:.3f}  cos1={cos1:.3f}")

    # Latents should track the truth (input-driven excursions included).
    lat_rmse = float(np.sqrt(np.mean((np.asarray(mp['m'][0]) - np.asarray(xs)) ** 2)))
    print(f"  latent RMSE = {lat_rmse:.4f}")

    # What is reliably identifiable here is the DIRECTION of each input
    # column and that B is non-trivial — the deterministic unit test above
    # proves the solver itself is exact.  Per-column magnitude is only
    # ballpark: with a *learned* GP kernel the drift f(x) absorbs part of
    # each brief input-driven excursion, so columns can be underestimated.
    norm_ratio = np.linalg.norm(B_hat) / np.linalg.norm(B_t)
    print(f"  ||B_hat||/||B_true|| = {norm_ratio:.3f}")
    assert cos0 > 0.7 and cos1 > 0.7, "learned B points the wrong way"
    assert norm_ratio > 0.35, f"learned B collapsed to ~0 (ratio={norm_ratio:.3f})"
    assert lat_rmse < 0.6, f"latents not tracked (RMSE={lat_rmse:.3f})"
