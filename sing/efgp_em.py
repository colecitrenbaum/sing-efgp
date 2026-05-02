"""
Plain-Python variational-EM loop for the EFGP-SING drift block.

Why plain Python?  See ``sing.efgp_drift``: the EFGP block runs in PyTorch
and bridges to JAX via numpy round-trips, so we can't keep SING's original
``@jit``-then-``lax.scan`` E-step.  Instead we drive the same SING natural-
gradient updates from a vanilla Python loop.

# TODO(efgp-jax): once an EFGPBackend with native JAX exists, the EM loop
# here can be re-jitted (replace this whole module by a thin wrapper around
# the existing ``sing.sing.fit_variational_em``).

Scope (v0)
----------
Single trial.  Gaussian likelihood with closed-form ``(C, d, R)`` update.
Kernel hyperparameters via the MAP-like collapsed M-step (see
``EFGPDrift.m_step_kernel``).  Diffusion ``Σ`` and input matrix ``B`` held
fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from sing.sde import SDE
from sing.likelihoods import Likelihood
from sing.utils.params import MarginalParams, NaturalParams
from sing.utils.sing_helpers import (
    natural_to_marginal_params, natural_to_mean_params,
    dynamics_to_natural_params, mean_to_marginal_params,
    InitParams, update_init_params,
)
from sing.sing import nat_grad_likelihood, nat_grad_transition
from sing.initialization import initialize_params

from sing.efgp_drift import EFGPDrift, _FrozenEFGPDrift
from sing.efgp_emissions import update_emissions_gaussian

from functools import partial


def _build_jit_estep(D, T, t_grid, lik, output_params_static_keys, sigma):
    """Compile a single inner-E-step iteration.

    The compiled function takes natural_params, init_params, frozen-drift
    arrays (Ef, Eff, Edfdx) and observation tensors as JAX inputs; building a
    fresh _FrozenEFGPDrift inside the jit body is cheap because it only
    closes over the input arrays, not over any python state that changes
    between iters.

    Caching the compiled artifact across iters is what makes the EM loop
    competitive with SING's native ``fit_variational_em`` after a single
    warm-up iter.

    # TODO(efgp-jax): when EFGPBackend is JAX-native, the whole
    # outer EM loop can be jit'd and lax.scan'd.
    """
    inputs = jnp.zeros((T, 1))
    input_effect = jnp.zeros((D, 1))

    @jax.jit
    def _step(natural_params, init_params, Ef, Eff, Edfdx,
              ys_obs, t_mask, output_params, key, rho):
        trial_mask = jnp.ones((1, T), dtype=bool)
        # mean params from natural
        mean_params_b, _ = vmap(natural_to_mean_params)(natural_params,
                                                        trial_mask)
        frozen = _FrozenEFGPDrift(latent_dim=D, t_grid=t_grid,
                                  Ef_per_t=Ef, Eff_per_t=Eff,
                                  Edfdx_per_t=Edfdx)
        # likelihood nat-grads
        lik_g = vmap(lambda mp_, tm_, ys_: nat_grad_likelihood(
            mp_, tm_, ys_, lik, output_params))(mean_params_b, t_mask, ys_obs)
        # transition nat-grads
        tr_g = vmap(lambda k, mp_, tm_, ip_: nat_grad_transition(
            k, frozen, None, {}, tm_, ip_, t_grid, mp_,
            inputs, input_effect, sigma))(jr.split(key, 1),
                                          mean_params_b, trial_mask, init_params)
        return {
            'J': (1 - rho) * natural_params['J']
                 + rho * (lik_g['J'] + tr_g['J']),
            'L': (1 - rho) * natural_params['L'] + rho * tr_g['L'],
            'h': (1 - rho) * natural_params['h']
                 + rho * (lik_g['h'] + tr_g['h']),
        }

    return _step


@dataclass
class EFGPEMHistory:
    """Per-iteration diagnostics from the EM loop."""
    latent_rmse: List[float] = field(default_factory=list)
    drift_rmse: List[float] = field(default_factory=list)
    lengthscale: List[float] = field(default_factory=list)
    variance: List[float] = field(default_factory=list)
    mstep_loss: List[float] = field(default_factory=list)


def fit_efgp_sing(
    *,
    drift: EFGPDrift,
    likelihood: Likelihood,
    t_grid: jnp.ndarray,
    output_params: Dict[str, jnp.ndarray],
    init_params: InitParams,
    sigma: float = 1.0,
    n_em_iters: int = 20,
    n_estep_iters: int = 15,
    n_mstep_iters: int = 10,
    rho_sched: Optional[jnp.ndarray] = None,
    mstep_lr: float = 0.05,
    learn_emissions: bool = True,
    learn_kernel: bool = True,
    update_R: bool = True,
    refresh_qf_every: int = 1,
    seed: int = 0,
    true_xs: Optional[np.ndarray] = None,
    true_drift_fn=None,
    grid_eval_pts: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[MarginalParams, NaturalParams, Dict[str, jnp.ndarray],
           InitParams, EFGPEMHistory]:
    """Variational-EM with the EFGP drift block.

    Single-trial, Gaussian-likelihood, no-inputs version.

    Parameters
    ----------
    drift : EFGPDrift
        Pre-constructed EFGP drift block with kernel/hypers initialised.
    likelihood : Likelihood
        SING ``GaussianLikelihood`` instance (provides ``ys_obs`` and ``t_mask``).
    t_grid : (T,) jax array
    output_params : dict with keys ``'C'`` (N×D), ``'d'`` (N,), ``'R'`` (N,).
    init_params : ``{'mu0': (D,), 'V0': (D, D)}`` initial-state prior.
    sigma : scalar diffusion (held fixed).
    n_em_iters : outer EM iterations.
    n_estep_iters : SING natural-grad updates per E-step.
    n_mstep_iters : Adam steps for the kernel M-step per outer iter.
    rho_sched : optional length-``n_em_iters`` SING step-size schedule.
    mstep_lr : Adam lr for kernel M-step.
    learn_emissions : update ``C, d, R`` in the M-step.
    learn_kernel : update kernel hypers in the M-step.
    update_R : whether to update observation noise (only if ``learn_emissions``).
    refresh_qf_every : refresh ``q(f)`` every K E-step inner iterations.
    seed : RNG seed for SING natural-grad keys.
    true_xs, true_drift_fn, grid_eval_pts : if provided, log RMSEs.
    verbose : print per-iter diagnostics.

    Returns
    -------
    marginal_params, natural_params, output_params, init_params, history
    """
    if rho_sched is None:
        rho_sched = jnp.logspace(-2, -1, n_em_iters)
    rho_sched = jnp.asarray(rho_sched)
    assert rho_sched.shape == (n_em_iters,)

    ys_obs = likelihood.ys_obs                                  # (n_trials, T, N)
    t_mask = likelihood.t_mask                                  # (n_trials, T)
    assert ys_obs.shape[0] == 1, "v0 supports a single trial."
    n_trials, T, N = ys_obs.shape
    D = drift.latent_dim
    trial_mask = jnp.ones((n_trials, T), dtype=bool)

    # SING expects per-time inputs even when none exist
    inputs = jnp.zeros((T, 1))
    input_effect = jnp.zeros((D, 1))

    # ---------------- initialise natural params via SING ----------------
    key = jr.PRNGKey(seed)
    init_key, key = jr.split(key, 2)
    natural_params, marginal_params, _ = vmap(
        lambda k, ip, tm: initialize_params(drift, drift_params={}, key=k,
                                            init_params=ip, trial_mask=tm,
                                            t_grid=t_grid, sigma=sigma),
        out_axes=(0, 0, None),
    )(jr.split(init_key, n_trials), init_params, trial_mask)

    # First q(f) build
    qf_key, key = jr.split(key, 2)
    drift.update_dynamics_params(qf_key, t_grid, marginal_params, trial_mask,
                                 drift_params={}, inputs=inputs,
                                 input_effect=input_effect, sigma=sigma)

    history = EFGPEMHistory()

    # Pre-build jit'd E-step iteration; caches the compiled artifact across
    # iterations so per-iter dispatch matches SING's native fit_variational_em.
    # Sanity: SING's outer sigma must match the drift's σ_r (tied case).
    sigma_d_sq = float(drift.sigma_drift_sq[0].cpu())
    if not np.isclose(sigma_d_sq, sigma ** 2, rtol=1e-6):
        raise ValueError(
            f"EFGPDrift's sigma_drift_sq={sigma_d_sq} disagrees with "
            f"EM-loop sigma²={sigma ** 2}. Pass them consistently.")
    estep_step_fn = _build_jit_estep(D=D, T=T, t_grid=t_grid,
                                     lik=likelihood,
                                     output_params_static_keys=tuple(output_params),
                                     sigma=sigma)
    # Profiling showed that the per-iter Bayesian-smoother conversion
    # (vmap(natural_to_marginal_params)) was ~95% of the wall time when
    # called bare from Python — it gets retraced every call.  Jitting it
    # collapses to ~5 ms / iter.
    nat_to_marg_jit = jax.jit(vmap(natural_to_marginal_params))

    # ---------------- main EM loop --------------------------------------
    for it in range(n_em_iters):
        rho = float(rho_sched[it])

        # ----- E-step ------------------------------------------------------
        for e in range(n_estep_iters):
            # Get current marginals (jit-cached after first call)
            mp_b, _ = nat_to_marg_jit(natural_params, trial_mask)

            # Refresh q(f) on schedule (Torch side; not JAX-traced)
            if e % refresh_qf_every == 0:
                qf_key, key = jr.split(key, 2)
                drift.update_dynamics_params(qf_key, t_grid, mp_b, trial_mask,
                                             drift_params={},
                                             inputs=inputs, input_effect=input_effect,
                                             sigma=sigma)

            # Batched drift moments at each (m_t, S_t) — single torch round-trip
            Ef, Eff, Edfdx = drift.drift_moments_at_marginals(
                mp_b['m'][0], mp_b['S'][0]
            )

            tr_key, key = jr.split(key, 2)
            # Pass rho as a JAX scalar so changing rho doesn't retrigger jit
            # tracing each iter.
            natural_params = estep_step_fn(
                natural_params, init_params,
                Ef, Eff, Edfdx,
                ys_obs, t_mask, output_params,
                tr_key, jnp.asarray(rho, dtype=jnp.float32),
            )

        # ----- M-step ------------------------------------------------------
        # Reuse the marginals computed at the end of the E-step (jit cached).
        # Avoids a second un-jit'd Bayesian-smoother call per outer EM iter
        # (~1 s overhead each).
        mp_b, _ = nat_to_marg_jit(natural_params, trial_mask)
        ms_t = mp_b['m'][0]                                       # (T, D)
        Ss_t = mp_b['S'][0]                                       # (T, D, D)

        # Closed-form (C, d, R) for Gaussian emissions
        if learn_emissions:
            C_new, d_new, R_new = update_emissions_gaussian(
                ms_t, Ss_t, ys_obs[0], t_mask[0], update_R=update_R,
            )
            output_params = {**output_params, 'C': C_new, 'd': d_new}
            if R_new is not None:
                output_params['R'] = R_new

        # Update init prior
        init_params = vmap(update_init_params)(ms_t[None, 0], Ss_t[None, 0])

        # Kernel M-step (drops trace term — see EFGPDrift.m_step_kernel docstring)
        ms_loss = []
        if learn_kernel:
            qf_key, key = jr.split(key, 2)
            drift.update_dynamics_params(qf_key, t_grid, mp_b, trial_mask,
                                         drift_params={}, inputs=inputs,
                                         input_effect=input_effect, sigma=sigma)
            info = drift.m_step_kernel(n_inner=n_mstep_iters, lr=mstep_lr,
                                       verbose=False)
            ms_loss = info['loss_history']

        # ----- Diagnostics -------------------------------------------------
        if true_xs is not None:
            x_inferred = np.asarray(ms_t)
            rmse_lat = float(np.sqrt(np.mean((x_inferred - true_xs) ** 2)))
            history.latent_rmse.append(rmse_lat)
        if true_drift_fn is not None and grid_eval_pts is not None:
            import torch
            X_e = torch.tensor(grid_eval_pts, dtype=drift.backend.rdtype)
            f_pred = drift._eval_mean_torch(X_e).cpu().numpy()
            f_true = true_drift_fn(grid_eval_pts)
            history.drift_rmse.append(float(np.sqrt(np.mean((f_pred - f_true) ** 2))))
        history.lengthscale.append(float(drift.kernel.lengthscale))
        history.variance.append(float(drift.kernel.variance))
        if ms_loss:
            history.mstep_loss.append(ms_loss[-1])

        if verbose:
            extras = []
            if history.latent_rmse: extras.append(f"latent_rmse={history.latent_rmse[-1]:.3f}")
            if history.drift_rmse:  extras.append(f"drift_rmse={history.drift_rmse[-1]:.3f}")
            extras.append(f"ℓ={history.lengthscale[-1]:.3f}")
            extras.append(f"σ²={history.variance[-1]:.3f}")
            print(f"EM {it + 1:2d}/{n_em_iters}  rho={rho:.3f}  " + "  ".join(extras))

    marginal_params, _ = vmap(natural_to_marginal_params)(natural_params, trial_mask)
    return marginal_params, natural_params, output_params, init_params, history
