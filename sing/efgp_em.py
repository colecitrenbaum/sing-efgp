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

from sing.efgp_emissions import update_emissions_gaussian

from functools import partial

# JAX-native path (must be importable WITHOUT pulling in torch).
# Import jax_finufft FIRST via efgp_jax_primitives so it loads before any
# pytorch_finufft (which efgpnd uses) — they share libfinufft.
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import FrozenEFGPDrift as _FrozenEFGPDrift

# Torch path is optional: only import when fit_efgp_sing (not _jax) is called.
def _import_torch_drift():
    from sing.efgp_drift import EFGPDrift, _FrozenEFGPDrift as _FED_torch
    return EFGPDrift, _FED_torch


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


# ============================================================================
# JAX-native EM loop (lax.scan'd inner E-step, no torch round-trip)
# ============================================================================
def _build_jit_estep_scan_jax(*, D, T, t_grid, lik, sigma, sigma_drift_sq,
                                S_marginal, n_estep_iters):
    """Compile a function that runs ``n_estep_iters`` inner E-step iterations
    via ``lax.scan``, all JAX, no torch.

    The spectral grid is passed AS AN INPUT (not closure-captured) so the
    same compiled artifact is reused for any grid with the same
    ``mtot_per_dim``.  ``mtot_per_dim`` is a ``static_argnames``: JAX
    automatically caches one compiled artifact per ``mtot_per_dim`` value.

    For an EM run that explores ℓ across only a handful of grid sizes,
    this reduces compilation cost from "every M-step" to "first time we
    see each grid size".
    """
    inputs = jnp.zeros((T, 1))
    input_effect = jnp.zeros((D, 1))
    del_t = (t_grid[1:] - t_grid[:-1])

    nat_to_marg_b = jax.jit(vmap(natural_to_marginal_params))

    @partial(jax.jit, static_argnames=('mtot_per_dim',))
    def scan_estep(natural_params, init_params, ys_obs, t_mask,
                    output_params, key, rho,
                    xis_flat, ws, h_per_dim, xcen, mtot_per_dim):
        # Reconstruct the JaxGridState pytree inside the trace.  M and d
        # are derived from xis_flat shape, so they are static at trace time.
        grid = jp.JaxGridState(
            xis_flat=xis_flat, ws=ws, h_per_dim=h_per_dim,
            mtot_per_dim=mtot_per_dim, xcen=xcen,
            M=int(xis_flat.shape[0]), d=int(xis_flat.shape[1]),
        )
        trial_mask = jnp.ones((1, T), dtype=bool)

        def one_iter(carry, key_iter):
            nat_p = carry
            mp_b, _ = vmap(natural_to_marginal_params)(nat_p, trial_mask)
            ms = mp_b['m'][0]
            Ss = mp_b['S'][0]
            SSs = mp_b['SS'][0]

            key_qf, key_grad = jr.split(key_iter, 2)
            mu_r, Ef, Eff, Edfdx = jpd.qf_and_moments_jax(
                ms, Ss, SSs, del_t, grid, key_qf,
                sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
                D_lat=D, D_out=D,
            )
            frozen = _FrozenEFGPDrift(
                latent_dim=D, t_grid=t_grid,
                Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)
            mean_params_b, _ = vmap(natural_to_mean_params)(nat_p, trial_mask)
            lik_g = vmap(lambda mp_, tm_, ys_: nat_grad_likelihood(
                mp_, tm_, ys_, lik, output_params))(mean_params_b, t_mask, ys_obs)
            tr_g = vmap(lambda k, mp_, tm_, ip_: nat_grad_transition(
                k, frozen, None, {}, tm_, ip_, t_grid, mp_,
                inputs, input_effect, sigma))(jr.split(key_grad, 1),
                                              mean_params_b, trial_mask, init_params)
            new_nat_p = {
                'J': (1 - rho) * nat_p['J']
                     + rho * (lik_g['J'] + tr_g['J']),
                'L': (1 - rho) * nat_p['L'] + rho * tr_g['L'],
                'h': (1 - rho) * nat_p['h']
                     + rho * (lik_g['h'] + tr_g['h']),
            }
            return new_nat_p, None

        keys = jr.split(key, n_estep_iters)
        final_nat_p, _ = jax.lax.scan(one_iter, natural_params, keys)
        return final_nat_p

    return scan_estep, nat_to_marg_b


def fit_efgp_sing_jax(
    *,
    likelihood: Likelihood,
    t_grid: jnp.ndarray,
    output_params: Dict[str, jnp.ndarray],
    init_params: InitParams,
    latent_dim: int,
    lengthscale: float,
    variance: float,
    sigma: float = 1.0,
    sigma_drift_sq: Optional[float] = None,   # defaults to sigma**2
    eps_grid: float = 1e-2,
    S_marginal: int = 2,
    n_em_iters: int = 8,
    n_estep_iters: int = 10,
    rho_sched: Optional[jnp.ndarray] = None,
    learn_emissions: bool = True,
    update_R: bool = True,
    learn_kernel: bool = False,
    n_mstep_iters: int = 4,
    mstep_lr: float = 0.05,
    n_hutchinson_mstep: int = 4,
    kernel_warmup_iters: int = 2,
    pin_grid: bool = False,                # if True, build grid once and never rebuild
    pin_grid_eps: Optional[float] = None,  # if pin_grid, eps_grid override (defaults to eps_grid)
    pin_grid_lengthscale: Optional[float] = None,  # if pin_grid, ls override
    X_template: Optional[jnp.ndarray] = None,
    seed: int = 0,
    true_xs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[MarginalParams, NaturalParams, Dict[str, jnp.ndarray],
           InitParams, EFGPEMHistory]:
    """Fully JAX-native fit: inner E-step is ``lax.scan``'d inside ``jit``.

    Takes raw SE-kernel hypers (lengthscale, variance) instead of a torch-
    backed ``EFGPDrift`` instance, so this function never imports torch.
    Avoids the ``pytorch_finufft`` / ``jax_finufft`` libfinufft load-order
    segfault: callers must not import any torch-using SING modules in the
    same process.
    """
    if rho_sched is None:
        rho_sched = jnp.linspace(0.1, 0.9, n_em_iters)
    rho_sched = jnp.asarray(rho_sched)
    if sigma_drift_sq is None:
        sigma_drift_sq = float(sigma) ** 2

    ys_obs = likelihood.ys_obs
    t_mask = likelihood.t_mask
    assert ys_obs.shape[0] == 1, "v0 supports a single trial."
    n_trials, T, N = ys_obs.shape
    D = latent_dim
    trial_mask = jnp.ones((n_trials, T), dtype=bool)

    ls = float(lengthscale)
    var = float(variance)

    # Build a *representative* cloud at the prior marginals so the spectral
    # grid (and thus the jit'd graph) is built once.  Caller can supply a
    # custom one if the latent posterior is expected to roam beyond the
    # prior covariance.
    if X_template is None:
        diag_V0 = jnp.diag(jnp.asarray(init_params['V0'][0]))
        X_template = (jnp.sqrt(diag_V0)[None, :]
                      * jnp.linspace(-2., 2., max(T, 16))[:, None]
                      + jnp.asarray(init_params['mu0'][0])[None, :])

    # ---- Initialise natural params with a "diffusion-only" SDE ----
    # Avoid SING's initialize_params (which would import torch via
    # the drift module).  Use a Brownian-like initial chain instead.
    nat_to_marg_b = jax.jit(vmap(natural_to_marginal_params))
    A_init = jnp.zeros((T - 1, D, D))
    b_init = jnp.zeros((T, D)).at[0].set(jnp.asarray(init_params['mu0'][0]))
    Q_init = jnp.tile((sigma ** 2) * jnp.eye(D), (T, 1, 1)).at[0].set(
        jnp.asarray(init_params['V0'][0]))
    from sing.utils.sing_helpers import dynamics_to_natural_params
    nat_one = dynamics_to_natural_params(A_init, b_init, Q_init, trial_mask[0])
    natural_params = jax.tree_util.tree_map(lambda x: x[None], nat_one)

    init_key, key = jr.split(jr.PRNGKey(seed), 2)

    # Build the jit'd E-step ONCE.  The grid (xis, ws, h, xcen) is now a
    # JIT INPUT, and ``mtot_per_dim`` is a static_argnames — JAX
    # automatically caches one compiled artifact per ``mtot_per_dim``
    # value.
    scan_estep, _ = _build_jit_estep_scan_jax(
        D=D, T=T, t_grid=t_grid, lik=likelihood, sigma=sigma,
        sigma_drift_sq=sigma_drift_sq,
        S_marginal=S_marginal, n_estep_iters=n_estep_iters,
    )

    def _make_grid(ls_, var_, eps_):
        return jp.spectral_grid_se(ls_, var_, X_template, eps=eps_)

    if pin_grid:
        # Conservative grid that stays valid across the M-step's ℓ range.
        # Use the (optionally larger) ``pin_grid_lengthscale`` as the
        # grid-determining ls, with optionally tighter eps for safety.
        pin_ls = pin_grid_lengthscale if pin_grid_lengthscale is not None else ls
        pin_eps = pin_grid_eps if pin_grid_eps is not None else eps_grid
        grid = _make_grid(pin_ls, var, pin_eps)
    else:
        grid = _make_grid(ls, var, eps_grid)

    if verbose:
        print(f"  [jax] M={grid.M}, mtot_per_dim={grid.mtot_per_dim}, "
              f"pin_grid={pin_grid}")

    history = EFGPEMHistory()
    log_ls = float(np.log(ls))
    log_var = float(np.log(var))

    for it in range(n_em_iters):
        rho_jax = jnp.asarray(float(rho_sched[it]), dtype=jnp.float32)
        em_key, key = jr.split(key, 2)
        natural_params = scan_estep(natural_params, init_params,
                                     ys_obs, t_mask, output_params,
                                     em_key, rho_jax,
                                     grid.xis_flat, grid.ws,
                                     grid.h_per_dim, grid.xcen,
                                     mtot_per_dim=grid.mtot_per_dim)

        # ---- M-step ----
        mp_b, _ = nat_to_marg_b(natural_params, trial_mask)
        ms_t = mp_b['m'][0]
        Ss_t = mp_b['S'][0]
        SSs_t = mp_b['SS'][0]

        # 1) Closed-form (C, d, R) for Gaussian emissions
        if learn_emissions:
            C_new, d_new, R_new = update_emissions_gaussian(
                ms_t, Ss_t, ys_obs[0], t_mask[0], update_R=update_R)
            output_params = {**output_params, 'C': C_new, 'd': d_new}
            if R_new is not None:
                output_params['R'] = R_new
        init_params = vmap(update_init_params)(ms_t[None, 0], Ss_t[None, 0])

        # 2) Collapsed kernel M-step (after warmup so q(f) isn't trivial)
        if learn_kernel and it >= kernel_warmup_iters:
            mstep_key, key = jr.split(key, 2)
            del_t = t_grid[1:] - t_grid[:-1]
            mu_r, X_pseudo, top = jpd.compute_mu_r_jax(
                ms_t, Ss_t, SSs_t, del_t, grid, mstep_key,
                sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
                D_lat=D, D_out=D,
            )
            # z_r = h_θ,r / ws  (so M-step varies hypers via ws and recovers h_r = ws ⊙ z_r)
            ws_real_c = grid.ws.real.astype(grid.ws.dtype)
            ws_safe = jnp.where(jnp.abs(ws_real_c) < 1e-30,
                                 jnp.array(1e-30, dtype=ws_real_c.dtype),
                                 ws_real_c)
            # h_θ_0,r = A_θ_0 μ at the E-step optimum
            from sing.efgp_jax_primitives import make_A_apply
            A_at_theta0 = make_A_apply(grid.ws, top, sigmasq=1.0)
            h_r0 = jax.vmap(A_at_theta0)(mu_r)
            z_r = h_r0 / ws_safe
            log_ls_new, log_var_new, _ = jpd.m_step_kernel_jax(
                log_ls, log_var,
                mu_r_fixed=mu_r, z_r=z_r, top=top,
                xis_flat=grid.xis_flat, h_per_dim=grid.h_per_dim,
                D_lat=D, D_out=D,
                n_inner=n_mstep_iters, lr=mstep_lr,
                n_hutchinson=n_hutchinson_mstep, include_trace=True,
                key=mstep_key,
            )
            new_ls = float(jnp.exp(log_ls_new))
            new_var = float(jnp.exp(log_var_new))
            # Avoid pathological collapse (a v0 safety belt; the cold-start
            # collapse is documented as a known issue)
            if 0.05 < new_ls < 50.0 and 0.01 < new_var < 100.0:
                ls = new_ls
                var = new_var
                log_ls = float(log_ls_new)
                log_var = float(log_var_new)
                # Rebuild only the GRID (a cheap numpy bisection + JAX
                # array build), not the jit'd scan_estep.  JAX caches one
                # compiled artifact per ``mtot_per_dim`` value, so as long
                # as the kernel range visits only a few discrete grid
                # sizes during the M-step, recompilation is rare.  With
                # ``pin_grid=True`` the grid is fixed and we never rebuild.
                if not pin_grid:
                    grid = _make_grid(ls, var, eps_grid)

        if true_xs is not None:
            rmse_lat = float(np.sqrt(np.mean(
                (np.asarray(ms_t) - true_xs) ** 2)))
            history.latent_rmse.append(rmse_lat)
        history.lengthscale.append(ls)
        history.variance.append(var)

        if verbose:
            extras = []
            if history.latent_rmse:
                extras.append(f"latent_rmse={history.latent_rmse[-1]:.3f}")
            extras.append(f"ℓ={ls:.3f}")
            extras.append(f"σ²={var:.3f}")
            print(f"[jax] EM {it + 1:2d}/{n_em_iters}  rho={float(rho_sched[it]):.3f}  "
                  + "  ".join(extras))

    marginal_params, _ = nat_to_marg_b(natural_params, trial_mask)
    return marginal_params, natural_params, output_params, init_params, history
