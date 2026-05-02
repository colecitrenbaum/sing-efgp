"""
JAX-native variational-EM driver for the EFGP-SING drift block.

This module provides ``fit_efgp_sing_jax``: a fully jit-compiled EM loop
whose inner E-step is ``lax.scan``'d through ``n_estep_iters`` SING
natural-grad updates with a Stein-corrected EFGP q(f) refresh on every
iteration.  Drift moments, q(f) update, and SING natural-grad live in a
single compiled graph — no torch round-trips.

Scope (v0)
----------
Single trial.  Gaussian likelihood with closed-form ``(C, d, R)`` update.
Kernel hyperparameters via the MAP-form collapsed M-step (see
:func:`sing.efgp_jax_drift.m_step_kernel_jax`).  Diffusion ``Σ`` and
input matrix ``B`` held fixed.
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

from sing.efgp_emissions import update_emissions_gaussian

from functools import partial

# JAX-native EFGP primitives + drift
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import FrozenEFGPDrift as _FrozenEFGPDrift


@dataclass
class EFGPEMHistory:
    """Per-iteration diagnostics from the EM loop."""
    latent_rmse: List[float] = field(default_factory=list)
    drift_rmse: List[float] = field(default_factory=list)
    lengthscale: List[float] = field(default_factory=list)
    variance: List[float] = field(default_factory=list)
    mstep_loss: List[float] = field(default_factory=list)


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
    # Warmup: run at least this many E-step iters before the first M-step.
    # With rho_sched starting at 0.01, the smoother needs ~8 outer EM iters
    # (64+ inner SING steps) before its pseudo-velocities are smooth enough
    # for the collapsed-ML M-step to correctly identify long ℓ.  Below ~5
    # iters the deterministic gradient dominates the Hutchinson log-det term
    # and ℓ collapses toward 0 (fitted to near-prior noise).  Set to 0 only
    # if you pre-warm the E-step externally or are running a very large rho.
    kernel_warmup_iters: int = 8,
    # Grid policy (in priority order):
    #   1) ``pin_grid=True``           — legacy: build once, never rebuild.
    #   2) ``K_per_dim`` set explicitly — adaptive-h with user-supplied K.
    #   3) ``K_min_lengthscale`` set    — adaptive-h, auto K from this min ℓ.
    #   4) DEFAULT                     — adaptive-h, K_per_dim=10
    #                                    (≈ 21 modes per dim).
    # Adaptive-h means: integer mode lattice ``mtot = 2K+1`` is fixed, but
    # the physical spacing ``h(θ)`` adapts to the current ℓ each outer EM
    # iter.  No JIT retrace as ℓ moves; spacing tracks the kernel.
    pin_grid: bool = False,                # if True, build grid once and never rebuild
    pin_grid_eps: Optional[float] = None,  # if pin_grid, eps_grid override (defaults to eps_grid)
    pin_grid_lengthscale: Optional[float] = None,  # if pin_grid, ls override
    # Adaptive pin: refresh pin ONCE after `adaptive_pin_after` EM iters,
    # to ls_now * `adaptive_pin_factor`.  Bounds the M dimension to ~the
    # right resolution for the converged ℓ.  Costs one JAX retrace.
    adaptive_pin_after: Optional[int] = None,
    adaptive_pin_factor: float = 0.5,
    K_per_dim: Optional[int] = None,
    K_min_lengthscale: Optional[float] = None,  # used to auto-pick K if not given
    K_eps: Optional[float] = None,         # tolerance for the K choice (defaults eps_grid)
    X_template: Optional[jnp.ndarray] = None,
    seed: int = 0,
    true_xs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[MarginalParams, NaturalParams, Dict[str, jnp.ndarray],
           InitParams, EFGPEMHistory]:
    """Fully JAX-native fit: inner E-step is ``lax.scan``'d inside ``jit``.

    Takes raw SE-kernel hypers (lengthscale, variance) directly, so this
    function never imports torch.  Avoids the ``pytorch_finufft`` /
    ``jax_finufft`` libfinufft load-order segfault: callers must not
    import any torch-using SING modules in the same process.
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
    # Use a Brownian-like initial chain: we don't import any torch-backed
    # drift module to seed natural params.
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

    # ---- Grid policy ----
    # Default: adaptive-h (fixed integer lattice ``mtot=2K+1``, h(θ) adapts
    # each EM iter so spacing tracks the current ℓ).  Triggered unless the
    # caller explicitly opts into the legacy ``pin_grid=True``.  When
    # neither ``K_per_dim`` nor ``K_min_lengthscale`` is given, auto-pick K
    # from the initial ℓ with 2× headroom for shrinkage.
    use_adaptive_h = (not pin_grid)
    if use_adaptive_h:
        K_eps_used = K_eps if K_eps is not None else eps_grid
        if K_per_dim is None and K_min_lengthscale is None:
            K_per_dim = 10
            if verbose:
                print(f"  [jax] adaptive-h default: K_per_dim=10 "
                      f"(21 modes/dim)")
        # Use cloud extent from X_template
        X_min = np.asarray(X_template.min(axis=0))
        X_max = np.asarray(X_template.max(axis=0))
        X_extent = float((X_max - X_min).max())
        xcen_arr = jnp.asarray(0.5 * (X_min + X_max), dtype=jnp.float32)

        if K_per_dim is None:
            ls_min = float(K_min_lengthscale)
            K_per_dim = jp.choose_K_for_min_lengthscale(
                ls_min, var, X_extent, eps=K_eps_used, d=D)

        def _make_grid_adaptive(ls_, var_):
            return jp.spectral_grid_se_fixed_K(
                log_ls=jnp.log(jnp.asarray(ls_, dtype=jnp.float32)),
                log_var=jnp.log(jnp.asarray(var_, dtype=jnp.float32)),
                K_per_dim=K_per_dim, X_extent=X_extent, xcen=xcen_arr,
                d=D, eps=K_eps_used,
            )

        grid = _make_grid_adaptive(ls, var)
        if verbose:
            print(f"  [jax] adaptive-h: K={K_per_dim}, M={grid.M}, "
                  f"mtot_per_dim={grid.mtot_per_dim}")
    else:
        # Legacy ``pin_grid=True``: build once, never rebuild.  Use the
        # (optionally larger) ``pin_grid_lengthscale`` as the grid-determining
        # ls, with optionally tighter eps for safety.
        pin_ls = pin_grid_lengthscale if pin_grid_lengthscale is not None else ls
        pin_eps = pin_grid_eps if pin_grid_eps is not None else eps_grid
        grid = _make_grid(pin_ls, var, pin_eps)
        if verbose:
            print(f"  [jax] M={grid.M}, mtot_per_dim={grid.mtot_per_dim}, "
                  f"pin_grid=True (legacy)")

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
                # Rebuild the grid for the new ℓ.  Default adaptive-h
                # keeps ``mtot`` fixed (no JIT retrace) and only updates
                # spacing ``h(θ)``.  Legacy ``pin_grid=True`` skips this.
                if use_adaptive_h:
                    grid = _make_grid_adaptive(ls, var)

        # Adaptive pin: ONCE, at iteration `adaptive_pin_after`, refresh
        # the pinned grid to ls_now * adaptive_pin_factor.  This triggers
        # ONE new JAX compilation (since mtot_per_dim is static_argnames)
        # but the old artifact stays cached.  Lets the user start with a
        # crude pin and get a well-resolved grid post-warmup.
        if (pin_grid and adaptive_pin_after is not None
                and it == adaptive_pin_after and learn_kernel):
            new_pin_ls = ls * adaptive_pin_factor
            old_M = grid.M
            grid = _make_grid(new_pin_ls, 1.0, pin_eps)
            if verbose:
                print(f"[adaptive pin] iter {it+1}: pin_ls "
                      f"-> {new_pin_ls:.3f}  (M {old_M} -> {grid.M})")

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
