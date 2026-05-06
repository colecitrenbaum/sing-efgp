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
def _build_jit_estep_scan_jax(*, K, D, T, t_grid, trial_mask,
                                lik, sigma, sigma_drift_sq,
                                S_marginal, n_estep_iters,
                                qf_cg_tol, qf_max_cg_iter, qf_nufft_eps,
                                estep_method='mc',
                                gmix_fine_N=None, gmix_stencil_r=None):
    """Compile a function that runs ``n_estep_iters`` inner E-step iterations
    via ``lax.scan``, all JAX, no torch.

    The spectral grid is passed AS AN INPUT (not closure-captured) so the
    same compiled artifact is reused for any grid with the same
    ``mtot_per_dim``.  ``mtot_per_dim`` is a ``static_argnames``: JAX
    automatically caches one compiled artifact per ``mtot_per_dim`` value.

    Multi-trial: ``K`` and ``trial_mask`` are baked into the closure;
    each new ``K`` triggers a new JIT artifact (one per K, per
    mtot_per_dim).  The inner SING natural-grad operates per-trial via
    ``vmap``; the q(f) update flattens sources across trials.
    """
    inputs = jnp.zeros((T, 1))
    input_effect = jnp.zeros((D, 1))
    del_t = (t_grid[1:] - t_grid[:-1])
    trial_mask_b = jnp.broadcast_to(jnp.asarray(trial_mask, dtype=bool), (K, T))

    nat_to_marg_b = jax.jit(vmap(natural_to_marginal_params))

    @partial(jax.jit, static_argnames=('mtot_per_dim',))
    def scan_estep_inner_refresh(natural_params, init_params, ys_obs, t_mask,
                                 output_params, key, rho,
                                 xis_flat, ws, h_per_dim, xcen, mtot_per_dim):
        # Reconstruct the JaxGridState pytree inside the trace.  M and d
        # are derived from xis_flat shape, so they are static at trace time.
        grid = jp.JaxGridState(
            xis_flat=xis_flat, ws=ws, h_per_dim=h_per_dim,
            mtot_per_dim=mtot_per_dim, xcen=xcen,
            M=int(xis_flat.shape[0]), d=int(xis_flat.shape[1]),
        )

        def one_iter(carry, key_iter):
            nat_p = carry
            mp_b, _ = vmap(natural_to_marginal_params)(nat_p, trial_mask_b)
            ms = mp_b['m']                           # (K, T, D)
            Ss = mp_b['S']                           # (K, T, D, D)
            SSs = mp_b['SS']                         # (K, T-1, D, D)

            key_qf, key_grad = jr.split(key_iter, 2)
            if estep_method == 'mc':
                mu_r, Ef, Eff, Edfdx = jpd.qf_and_moments_jax(
                    ms, Ss, SSs, del_t, trial_mask_b, grid, key_qf,
                    sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
                    D_lat=D, D_out=D,
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
                    nufft_eps=qf_nufft_eps,
                )
            elif estep_method == 'gmix':
                mu_r, Ef, Eff, Edfdx = jpd.qf_and_moments_gmix_jax(
                    ms, Ss, SSs, del_t, trial_mask_b, grid,
                    sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D,
                    fine_N=gmix_fine_N, stencil_r=gmix_stencil_r,
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
                    nufft_eps=qf_nufft_eps,
                )
            else:
                raise ValueError(f"unknown estep_method {estep_method!r}")
            # Per-trial frozen drift: pytree with (K, T, ...) leaves; vmap
            # below slices the leading axis so each lambda invocation sees a
            # logical single-trial FrozenEFGPDrift.
            frozen_K = _FrozenEFGPDrift(
                latent_dim=D, t_grid=t_grid,
                Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)
            mean_params_b, _ = vmap(natural_to_mean_params)(nat_p, trial_mask_b)
            lik_g = vmap(lambda mp_, tm_, ys_: nat_grad_likelihood(
                mp_, tm_, ys_, lik, output_params))(mean_params_b, t_mask, ys_obs)
            tr_g = vmap(lambda k, fr_, mp_, tm_, ip_: nat_grad_transition(
                k, fr_, None, {}, tm_, ip_, t_grid, mp_,
                inputs, input_effect, sigma))(jr.split(key_grad, K),
                                              frozen_K, mean_params_b,
                                              trial_mask_b, init_params)
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

    @jax.jit
    def scan_estep_frozen_qf(natural_params, init_params, ys_obs, t_mask,
                             output_params, key, rho,
                             Ef, Eff, Edfdx):
        frozen_K = _FrozenEFGPDrift(
            latent_dim=D, t_grid=t_grid,
            Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)

        def one_iter(carry, key_grad):
            nat_p = carry
            mean_params_b, _ = vmap(natural_to_mean_params)(nat_p, trial_mask_b)
            lik_g = vmap(lambda mp_, tm_, ys_: nat_grad_likelihood(
                mp_, tm_, ys_, lik, output_params))(mean_params_b, t_mask, ys_obs)
            tr_g = vmap(lambda k, fr_, mp_, tm_, ip_: nat_grad_transition(
                k, fr_, None, {}, tm_, ip_, t_grid, mp_,
                inputs, input_effect, sigma))(jr.split(key_grad, K),
                                              frozen_K, mean_params_b,
                                              trial_mask_b, init_params)
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

    return scan_estep_inner_refresh, scan_estep_frozen_qf, nat_to_marg_b


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
    # Closed-form Gaussian emission updates can be too aggressive while the
    # latent posterior is still close to the diffusion-only initialization.
    # Delay them until q(x) has sharpened enough to avoid a bad early
    # emissions/latents feedback loop.
    emission_warmup_iters: int = 8,
    learn_kernel: bool = False,
    n_mstep_iters: int = 4,
    mstep_lr: float = 0.05,
    n_hutchinson_mstep: int = 4,
    # If True, refresh q(f) on every inner SING iteration. If False,
    # compute q(f) once per outer EM iteration and freeze it through the
    # inner natural-gradient scan, matching SparseGP's variational-EM
    # structure more closely and improving stability.
    refresh_qf_every_estep_iter: bool = True,
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
    #   4) DEFAULT                     — adaptive-h, K matched to the
    #                                    initial tailored grid at the
    #                                    requested ``eps_grid``.
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
    # Diagnostic mode: rebuild a fully tailored spectral grid from the
    # current (ℓ, σ²) every outer EM iter instead of using the jit-friendly
    # fixed-K / adaptive-h policy. This can retrace when mtot changes, so it
    # is meant for comparison experiments rather than routine use.
    tailored_grid_every_iter: bool = False,
    K_per_dim: Optional[int] = None,
    K_min_lengthscale: Optional[float] = None,  # used to auto-pick K if not given
    K_eps: Optional[float] = None,         # tolerance for the K choice (defaults eps_grid)
    X_template: Optional[jnp.ndarray] = None,
    qf_cg_tol: float = 1e-5,
    qf_max_cg_iter: int = 2000,
    qf_nufft_eps: float = 6e-8,
    # E-step q(f)-update method:
    #   'gmix' (default): closed-form Gaussian-mixture spreader (variant A
    #     from efgp_estep_charfun.tex).  Deterministic, exact up to FFT
    #     truncation; no S_marginal MC noise.
    #   'mc' (legacy): pseudo-cloud Monte Carlo (S_marginal samples per t).
    estep_method: str = 'gmix',
    # Gaussian-mixture spreader knobs (only used when estep_method == 'gmix'):
    #   gmix_fine_N : int — FFT grid size per dim.  Picked automatically if
    #     None, based on max sigma seen at the initial q(x).  Recompiles if
    #     it changes between EM iters, so a generous initial value is best.
    #   gmix_stencil_r : int — Gaussian spread stencil half-width.  Picked
    #     automatically from gmix_n_sigma * sigma_max / h_grid if None.
    #   gmix_n_sigma : float — stencil radius in units of σ_max.  Truncation
    #     error decays as exp(-n²/2): n=4 → 3e-4, n=3 → 1e-2, n=2.5 → 4e-2,
    #     n=2 → 0.14, n=1.5 → 0.32.  Default n=1.5 matches the empirical
    #     wall-time optimum (r ≈ 8 in 2D): smaller r → cheaper spreads,
    #     CG iter count stays comparable to MC, and 33% truncation is
    #     still 2× more accurate than MC's S=2 shot noise (~0.7).  Bump
    #     to 2.0 if drift_rmse looks worse than expected; below 1.0 is
    #     unsafe (CG amplification of stencil bias).
    gmix_fine_N: Optional[int] = None,
    gmix_stencil_r: Optional[int] = None,
    gmix_n_sigma: float = 1.5,
    # If True, recompute stencil_r ONCE at iter `kernel_warmup_iters` based
    # on the current Σ.  This is mathematically a tighter stencil but
    # costs one JIT recompile (~10-15 s); empirically the savings (smaller
    # r → faster spreads in steady state) do not pay back the recompile
    # within a typical 25-iter EM run.  Default False.
    gmix_adapt_stencil: bool = False,
    seed: int = 0,
    true_xs: Optional[np.ndarray] = None,
    verbose: bool = True,
    # Optionally warm-start natural params from an earlier fit (e.g. a
    # final-stage refinement at tighter eps_grid).  Defaults to the
    # diffusion-only Brownian-chain init.
    init_natural_params: Optional[NaturalParams] = None,
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
    n_trials, T, N = ys_obs.shape
    D = latent_dim
    trial_mask = jnp.ones((n_trials, T), dtype=bool)

    ls = float(lengthscale)
    var = float(variance)

    # Build a representative cloud that sets the spectral grid support.
    # A prior-covariance-sized cloud is often too narrow in practice: the
    # adaptive-h grid can then be centered on a domain much smaller than the
    # actual latent trajectory, which hurts both q(f) quality and the kernel
    # M-step on broader nonlinear flows. Use at least a [-3, 3]^D box around
    # the initial mean, expanding further if the prior variance is larger.
    # Caller can still supply a custom template when they know a better
    # support box for the problem.
    #
    # Multi-trial: take the union bounding box across all trials' initial
    # means + 4·std halos so one shared spectral grid covers every trial.
    if X_template is None:
        V0_arr = jnp.asarray(init_params['V0'])               # (K, D, D)
        mu0_arr = jnp.asarray(init_params['mu0'])             # (K, D)
        diag_std0 = jnp.sqrt(jnp.diagonal(V0_arr, axis1=-2, axis2=-1))  # (K, D)
        half_span_per = jnp.maximum(
            jnp.full_like(diag_std0, 3.0),
            4.0 * diag_std0,
        )                                                      # (K, D)
        # Per-trial box, then union across K
        per_min = (mu0_arr - half_span_per).min(axis=0)        # (D,)
        per_max = (mu0_arr + half_span_per).max(axis=0)        # (D,)
        center = 0.5 * (per_min + per_max)
        half_span = 0.5 * (per_max - per_min)
        X_template = (
            center[None, :]
            + jnp.linspace(-1., 1., max(T, 16))[:, None] * half_span[None, :]
        )

    # ---- Initialise natural params ----
    nat_to_marg_b = jax.jit(vmap(natural_to_marginal_params))
    if init_natural_params is not None:
        natural_params = init_natural_params
    else:
        # Default: diffusion-only Brownian-chain seed (per-trial via vmap).
        from sing.utils.sing_helpers import dynamics_to_natural_params
        mu0_K = jnp.asarray(init_params['mu0'])                  # (K, D)
        V0_K = jnp.asarray(init_params['V0'])                    # (K, D, D)

        def _init_one(mu0_, V0_, tm_):
            A_init = jnp.zeros((T - 1, D, D))
            b_init = jnp.zeros((T, D)).at[0].set(mu0_)
            Q_init = jnp.tile((sigma ** 2) * jnp.eye(D), (T, 1, 1)).at[0].set(V0_)
            return dynamics_to_natural_params(A_init, b_init, Q_init, tm_)

        natural_params = vmap(_init_one)(mu0_K, V0_K, trial_mask)

    init_key, key = jr.split(jr.PRNGKey(seed), 2)

    # Defer jit'd scan construction until the grid is known (so gmix
    # path can size fine_N from h_spec).
    if estep_method not in ('mc', 'gmix'):
        raise ValueError(f"estep_method must be 'mc' or 'gmix', "
                         f"got {estep_method!r}")
    _est_kw = dict(
        K=n_trials, D=D, T=T, t_grid=t_grid, trial_mask=trial_mask,
        lik=likelihood, sigma=sigma,
        sigma_drift_sq=sigma_drift_sq,
        S_marginal=S_marginal, n_estep_iters=n_estep_iters,
        qf_cg_tol=qf_cg_tol, qf_max_cg_iter=qf_max_cg_iter,
        qf_nufft_eps=qf_nufft_eps,
        estep_method=estep_method,
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
    if use_adaptive_h and tailored_grid_every_iter:
        grid = _make_grid(ls, var, eps_grid)
        if verbose:
            print(f"  [jax] tailored-grid: M={grid.M}, "
                  f"mtot_per_dim={grid.mtot_per_dim}")
    elif use_adaptive_h:
        K_eps_used = K_eps if K_eps is not None else eps_grid
        if K_per_dim is None and K_min_lengthscale is None:
            init_grid = _make_grid(ls, var, K_eps_used)
            K_per_dim = (int(init_grid.mtot_per_dim[0]) - 1) // 2
            if verbose:
                print(f"  [jax] adaptive-h default: K_per_dim={K_per_dim} "
                      f"(matches initial tailored grid)")
        # Use cloud extent from X_template
        X_min = np.asarray(X_template.min(axis=0))
        X_max = np.asarray(X_template.max(axis=0))
        X_extent = float((X_max - X_min).max())
        xcen_arr = jnp.asarray(0.5 * (X_min + X_max))

        if K_per_dim is None:
            ls_min = float(K_min_lengthscale)
            K_per_dim = jp.choose_K_for_min_lengthscale(
                ls_min, var, X_extent, eps=K_eps_used, d=D)

        def _make_grid_adaptive(ls_, var_):
            return jp.spectral_grid_se_fixed_K(
                log_ls=jnp.log(jnp.asarray(ls_)),
                log_var=jnp.log(jnp.asarray(var_)),
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

    # Now that the grid is built, size the gmix spreader (if used) and
    # build the jit'd scan once.  fine_N / stencil_r baked into closure.
    if estep_method == 'gmix':
        from sing.efgp_gmix_spreader import (
            stencil_radius_for as _stencil_for, pick_grid_size as _pick_N)
        h_spec0 = float(grid.h_per_dim[0])
        # Initial Σ_max: from prior covariance V0 (covers cold-start regime
        # where q(x) is broad).  Multi-trial: take max eigenvalue across
        # all trials' V0 so the stencil covers the widest prior.
        # fine_N and stencil_r are STATIC; if the user wants to recompile
        # after warmup with tighter values, they can re-instantiate.
        V0_all = jnp.asarray(init_params['V0'])
        if V0_all.ndim == 2:
            V0_all = V0_all[None]                              # (1, D, D)
        sigma0_max = float(jnp.sqrt(jnp.max(jnp.linalg.eigvalsh(V0_all))))
        # Span the cloud extent we already used to build the grid:
        X_min = np.asarray(X_template.min(axis=0))
        X_max = np.asarray(X_template.max(axis=0))
        m_extent = float((X_max - X_min).max())
        if gmix_fine_N is None:
            gmix_fine_N = _pick_N(h_spec=h_spec0, m_extent=m_extent,
                                    sigma_max=sigma0_max)
        if gmix_stencil_r is None:
            h_grid = 1.0 / (gmix_fine_N * h_spec0)
            # Use the widest prior V0 across trials to size the stencil
            gmix_stencil_r = _stencil_for(V0_all,
                                            float(h_grid),
                                            n_sigma=gmix_n_sigma)
        # Round to a coarse ladder so JIT cache only sees a handful of
        # distinct values across the EM trajectory (each value compiles once).
        _STENCIL_LADDER = (3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
        def _round_r(r_raw):
            for x in _STENCIL_LADDER:
                if r_raw <= x:
                    return x
            return _STENCIL_LADDER[-1]
        # Empirical wall-time sweet spot: r ≈ 8 in 2D.  Smaller r reduces
        # spread cost but coarsens the BTTB approximation, which inflates
        # CG iteration count and *increases* total wall time despite the
        # cheaper spread.  Larger r is dominated by spread cost.  Floor
        # at 8 unless caller explicitly sets a smaller `gmix_stencil_r`.
        gmix_stencil_r = max(8, _round_r(int(gmix_stencil_r)))
        if verbose:
            print(f"  [jax] estep_method=gmix  fine_N={gmix_fine_N}  "
                  f"stencil_r={gmix_stencil_r}  "
                  f"(sigma0_max≈{sigma0_max:.3f}, h_spec0≈{h_spec0:.3f})")
        _est_kw['gmix_fine_N'] = int(gmix_fine_N)
        _est_kw['gmix_stencil_r'] = int(gmix_stencil_r)

    scan_estep_inner_refresh, scan_estep_frozen_qf, _ = \
        _build_jit_estep_scan_jax(**_est_kw)

    def _maybe_adapt_stencil(Ss_now):
        """Recompute gmix_stencil_r from current S; rebuild scan if it changed.

        Returns the (possibly-new) (refresh, frozen) scan closures.
        """
        nonlocal gmix_stencil_r, scan_estep_inner_refresh, scan_estep_frozen_qf
        if estep_method != 'gmix':
            return scan_estep_inner_refresh, scan_estep_frozen_qf
        h_grid_now = 1.0 / (gmix_fine_N * float(grid.h_per_dim[0]))
        sigma_max_now = float(jnp.sqrt(
            jnp.max(jnp.linalg.eigvalsh(Ss_now))))
        r_raw = int(np.ceil(gmix_n_sigma * sigma_max_now / h_grid_now))
        r_new = _round_r(r_raw)
        if r_new != gmix_stencil_r:
            if verbose:
                print(f"    [gmix adaptive] stencil_r {gmix_stencil_r} -> {r_new}  "
                      f"(σ_max={sigma_max_now:.3f}, h_grid={h_grid_now:.4f})")
            gmix_stencil_r = r_new
            _est_kw['gmix_stencil_r'] = int(gmix_stencil_r)
            new_refresh, new_frozen, _ = _build_jit_estep_scan_jax(**_est_kw)
            scan_estep_inner_refresh = new_refresh
            scan_estep_frozen_qf = new_frozen
        return scan_estep_inner_refresh, scan_estep_frozen_qf

    history = EFGPEMHistory()
    log_ls = float(np.log(ls))
    log_var = float(np.log(var))

    # Track whether we've already adapted the gmix stencil (to bound
    # JIT recompiles to ~1 per fit).
    _gmix_already_adapted = False
    for it in range(n_em_iters):
        rho_jax = jnp.asarray(float(rho_sched[it]))
        em_key, key = jr.split(key, 2)
        # Adapt gmix stencil radius ONCE, after kernel_warmup_iters.  At
        # that point Σ has shrunk meaningfully relative to the prior V0
        # so a tighter stencil is correct, and a SINGLE JIT recompile
        # (instead of one per ladder transition) keeps wall time low.
        if (estep_method == 'gmix' and gmix_adapt_stencil
                and not _gmix_already_adapted
                and it == max(1, kernel_warmup_iters)):
            mp_now_b, _ = nat_to_marg_b(natural_params, trial_mask)
            # Adapt to the largest current marginal across (K, T)
            S_now_max = mp_now_b['S'].reshape(-1, D, D)
            scan_estep_inner_refresh, scan_estep_frozen_qf = \
                _maybe_adapt_stencil(S_now_max)
            _gmix_already_adapted = True
        if refresh_qf_every_estep_iter:
            natural_params = scan_estep_inner_refresh(
                natural_params, init_params,
                ys_obs, t_mask, output_params,
                em_key, rho_jax,
                grid.xis_flat, grid.ws,
                grid.h_per_dim, grid.xcen,
                mtot_per_dim=grid.mtot_per_dim)
        else:
            mp_prev_b, _ = nat_to_marg_b(natural_params, trial_mask)
            ms_prev = mp_prev_b['m']                          # (K, T, D)
            Ss_prev = mp_prev_b['S']                          # (K, T, D, D)
            SSs_prev = mp_prev_b['SS']                        # (K, T-1, D, D)
            qf_key, scan_key = jr.split(em_key, 2)
            del_t = t_grid[1:] - t_grid[:-1]
            if estep_method == 'mc':
                _, Ef_prev, Eff_prev, Edfdx_prev = jpd.qf_and_moments_jax(
                    ms_prev, Ss_prev, SSs_prev, del_t, trial_mask, grid, qf_key,
                    sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
                    D_lat=D, D_out=D,
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
                    nufft_eps=qf_nufft_eps,
                )
            else:
                _, Ef_prev, Eff_prev, Edfdx_prev = jpd.qf_and_moments_gmix_jax(
                    ms_prev, Ss_prev, SSs_prev, del_t, trial_mask, grid,
                    sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D,
                    fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r),
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
                    nufft_eps=qf_nufft_eps,
                )
            natural_params = scan_estep_frozen_qf(
                natural_params, init_params,
                ys_obs, t_mask, output_params,
                scan_key, rho_jax,
                Ef_prev, Eff_prev, Edfdx_prev)

        # ---- M-step ----
        mp_b, _ = nat_to_marg_b(natural_params, trial_mask)
        ms_t = mp_b['m']                                      # (K, T, D)
        Ss_t = mp_b['S']                                      # (K, T, D, D)
        SSs_t = mp_b['SS']                                    # (K, T-1, D, D)

        # 1) Closed-form (C, d, R) for Gaussian emissions
        if learn_emissions and it >= emission_warmup_iters:
            C_new, d_new, R_new = update_emissions_gaussian(
                ms_t, Ss_t, ys_obs, t_mask, update_R=update_R)
            output_params = {**output_params, 'C': C_new, 'd': d_new}
            if R_new is not None:
                output_params['R'] = R_new
        init_params = vmap(update_init_params)(ms_t[:, 0], Ss_t[:, 0])

        # 2) Collapsed kernel M-step (after warmup so q(f) isn't trivial)
        if learn_kernel and it >= kernel_warmup_iters:
            mstep_key, key = jr.split(key, 2)
            del_t = t_grid[1:] - t_grid[:-1]
            # Build flat supersources from all trials' Stein quantities
            m_src, S_src, d_src, C_src, w_src = jpd._flatten_stein(
                ms_t, Ss_t, SSs_t, del_t, trial_mask)
            if estep_method == 'mc':
                mu_r, X_pseudo, top = jpd.compute_mu_r_jax(
                    m_src, S_src, d_src, C_src, w_src, grid, mstep_key,
                    sigma_drift_sq=sigma_drift_sq, S_marginal=S_marginal,
                    D_lat=D, D_out=D,
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
                    nufft_eps=qf_nufft_eps,
                )
            else:  # 'gmix'
                mu_r, _, top = jpd.compute_mu_r_gmix_jax(
                    m_src, S_src, d_src, C_src, w_src, grid,
                    sigma_drift_sq=sigma_drift_sq, D_lat=D, D_out=D,
                    fine_N=int(gmix_fine_N), stencil_r=int(gmix_stencil_r),
                    cg_tol=qf_cg_tol, max_cg_iter=qf_max_cg_iter,
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
                if use_adaptive_h and tailored_grid_every_iter:
                    grid = _make_grid(ls, var, eps_grid)
                elif use_adaptive_h:
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
            # ``ms_t`` is (K, T, D); accept ``true_xs`` as either (T, D) (K=1
            # back-compat) or (K, T, D).
            true_arr = np.asarray(true_xs)
            if true_arr.ndim == 2:
                true_arr = true_arr[None]
            rmse_lat = float(np.sqrt(np.mean(
                (np.asarray(ms_t) - true_arr) ** 2)))
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
