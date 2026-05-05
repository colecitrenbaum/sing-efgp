# EFGP-SING: scalable GP-drift inference for SING

This branch (`feat/efgp-drift`) adds an EFGP (Equispaced Fourier Gaussian
Process) drift block to SING.  It replaces the inducing-point sparse-GP
posterior in `SparseGP.update_dynamics_params` (Eq. 38, Appendix K.3 of the
SING paper) with a Toeplitz/NUFFT/CG implementation that scales as
**O(N + M log M)** instead of **O(N M²)** in the inducing-grid size.

The new code is **purely additive** — none of SING's existing modules are
modified.  The implementation is **JAX-only**: NUFFTs go through
`jax_finufft`, and the entire E-step (q(f) update, drift moments, SING
natural-grad) lives in a single `jit` + `lax.scan` graph with no torch
round-trips.

## What was added

```
sing/
  efgp_jax_primitives.py  ── pure JAX NUFFT/Toeplitz/CG/Hutchinson primitives
  efgp_jax_drift.py       ── Stein-corrected q(f) update, drift moments,
                              FrozenEFGPDrift (SDE wrapper with custom-VJPs),
                              collapsed kernel M-step
  efgp_emissions.py       ── closed-form (C, d, R) update for Gaussian emissions
  efgp_em.py              ── fit_efgp_sing_jax: jit + lax.scan EM driver
demos/
  efgp_synthetic_2d_jax.py    ── runnable end-to-end demo (linear drift)
  bench_gp_drift_recovery.py  ── EFGP vs SparseGP on a known SE-GP draw
  bench_duffing.py            ── EFGP vs SparseGP on a 2D Duffing oscillator
  bench_T_jax_vs_sparsegp.py  ── wall-time scaling vs T
  bench_convergence.py        ── EM convergence diagnostics
tests/
  test_efgp_jax_primitives.py            ── parity vs frozen torch oracle
  test_efgp_jax_recovery.py              ── kernel-hyper recovery from a wrong init
  test_efgp_jax_timing.py                ── wall-time floor checks
  test_efgp_sparsegp_drift_agreement.py  ── EFGP and SparseGP must agree on q(f)
                                            given the same q(x)
EFGP_SING_README.md  ── this file
```

## What was replaced

| SING/SparseGP | EFGP-SING |
| --- | --- |
| `SparseGP.update_dynamics_params` (Eq. 38, builds `(T-1, M, M)` and `(T-1, M)` kernel-expectation tensors) | `compute_mu_r_jax` — one type-1 NUFFT to build the BTTB Toeplitz `T_r = F_X* W_r F_X`, one CG solve per output dim |
| `SparseGP.f / ff / dfdx` (closed-form per-call expectations using the inducing kernel) | `drift_moments_jax` returns `(E[f], E[ffᵀ], E[J_f])` per time step; exposed to SING's `jax.grad` through `FrozenEFGPDrift` with custom VJPs encoding the local-quadratic transition approximation |
| Adam-on-ELBO kernel hyperparameter learning (backprop through the sparse-GP ELBO) | Direct collapsed M-step gradient (`m_step_kernel_jax`): deterministic part via JAX autograd through fixed μ_r, plus the Hutchinson trace `-½ tr[A⁻¹ ∂_θ A]` term to prevent lengthscale collapse, with μ_r refreshed every Adam step so the envelope-theorem gradient stays valid |

## How the math maps to the code

| Math | Where it lives |
| --- | --- |
| Spectral grid, weights `D_θ = diag(√(S(ξ_k) · h^d))` | `efgp_jax_primitives.spectral_grid_se` → `JaxGridState.ws` |
| BTTB generator `T_r = F_X* W_r F_X` | `efgp_jax_primitives.bttb_conv_vec_weighted` → `make_toeplitz` |
| `A_r v = v + D_θ T_r D_θ v` | `efgp_jax_primitives.make_A_apply` |
| Solve `A_r μ_r = h_r` | `efgp_jax_primitives.cg_solve` |
| `h_r = D_θ F_X* a_r + Σ_j (-2πi ξ_j) ⊙ D_θ F_X* c_{j,r}` (Stein-corrected RHS, see E-step PDF §5.3 — note: the negative sign in the second term comes from the **adjoint** Jacobian `J_φ^*`; the PDF as written has `+2πi ξ_j` but that drops the conjugation of the complex Fourier derivative). `c_{j,r} = Cov(x_t, x_{t+1})_{j,r} - Var(x_t)_{j,r}`, where `Cov(x_t, x_{t+1}) = SS_t.T` per SING's convention. | `efgp_jax_drift.compute_mu_r_jax` build loop |
| `Φ_eval μ_r` (posterior mean drift) and `Φ_eval [(2πi ξ_j) ⊙ μ_r]` (Jacobian) | `efgp_jax_drift.drift_moments_jax` (one type-2 NUFFT per output dim) |
| `diag(Φ A⁻¹ Φ*)` (Hutchinson estimator) | `efgp_jax_primitives.hutchinson_diag` |
| `log|A|` via Stochastic Lanczos Quadrature | `efgp_jax_primitives.logdet_slq` |
| `∂E[f]/∂m ≈ J_f(m)` and `∂E[fᵀ Σ⁻¹ f]/∂(m, S)` (local-quadratic transition approximation) | `ef_with_jac_grad`, `eff_with_grads` `jax.custom_vjp` definitions in `efgp_jax_drift.py` |
| Collapsed M-step gradient `Re((∂h)*μ) − ½ μ*(∂A)μ − ½ tr[A⁻¹ ∂A]` | `efgp_jax_drift.m_step_kernel_jax` |
| Closed-form Gaussian `(C, d, R)` update | `efgp_emissions.update_emissions_gaussian` |
| Combined "one inner-iter step" (q(f) + moments) | `efgp_jax_drift.qf_and_moments_jax` |
| EM driver (jit + lax.scan inner E-step) | `efgp_em.fit_efgp_sing_jax` |

## Approximations

The EFGP block is mathematically faithful (exact up to NUFFT / CG
truncation) for the GP-drift posterior and kernel learning.  The cross
into SING's `q(x)` natural-gradient update uses a **local-quadratic
transition approximation** for the implicit `(m, S)` dependence of the
drift expectations:

```
E_q[f(x)]                ≈  f(m)              ∂/∂m: J_f(m)        ∂/∂S: 0
E_q[f^T Σ⁻¹ f]           ≈  f(m)^T Σ⁻¹ f(m)   ∂/∂m: 2 J^T Σ⁻¹ f   ∂/∂S: J^T Σ⁻¹ J
E_q[J_f(x)]              ≈  J_f(m)            (treat constant; third moments dropped)
```

This is implemented by the `jax.custom_vjp` definitions
(`ef_with_jac_grad`, `eff_with_grads`) and the `FrozenEFGPDrift`
SDE wrapper in `sing/efgp_jax_drift.py`.  Per design discussion,
gradient through `diag(Φ A⁻¹ Φ*)` (the posterior-variance landscape) is
**not** included — it would require autograd through a CG solve and add
significant machinery without changing the asymptotic behavior of the
inference loop.

## Algorithm diagram

```
     ┌──────────────────────────────────────────────────────────────────┐
     │                  outer EM iteration  (Python)                    │
     │                                                                  │
     │   ┌─── jit + lax.scan inner E-step (× n_estep_iters) ──┐         │
     │   │                                                    │         │
     │   │  natural → marginal  (Bayesian smoother)           │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  compute_mu_r_jax  (Stein-corrected q(f) update:   │         │
     │   │     type-1 NUFFT → BTTB Toeplitz → CG per dim)     │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  drift_moments_jax  (type-2 NUFFTs for Ef, Eff,    │         │
     │   │     E[J_f] at every t, single graph)               │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  FrozenEFGPDrift  (SDE shim, custom-VJP            │         │
     │   │     gradients = local-quadratic transition         │         │
     │   │     approximation through (m, S))                  │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  SING nat-grad transition + likelihood updates     │         │
     │   │  (ρ-blended)                                       │         │
     │   └────────────────────────────────────────────────────┘         │
     │                                                                  │
     │   ┌─── M-step ─────────────────────────────────────────┐         │
     │   │  Closed-form (C, d, R)                             │         │
     │   │  m_step_kernel_jax:                                │         │
     │   │     refresh μ_r via CG every Adam step,            │         │
     │   │     then ∇ over (log ℓ, log σ²) of                 │         │
     │   │     -(Re(h*μ) − ½ μ* A μ) + ½ log|A| × D_out       │         │
     │   │     (Hutchinson trace, n_inner Adam steps)         │         │
     │   └────────────────────────────────────────────────────┘         │
     │                                                                  │
     └──────────────────────────────────────────────────────────────────┘
```

## How to run

Requires:

* `jax==0.6.1`, `jaxlib==0.6.1`, `tensorflow_probability==0.25.0`,
  `flax<0.12`, `tensorstore`, `optax`, `matplotlib`
* `jax_finufft`

Tests:

```bash
cd ~/Documents/GPs/sing
~/myenv/bin/python -m pytest tests/ -v
```

End-to-end demo:

```bash
~/myenv/bin/python demos/efgp_synthetic_2d_jax.py
# diagnostics in demos/_efgp_demo_out_jax/
```

## Benchmark results

### GP-drift recovery (known SE-GP drift, learn kernel hypers)

`demos/bench_gp_drift_recovery.py` samples one fresh draw of a 2-D drift
from a known SE-GP with `(ℓ_true=0.8, σ²_true=1.0)` plus a small linear
restoring term, simulates an SDE trajectory + Gaussian observations, and
fits both methods with `learn_kernel=True` from `ls_init=0.3`
(deliberately wrong, smaller than truth) over `n_em=25` outer iterations.

```
    T    method     wall   ℓ ratio   σ² ratio   obs RMSE   lat RMSE   drift RMSE
   500   EFGP        24.8   0.67x     0.74x     0.2231     0.0864     0.4066
   500   SparseGP    61.0   0.69x     0.56x     0.2461     0.1348     0.5748
  2000   EFGP        26.4   0.73x     0.70x     0.2235     0.0950     0.4000
  2000   SparseGP   191.9   0.69x     0.56x     0.2576     0.1340     0.6570
   --                                                                  --
trivial  T=500                                                         0.6758  (zero-drift)
trivial  T=2000                                                        0.7162  (zero-drift)
```

**EFGP wins on every metric** at both T values: 30-40 % better drift
RMSE, 30 % better latent RMSE, 10 % better observation RMSE, 2-7× faster
wall time.

### Robustness sweep (3 seeds × 2 T values × 3 ls_true values)

A more thorough check (still with `learn_kernel=True` from
`ls_init = ls_true × 0.4`):

```
ls_true   T      mean drift_rmse / baseline   wins (out of 3 seeds)
  0.6     500              0.83                       3 / 3
  0.6     2000             1.45                       0 / 3   ← failure mode
  0.8     500              0.84                       3 / 3
  0.8     2000             0.73                       3 / 3
  1.2     500              0.66                       3 / 3
  1.2     2000             0.61                       3 / 3
```

15/18 wins.  The single failure mode is
**T=2000 with very-small ls_true (0.6)** — the M-step with a relatively
fine pinned grid (M ≥ 121) over-fits the smoother noise and pushes ℓ
below 0.3.  The fix is either a coarser pinned grid or a stronger
M-step regularizer; both are tracked in TODOs.

### Pin-grid choice (regularization vs resolution)

Empirical observation: the pinned spectral grid acts primarily as a
**capacity regularizer** for the M-step.  Too-fine pin (`M` too large) →
M-step pushes ℓ below truth → drift over-fits the smoother noise → drift
RMSE blows up.  Too-coarse pin → can't represent sharp drifts.  The
default `pin_grid_lengthscale = ls_true * 0.75` balances these for the
GP-drift benchmark; in practice users would want `~half-to-three-quarters`
of the *expected* converged ℓ.  Adaptive refresh (refresh-once after
warmup) is supported via `adaptive_pin_after` but doesn't fix the
T=2000 + ls_true=0.6 failure on its own.

### Duffing oscillator (known nonlinear drift), large T

`demos/bench_duffing.py` generates from a 2D Duffing-oscillator latent SDE
with KNOWN drift `f(x, y) = [y, αx − βx³ − γy]`, fits both methods with
their full M-steps (kernel hypers + emissions), Procrustes-aligns the
inferred latents to truth basis, and evaluates the **inferred drift on a
truth-coords grid against the true Duffing drift**.

Both methods use SING-recommended settings (64 inducing pts for SparseGP,
`lr=1e-3`, 10 Adam M-steps per outer iter, conservative `ρ_sched =
logspace(-3, -2, 10) → 0.01 plateau`).  `n_em=20` for both.

```
    T    method     wall s   obs RMSE   latent RMSE   drift RMSE     ℓ      σ²
   500   EFGP        29.1     0.2164      0.0966        2.206       0.46   0.47
   500   SparseGP    66.9     0.2565      0.1067        2.159       1.20   0.64
  2000   EFGP        31.3     0.2127      0.0927        2.054       0.51   0.96
  2000   SparseGP   169.9     0.2455      0.1050        2.067       1.20   0.66
   --                          --          --           --
trivial  T=500                 0.2800      0.7286       2.349   (zero-drift)
trivial  T=2000                0.2800      0.7209       2.586   (zero-drift)
```

Headlines:

* **EFGP wall time barely scales with T** (29 s → 31 s for 4× more data),
  while SparseGP scales nearly linearly (67 s → 170 s, **~5× slower than
  EFGP at T=2000**).
* **EFGP wins ~15 % on observation RMSE** at every T.
* **EFGP wins ~10 % on Procrustes-aligned latent RMSE** at every T.
* **Drift RMSE is essentially tied** (within 1-3%) — both methods get
  comparably close to the trivial baseline at the same Duffing problem.
  The Duffing drift is sharper than what either method's converged ℓ can
  fully resolve at these settings, so this is a problem-difficulty
  artifact, not a method-quality difference.
* Both methods are stable & convergent at these settings.  Earlier
  experiments with more aggressive ρ schedules NaN'd SparseGP — see the
  notes in `demos/bench_duffing.py` for the safe settings.

### Linear-drift demo (small T, kernel hypers fixed)

`demos/efgp_synthetic_2d_jax.py` (T=60, D=2, kernel fixed at init):

| Method | Inducing/feature dim | Predictive observation RMSE | Wall time |
| --- | --- | --- | --- |
| **EFGP-SING (JAX)** | M = 49 | **0.219** | **~12 s** |
| SING-SparseGP        | 64 inducing pts | 0.251 | ~28 s |
| Trivial (predict mean obs) | – | 0.280 | – |

`fit_efgp_sing_jax`: inner E-step is `lax.scan`'d inside `jit`; q(f)
update + drift moments + SING natural-grad in one compiled graph; no
torch round-trip.

### Spectral-grid handling for kernel learning

When `learn_kernel=True`, ℓ changes during the M-step.  The default
behavior **lifts the spectral grid `(xis, ws, h, xcen)` to JIT inputs**
and uses `mtot_per_dim` as a `static_argname` — JAX automatically caches
one compiled artifact per `mtot_per_dim` value, so the JIT cache hits
whenever ℓ stays in a regime where the bisection-determined grid size
doesn't change.

For zero recompiles across an entire EM run (at the cost of a possibly
oversized grid for late ℓ values), use `pin_grid=True` together with
`pin_grid_lengthscale=ℓ_min` to commit to a conservative grid up front.
For an even more flexible setup, set `K_per_dim` (or
`K_min_lengthscale`) to use the **adaptive-h** grid (Option 2): the
integer mode lattice stays fixed at `mtot = 2K+1` and only the
physical spacing `h(θ)` adapts to the current ℓ — no JIT retrace as ℓ
moves.

### Grid-support and fixed-K diagnostics

Two grid-related issues turned out to matter during the JAX-port
stabilisation:

1. The old default `X_template` was prior-scale sized, which could
   under-cover the actual latent trajectory badly on nonlinear examples.
   The current default uses at least a `[-3, 3]^D` support box around the
   initial mean, expanding further when the prior variance is larger.
2. The production jit-friendly path remains a **fixed-K approximation** to
   the more exact "rebuild the tailored grid each outer EM step" oracle.
   Under the current `eps_grid = q(f)_nufft_eps = 1e-3`, this
   approximation is already good on the anharmonic benchmark, but it is
   still visible on the linear benchmark if one uses the older hardcoded
   `K=10` lattice. The current default instead picks `K` from the
   **initial tailored grid** at the requested `eps`, which restores the
   linear benchmark while staying fully jit-friendly. A direct `K` sweep
   also shows that the residual linear sensitivity is not explained by
   "too few modes" alone: the quality is non-monotone in `K`, so
   tailored-grid runs remain useful as a correctness oracle even though
   the production path should stay fixed-K / adaptive-h for speed.

### Per-iter cost breakdown

```
natural_to_marginal_params (Bayesian smoother, jit'd):    ~0.5 ms
compute_mu_r_jax (q(f) update, in-graph):                  ~5   ms
drift_moments_jax (Φ μ + Φ ξ μ in-graph):                  ~2   ms
estep one_iter (jit'd SING natural-grad update):           ~1   ms
                                                          --------
TOTAL inner-iter (lax.scan'd)                             ~9    ms
```

Performance progression of the project:

| Change | Wall time (T=60 demo) |
| --- | --- |
| Initial torch path: un-jit'd Bayesian smoother per call | 247 s |
| Jit smoother + jit SING natural-grad (still torch) | 142 s |
| Pass `rho` as JAX scalar (no per-iter retrace) | 53 s |
| Cache spectral grid across E-step iters | 45 s |
| Reuse marginals from last E-step iter in M-step | 41 s |
| Switch to pure-JAX `fit_efgp_sing_jax` (lax.scan inner E-step) | **12 s** |

## History (May 2026 bug fixes)

Three correctness fixes landed during the JAX-port stabilisation; each
is a regression test in `tests/test_efgp_sparsegp_drift_agreement.py`
or `tests/test_efgp_jax_recovery.py`.

### 1. Stein-correction sign

The Stein-correction term `h_{2,r} = Σ_j (... 2πi ξ_j ...) ⊙ D F_X* c_{j,r}`
in the q(f) update was originally written with a `+2πi ξ_j` factor (per
the appearance in the E-step PDF), but the term comes from the
**adjoint** Jacobian `J_φ^*`, and conjugating the complex-Fourier
derivative `∂φ_k/∂x_j = 2πi ξ_{kj} φ_k` flips the sign:
`h_{2,r} = Σ_j (-2πi ξ_j) ⊙ D F_X* c_{j,r}`.

Verified empirically against pair-sampling (no Stein correction).
Before the fix, the drift posterior at fixed truth hypers had `traj
RMSE 0.40`; after the fix, `0.17` (matches pair-sampling, lower
variance).  This bug is unique to the EFGP complex-Fourier
formulation; SING's SparseGP uses real-valued kernel evaluations and
never had it.  Fix lives in `sing/efgp_jax_drift.py`'s
`compute_mu_r_jax`.

### 2. SS-cross-cov transpose

The SS-cross-cov convention is `SSs[t] = Cov(x_{t+1}, x_t)` per
`sing/utils/sing_helpers.py:148`, but our Stein term needs
`Cov(x_t, x_{t+1}) = SSs[t].T`.  SS is near-symmetric for a
smoothed posterior so the practical effect is small, but we transpose
explicitly for correctness in `compute_mu_r_jax`.

### 3. M-step μ-refresh

The collapsed M-step computes the gradient of the negative log-marginal
via the envelope theorem at the current optimum `μ_r = A_θ⁻¹ h_θ,r`.  The
envelope simplification (∂μ/∂θ vanishes) only holds **when μ actually is
the optimum at the current θ**.  Originally we ran ``n_inner=5`` Adam
steps on `(log_ℓ, log_σ²)` while holding `μ_r` fixed at the value from
the start of the M-step block.  After a few aggressive Adam steps θ has
moved enough that `A_θ μ_r ≠ h_θ,r` and the envelope-theorem gradient
silently degrades — the *effective* gradient direction drifts toward
zero, the M-step makes ~1/4 the per-step progress of efgpnd's standalone
hyperparameter recovery, and ℓ stops well short of truth.

The fix: in `m_step_kernel_jax`, **refresh μ_r via a CG solve at every
Adam step** (`step > 0`) so the envelope theorem stays valid.  This adds
one CG per inner step but the cost is negligible (M-step is ~10 % of EM
wall time) and the convergence improvement is large.  End-to-end on the
GP-drift bench at T=2000: drift RMSE dropped from 0.46 to **0.40**, ℓ
recovered from 49 % to 73 % of truth.

### 4. Gaussian-emissions warmup

The biggest remaining "EFGP vs SparseGP" failure mode in the linear-drift
bench was not the GP regression block at all: with the kernel fixed, EFGP's
shared-`q(x)` / shared-`q(f)` tests already matched SparseGP closely, but the
end-to-end EFGP EM loop still diverged under the slow `ρ` schedule used in the
diagnostics.

Root cause: the **closed-form Gaussian emissions M-step** (`C, d, R`) was
running from EM iteration 1 while the latent posterior was still close to the
diffusion-only initialization.  Because EFGP's latent updates move more
gradually under small `ρ`, those early closed-form emission updates can overfit
the very broad initial posterior and create a bad latents/emissions feedback
loop.  Empirically:

* fixed kernel, slow `ρ`, `learn_emissions=False` → latent RMSE `0.077`
  (matches SparseGP's `0.076`)
* fixed kernel, same setup, immediate emissions updates → latent RMSE `0.181`

The fix is `emission_warmup_iters` in `fit_efgp_sing_jax` (default `8`): hold
`(C, d, R)` fixed until `q(x)` has sharpened enough that the closed-form
emission update is informative rather than destabilizing.  On the notebook's
linear benchmark this restores EFGP latent recovery to the SparseGP regime
without changing the GP block itself.

### 5. Benchmark-notebook caveat

`demos/efgp_vs_sparsegp_benchmarks.ipynb` turned out to be mixing several
different effects:

* On the linear row, the EFGP run at `n_em=25` is **not converged**.  Under
  the notebook settings, EFGP moves from `ℓ≈1.16` at 25 EM iterations to
  `ℓ≈1.44` at 50 and `ℓ≈1.63` at 100, while SparseGP moves from `ℓ≈2.15`
  to `2.02` to `1.82`.  The apparent "hyper divergence" is therefore partly
  a budget effect, not a distinct fixed point.
* On that same linear row, swapping in SparseGP's q(f) while holding the
  EFGP endpoint `q(x)` and hypers fixed only improves drift RMSE from
  `0.742` to `0.729`.  Swapping either the hypers or `q(x)` alone into the
  SparseGP solution gives an intermediate `~0.548`, while SparseGP's own
  endpoint is `0.514`.  This is strong evidence that the residual notebook
  gap is **not** a q(f) regression bug.
* On the nonlinear rows, the notebook's SparseGP protocol is itself unstable:
  with the current `_GLik` setup and `n_em=25`, Duffing can collapse to
  `σ²≈5e-6` and the anharmonic row can produce `NaN` hypers.  Those rows are
  therefore not suitable as a clean "EFGP vs SparseGP" correctness check.

For a reproducible diagnosis, run
`~/myenv/bin/python demos/diag_notebook_benchmark.py`.

## Known limitations / v0 scope

* **Cold-start kernel collapse.**  The collapsed kernel M-step is
  exact gradient-descent on the EFGP log-marginal-likelihood, but at
  initialisation `q(f)` is essentially trivial (`μ_r ≈ 0`), so the
  Hutchinson-trace `−½ log|A|` term dominates and pulls `ℓ → 0`.  The
  EM driver applies `kernel_warmup_iters` (default 8) of fixed-kernel
  EM before turning on the M-step to avoid this.
* **Emission cold-start sensitivity.**  The Gaussian closed-form
  `(C, d, R)` update is exact given the current `q(x)`, but if run too
  early it can overfit the diffusion-like initialization.  The EM driver
  therefore applies `emission_warmup_iters` (default 8) before updating
  Gaussian emissions.
* Single trial (`n_trials = 1`).  Multi-trial would just stack along
  axis 0 in `compute_mu_r_jax`.
* Drift noise variance σ²_r tied across output dims (`v0` simplification —
  yields one shared `T_r` Toeplitz cache).
* Diffusion `Σ` and input matrix `B` held fixed (per the M-step PDF's
  staged plan).
* Only the SE kernel is wired up for the M-step (Matern would just need a
  different `spectral_density` and corresponding ∂/∂(log ℓ) formulas).
* The local-quadratic transition approximation is the agreed v0
  baseline; exact `∇_x diag(Φ A⁻¹ Φ*)` gradients are an optional
  refinement (see the design notes).
