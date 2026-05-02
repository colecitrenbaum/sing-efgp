# EFGP-SING: scalable GP-drift inference for SING

This branch (`feat/efgp-drift`) adds an EFGP (Equispaced Fourier Gaussian
Process) drift block to SING.  It replaces the inducing-point sparse-GP
posterior in `SparseGP.update_dynamics_params` (Eq. 38, Appendix K.3 of the
SING paper) with a Toeplitz/NUFFT/CG implementation that scales as
**O(N + M log M)** instead of **O(N M²)** in the inducing-grid size.

The new code is purely **additive** — none of SING's existing modules are
modified, and `gp-quadrature/efgpnd.py` (the upstream PyTorch EFGP
implementation) is **not** modified either.  We import efgpnd's
public-ish primitives (NUFFT, ToeplitzND, CG, SLQ) and wrap them in a
backend protocol so a future JAX-native EFGP can drop in by writing a
single new `EFGPBackend` subclass.

## What was added

```
sing/
  efgp_backend.py     ── EFGPBackend protocol  +  TorchEFGPBackend
  efgp_drift.py       ── EFGPDrift(SDE)  +  _FrozenEFGPDrift  +  custom-VJPs
  efgp_emissions.py   ── closed-form (C, d, R) update for Gaussian emissions
  efgp_em.py          ── plain-Python variational-EM loop
demos/
  efgp_synthetic_2d.py    ── runnable end-to-end demo (matches a notebook)
tests/
  test_efgp_drift.py      ── 6 unit tests (matvec, CG solve, Hutchinson, M-step grad, …)
  test_efgp_vs_sparsegp.py── side-by-side check vs. SING-SparseGP
EFGP_SING_README.md  ── this file
```

## What was replaced

| SING/SparseGP | EFGP-SING |
| --- | --- |
| `SparseGP.update_dynamics_params` (Eq. 38, builds `(T-1, M, M)` and `(T-1, M)` kernel-expectation tensors) | `EFGPDrift.update_dynamics_params` — one type-1 NUFFT to build the BTTB Toeplitz `T_r = F_X* W_r F_X`, one CG solve per output dim |
| `SparseGP.f / ff / dfdx` (closed-form per-call expectations using the inducing kernel) | Single batched torch round-trip via `EFGPDrift.drift_moments_at_marginals`, exposed to SING's `jax.grad` through `_FrozenEFGPDrift` with custom VJPs encoding the local-quadratic transition approximation |
| Adam-on-ELBO kernel hyperparameter learning (backprop through the sparse-GP ELBO) | Direct collapsed M-step gradient: deterministic part via torch autograd through fixed μ_r, plus the Hutchinson trace `-½ tr[A⁻¹ ∂_θ A]` term to prevent lengthscale collapse |

## How the math maps to the code

| Math | Where it lives |
| --- | --- |
| Spectral grid, weights `D_θ = diag(√(S(ξ_k) · h^d))` | `TorchEFGPBackend.setup_grid` → `GridState.ws` |
| BTTB generator `T_r = F_X* W_r F_X` | `TorchEFGPBackend.make_toeplitz_weighted` → `ToeplitzOp.toeplitz` |
| `A_r v = v + D_θ T_r D_θ v` | `TorchEFGPBackend.make_A_apply` |
| Solve `A_r μ_r = h_r` | `TorchEFGPBackend.cg_solve` (efgpnd's `ConjugateGradients`) |
| `h_r = D_θ F_X* a_r + Σ_j (2πi ξ_j) ⊙ D_θ F_X* c_{j,r}` (Stein-corrected RHS, see E-step PDF §5.3) | `EFGPDrift.update_dynamics_params` build loop |
| `Φ_eval μ_r` (posterior mean drift) | `EFGPDrift._eval_mean_torch` (one type-2 NUFFT per output dim) |
| `Φ_eval [(2πi ξ_j) ⊙ μ_r]` (Jacobian) | `EFGPDrift._eval_jacobian_torch` |
| `diag(Φ A⁻¹ Φ*)` (Hutchinson estimator) | `TorchEFGPBackend.hutchinson_diag` |
| `log|A|` via Stochastic Lanczos Quadrature | `TorchEFGPBackend.logdet_A` (efgpnd's `logdet_slq`) |
| `∂E[f]/∂m ≈ J_f(m)` and `∂E[f^T Σ⁻¹ f]/∂(m, S)` (local-quadratic transition approximation) | `_ef_with_jac_grad`, `_eff_with_grads` `jax.custom_vjp` definitions in `efgp_drift.py` |
| Collapsed M-step gradient `Re((∂h)*μ) − ½ μ*(∂A)μ − ½ tr[A⁻¹ ∂A]` | `EFGPDrift.m_step_kernel` |
| Closed-form Gaussian `(C, d, R)` update | `efgp_emissions.update_emissions_gaussian` |

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

This is implemented in the `jax.custom_vjp` definitions in
`efgp_drift.py`.  Per design discussion, gradient through
`diag(Φ A⁻¹ Φ*)` (the posterior-variance landscape) is **not** included —
it would require autograd through a CG solve and add significant
machinery without changing the asymptotic behavior of the inference loop.

## Why a plain-Python EM loop?

SING's stock `fit_variational_em` is `@jit`-compiled and uses
`lax.scan` over the inner E-step.  Our EFGP block does numerics in
PyTorch (so we never touch `efgpnd.py`); we bridge JAX↔Torch by
`np.asarray` round-trips, which can't run inside `jit`.  The EM loop in
`efgp_em.py` is therefore plain Python.  The bottleneck is *per-iter*
torch round-trips, not per-time-step: the drift moments are computed in
**one** batched torch call per E-step inner iteration.

> **TODO(efgp-jax)**: when a stable JAX-native EFGP backend exists, the
> bridge disappears and the EM loop can be re-jitted by writing one new
> `EFGPBackend` class.  Search for `# TODO(efgp-jax)` markers in the
> code for the exact lines that change.

## Algorithm diagram

```
     ┌──────────────────────────────────────────────────────────────────┐
     │                       outer EM iteration                         │
     │                                                                  │
     │   ┌─── E-step (× n_estep_iters) ───────────────────────┐         │
     │   │                                                    │         │
     │   │  natural → marginal  (JAX, Bayesian smoother)      │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  EFGPDrift.update_dynamics_params  (TORCH:         │         │
     │   │     one type-1 NUFFT + CG per output dim)          │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  drift_moments_at_marginals  (TORCH: type-2        │         │
     │   │     NUFFTs for f̄, J_f̄ at every t in one round-     │         │
     │   │     trip; bridge back to JAX as 3 arrays)          │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  _FrozenEFGPDrift  (JAX SDE with custom-VJP        │         │
     │   │     gradients implementing the local-quadratic     │         │
     │   │     transition approximation)                      │         │
     │   │           │                                        │         │
     │   │           ▼                                        │         │
     │   │  SING nat-grad transition + likelihood updates     │         │
     │   │  (JAX, ρ-blended)                                  │         │
     │   └────────────────────────────────────────────────────┘         │
     │                                                                  │
     │   ┌─── M-step ─────────────────────────────────────────┐         │
     │   │  Closed-form (C, d, R)  (JAX)                      │         │
     │   │  EFGPDrift.m_step_kernel  (TORCH:                  │         │
     │   │     autograd of Re(h*μ) − ½ μ* A μ                 │         │
     │   │     + Hutchinson −½ tr[A⁻¹ ∂_θ A]                  │         │
     │   │     × n_inner Adam steps in log-hyper space)       │         │
     │   └────────────────────────────────────────────────────┘         │
     │                                                                  │
     └──────────────────────────────────────────────────────────────────┘
```

## How to run

Requires:

* `jax==0.6.1`, `jaxlib==0.6.1`, `tensorflow_probability==0.25.0`,
  `flax<0.12`, `tensorstore`, `optax`, `matplotlib`
* `torch>=2.6`, `pytorch_finufft`
* a sibling checkout of [`gp-quadrature`](../gp-quadrature) at
  `~/Documents/GPs/gp-quadrature` (so `efgpnd` is on the path)

Tests:

```bash
cd ~/Documents/GPs/sing
~/myenv/bin/python -m pytest tests/test_efgp_drift.py -v
```

End-to-end demo (also creates the diagnostic plots):

```bash
~/myenv/bin/python demos/efgp_synthetic_2d.py
# diagnostics in demos/_efgp_demo_out/
```

## Demo result (v0)

On the 2D linear-drift latent SDE in `demos/efgp_synthetic_2d.py`
(`T = 60`, `D = 2`, Gaussian observations, 8 EM iterations, kernel
hyperparameters held fixed at the init):

| Method | Inducing/feature dim | Predictive observation RMSE | Wall time |
| --- | --- | --- | --- |
| **EFGP-SING** (this branch) | M = 49 (Fourier features) | **0.228** | ~250 s |
| SING-SparseGP | 64 inducing pts | 0.251 | ~22 s |
| Trivial (predict mean obs) | – | 0.280 | – |

EFGP wins by ~10% on observation prediction.  EFGP wall time is dominated by
JAX compilation in the first iteration (~60 s overhead), then ~5–10 s per
inner E-step iter.  Subsequent runs (with JAX cache warm) are faster.

Both methods beat the trivial baseline; EFGP comes out ~10% better on
predictive observation RMSE in this setting.  EFGP wall time is dominated
by JAX compilation in the first iteration (~60 s overhead), then ~5–10 s
per E-step iter.

## Known limitations / v0 scope

* **Cold-start kernel collapse.**  The collapsed kernel M-step is
  exact gradient-descent on the EFGP log-marginal-likelihood, but at
  initialisation `q(f)` is essentially trivial (`μ_r ≈ 0`), so the
  Hutchinson-trace `−½ log|A|` term dominates and pulls `ℓ → 0`.  In the
  demo we keep `ℓ` fixed; the kernel-learning code is exercised by
  `test_collapsed_mstep_gradient_matches_finite_difference`.  A simple
  fix (warm-up iterations with no kernel update + reset on collapse) is
  a v1 task.
* Single trial (`n_trials = 1`).  Multi-trial would just stack along axis 0
  in `EFGPDrift.update_dynamics_params`.
* Drift noise variance σ²_r tied across output dims (`v0` simplification —
  yields one shared `T_r` Toeplitz cache).
* Diffusion `Σ` and input matrix `B` held fixed (per the M-step PDF's
  staged plan).
* Only the SE kernel is wired up for the M-step (Matern would just need a
  different `spectral_density` and corresponding ∂/∂(log ℓ) formulas).
* The local-quadratic transition approximation is the agreed v0
  baseline; exact `∇_x diag(Φ A⁻¹ Φ*)` gradients are an optional
  refinement (see the design notes).
