# SING / EFGP project notes for Claude

## ⚠️ Git push policy — READ FIRST

This repo has TWO remotes:

```
origin    https://github.com/colecitrenbaum/sing-efgp.git   ← user's fork (push here)
upstream  https://github.com/lindermanlab/sing.git          ← lab repo (DO NOT push)
```

**Never push to `upstream` (lindermanlab/sing).** Only ever push to `origin`
(colecitrenbaum/sing-efgp). When committing, run `git push origin <branch>`
explicitly — never `git push --all`, never `git push upstream`, never an
unqualified `git push` if it could resolve to upstream.

If the user asks you to push, default to `git push origin main` (or whatever
the current branch is) and confirm if anything else is requested.



## Overview

Two GP-drift inference paths share the same SING natural-grad backbone:

- **EFGP-SING**: `sing/efgp_em.py::fit_efgp_sing_jax` — pure-JAX, multi-trial,
  closed-form gmix Gaussian-mixture spreader by default.
- **SparseGP-SING**: `sing/sing.py::fit_variational_em` — inducing-point GP,
  mature multi-trial path (predates EFGP).

Both consume the same `Likelihood`, `init_params`, `output_params` interfaces
and produce `marginal_params` of shape `(K, T, D)`.

## ⚠️ EFGP q(f) update default is DEVICE-DEPENDENT: `estep_method='auto'`

`fit_efgp_sing_jax` defaults to `estep_method='auto'`, which resolves to:

| device (`jax.default_backend()`) | resolved method |
|---|---|
| **`gpu`**            | **`gmix`** |
| **`cpu`** / `tpu`    | **`analytic`** (falls back to `gmix` if D≠2) |

**Use `analytic` on CPU, `gmix` on GPU.** The q(f) bottleneck is
device-specific, so the winner flips:

- **CPU:** the gmix per-source **spread-scatter** is the wall bottleneck.
  The **analytic** path replaces it with **type-1 NUFFTs + a Taylor-expanded
  Gaussian envelope** (write `S_i = S̄ + ΔS_i`; expand `exp(-2π²ξᵀΔS_iξ)` to
  `analytic_order`, default 1, each term one batched NUFFT with a
  ΔS-monomial-weighted source vector — see
  `sing/efgp_jax_drift.py::compute_mu_r_analytic_jax`). On CPU: **~6× faster**
  on the q(f) update (T=10K: 171→29 ms), **~1.66× faster end-to-end**
  (187→113 s), and *more accurate* — exact for homogeneous S vs gmix's
  n_sigma=1.5 stencil-tail-truncation bias (~10% vs MC even at het=0).

- **GPU:** scatter + cuFFT are ~free and the wall is dominated by **JIT
  compile + the sequential-in-T block-tridiag smoother `lax.scan`** (thousands
  of tiny sequential kernels — latency-bound, GPU's worst case; note CPU can
  even beat GPU for single-trial / modest-T / D=2). gmix's q(f) is cheap
  there, so it's *not* the lever. The analytic path only *adds* cuFINUFFT
  launch overhead (measured **slower** on GPU: 210 vs 157 s at T=10K) **and**
  regressed large-T accuracy under order 1 (nrmse 0.101 vs 0.052 at T=100K —
  the S-heterogeneity over a long trajectory outran first-order). So GPU stays
  on `gmix`. GPU speedups live in **compile** + the **scan** (parallel/
  associative-scan smoother, or the K-split: more, shorter trials → shallower
  scan → better occupancy; see the wall-time table below), *not* the q(f)
  method.

`analytic_order` (CPU): `1` (default) accurate to <4% up to ~20× S-spread
(converged fits ~1.4×); `2` up to ~60× (~10%); batched NUFFTs share `m_src`
so order→order cost is ~flat. Any explicit `estep_method` (`'gmix'`,
`'analytic'`, `'mc'`) overrides `'auto'`. Analytic is **2-D only in v0** and
its `mu_r` is a drop-in (same relative, xcen-aware frame as the MC path) for
`drift_moments_jax`, the gmix gather, and the kernel M-step.

## Project venv

```
/Users/colecitrenbaum/myenv/bin/python
```
The system Python is missing `jax_finufft` — do not use it.

## ⚠️ Use fp64 for any SING fit

JAX defaults to fp32. The SING block-tridiagonal smoother + SparseGP
M-step diverge into NaN at higher K·T in fp32 (empirically, K·T ≳ 5000 with
mstep_lr=0.01 is enough to blow up the SparseGP hyper Adam step).

**Always enable fp64 at the top of any benchmark script:**

```python
import jax
jax.config.update("jax_enable_x64", True)   # MUST be before any jax.* call
# ... rest of imports
```

In return, fp64 ~doubles EFGP wall and ~doubles SparseGP wall. Worth it
— without fp64, results at K·T ≳ a few thousand are unreliable. All the
canonical x64 demos (`bench_gpdrift_x64.py`, `bench_duffing_lsinit_x64.py`)
do this.

## Recommended fit settings (head-to-head benchmarks)

These come from the most-recent dialed-in EFGP-vs-SparseGP comparison in
`demos/bench_gpdrift_x64.py`. Use these unless you have a specific reason
to deviate:

| Setting | EFGP | SparseGP | notes |
|---|---|---|---|
| `n_em_iters` / `n_iters` | **50** | **50** | 20 isn't enough for ℓ-recovery to settle |
| `n_estep_iters` / `n_iters_e` | **10** | **10** | |
| `rho_sched` | `jnp.linspace(0.05, 0.7, n_em)` | same | conservative-to-aggressive ramp |
| `mstep_lr` / `learning_rate` | **0.01** | **0.01** | NOT 1e-3 (too small to move ℓ) |
| `n_mstep_iters` / `n_iters_m` | **4** | **4** | |
| `n_hutchinson_mstep` (EFGP) | **4** | n/a | |
| `kernel_warmup_iters` (EFGP) | **8** | n/a | smoother needs ≥8 outer EM iters before M-step works (collapse risk below ~5) |
| `eps_grid` (EFGP) | **1e-3** | n/a | tighter than 1e-2 |
| `S_marginal` (EFGP, MC mode) | **2** | n/a | gmix mode (default) doesn't need this |

### `output_scale` gotcha for SparseGP

`SparseGP`'s `RBF` kernel stores **`output_scale` as the SQRT of σ²**:
internally it squares to compute the kernel variance.

- ✅ Init: `output_scale=jnp.asarray(math.sqrt(VAR_INIT))`
- ❌ Don't pass `VAR_INIT` directly — that's σ² → variance σ⁴
- When reading back, square: `var_recovered = float(dp['output_scale']) ** 2`

EFGP's `variance` parameter is σ² directly (no sqrt), and EFGP `hist.variance`
records σ² directly.

### Inducing-point layout for SparseGP

Use a **data-aware bbox** rather than a fixed symmetric grid:

```python
def _data_aware_zs(num_per_dim, xs_np, pad=0.4):
    lo = xs_np.min(axis=(0, 1)) - pad   # (D,)
    hi = xs_np.max(axis=(0, 1)) + pad
    per_dim = [jnp.linspace(lo[d], hi[d], num_per_dim)
                for d in range(xs_np.shape[-1])]
    return jnp.stack(jnp.meshgrid(*per_dim, indexing='ij'),
                      axis=-1).reshape(-1, xs_np.shape[-1])
```

For 25 inducing in 2D: `_data_aware_zs(num_per_dim=5, xs_np=...)`.

Imports:

```python
from sing.expectation import GaussHermiteQuadrature
from sing.kernels import RBF
from sing.sde import SparseGP
from sing.initialization import initialize_zs   # for fixed grids
```

### Per-iter hyper history hook for SparseGP

`fit_variational_em` accepts `drift_params_history=[]` (a mutable list it
fills with per-iter snapshots of `drift_params`). Use this to plot ℓ, σ²
trajectories. Remember to square `output_scale` to get σ².

## Multi-trial EFGP

`fit_efgp_sing_jax` is multi-trial (since 2026-05). Inputs:

- `likelihood.ys_obs`: `(K, T, N)`
- `likelihood.t_mask`: `(K, T)` bool
- `init_params['mu0']`: `(K, D)`, `init_params['V0']`: `(K, D, D)`
- `output_params`: `C: (N, D)`, `d: (N,)`, `R: (N,)` — shared across trials

Outputs `marginal_params['m']: (K, T, D)`, etc.

**Each `K` value triggers a JIT recompile** (~10-15 s with gmix). Fix K per
experiment.

For ragged trials, pad to `T_max` and use `trial_mask`. Padded `S_src=0`
slots would NaN the gmix Gaussian (det(S)=0); `_flatten_stein` already
substitutes `eye(D)` at masked slots.

## Log-marginal landscapes (two variants)

### 1. GT (oracle) landscape — `gt_landscape`

Canonical implementations: `demos/bench_gpdrift_x64.py:288`,
`demos/bench_duffing_lsinit_x64.py:165`. Identical code in both.

Pseudo-velocity GP marginal NLL on the **true** latent paths:

```
v_t = (x_{t+1} - x_t) / dt                      # pseudo-velocity
v_t | x_t ~ N(f(x_t), (σ²/dt) I)                # transition lik
f ~ GP(0, σ²_f K_RBF(·; ℓ))
```

Negative log marginal likelihood:

```
NLL(ℓ, σ²_f) = ½ Σ_d v_dᵀ A⁻¹ v_d  +  ½ D log|A|
A = σ²_f K(x_in, x_in) + (σ²/dt) I
```

Implementation outline (per (ℓ, σ²_f)):

```python
import scipy.linalg
inputs = xs_np[:-1]                    # (n, D)
velocities = (xs_np[1:] - xs_np[:-1]) / dt
sq_dists = ((inputs[:, None] - inputs[None, :]) ** 2).sum(-1)
noise_var = sigma_drift ** 2 / dt
K = np.exp(-0.5 * sq_dists / ell ** 2)
A = var_f * K + noise_var * np.eye(n)
L = np.linalg.cholesky(A)
logdet = 2 * np.log(np.diag(L)).sum()
quad = sum(z @ z for z in [solve_triangular(L, velocities[:, d], lower=True)
                            for d in range(D)])
nll = 0.5 * quad + 0.5 * D * logdet
```

This is the **oracle objective** — what you'd minimize if you knew the latent
paths perfectly. Argmin gives the **oracle MLE**, which is the best a GP
inference method could possibly do.

Multi-trial: concatenate `(x_t, v_t)` pairs across trials into one big GP
regression. K=4, T=500 → ~2000 sources, fine for dense Cholesky.

Standard sweep ranges (from x64 benches):
```
LOG_LS_RANGE = (-2.0, 1.5)      # ℓ ∈ [0.135, 4.48]
LOG_VAR_RANGE = (-2.5, 2.0)     # σ²_f ∈ [0.082, 7.39]
N_GRID = 21
```

### 2. EFGP collapsed landscape — `_build_landscape_evaluator`

`demos/plot_mc_vs_gmix_landscape.py:46`. The actual objective `m_step_kernel_jax`
minimises, evaluated under the converged q(x):

```
L(θ) = -½ Σ_r Re⟨h_θ,r, A_θ⁻¹ h_θ,r⟩ + ½ D_out log|A_θ|
```

Use this when you want to see where EFGP's M-step actually thinks the optimum
is, given the learned q(x). Use the GT landscape when you want the oracle.

### Picking which to plot

- "Where does each method's hyper recovery sit relative to the oracle?"
  → **GT landscape**, mark truth + oracle MLE + EFGP-final + SparseGP-final.
- "Why is EFGP's M-step pulling toward this point?" → EFGP collapsed.
- Both can be sanity-checked: argmin(GT) and argmin(EFGP-collapsed at
  converged q(x)) should coincide if smoothing is good and EFGP is
  well-calibrated.

### Interpretation gotchas

- Truth (ℓ_true, σ²_true) is generally NOT at the oracle MLE — finite-data
  noise pulls the MLE off the data-generating value, sometimes
  significantly. Don't be alarmed if MLE is at e.g. ℓ=0.4 when ℓ_true=0.6.
- Method recovery is only meaningful relative to the MLE, not the truth.

## Wall-time scaling: K trials × T vs single trial × K·T

EFGP and SparseGP have **different bottlenecks**, so they react differently
when you split a fixed total budget K·T across more or fewer trials:

| Method | Dominant per-EM cost | Sequential-in-T? | K-split helps? |
|---|---|---|---|
| **SparseGP** | SING block-tridiag smoother (`natural_to_marginal_params` via `lax.scan`) | **YES** — depth = T_per_trial | **YES**: K=4×T=500 ≈ 4× faster than K=1×T=2000 (vmap reduces scan depth, total math unchanged) |
| **EFGP-gmix** | q(f) update: gmix spread + FFT + CG on K·(T−1) flat sources | NO — one composite op | **NO**: scales with K·T total, same wall regardless of split |

EFGP's natural-grad SING portion *does* benefit from K-axis vmap the same way,
but it's a small fraction of EFGP wall — the q(f) build dominates.

Practical implication: if you have a fixed sample budget and care about wall
time, **SparseGP wants more, shorter trials**; **EFGP is indifferent to the
K/T split** (modulo per-K JIT recompile). At the K=4 × T=500, K(T-1)≈2000 source
count, EFGP wall ≈ what it was on K=1 × T=2000, while SparseGP got faster.

## EFGP M-step internals

`m_step_kernel_jax` (`sing/efgp_jax_drift.py:500`) returns
`(log_ls_new, log_var_new, loss_history)` — the third value is the per-Adam-
step EFGP collapsed-NLL trajectory. `fit_efgp_sing_jax` currently discards it.
Add it to history if you need M-step convergence diagnostics.

## Production defaults — q(x) moments + V restoration

Two knobs are now defaulted to the gather-based production paths:

**`qx_moments_method='gmix_batched'` (DEFAULT).** Gaussian-averaged
`(Ef, Edfdx)` under q(x_t)=N(m,S) precomputed in ONE batched
gmix-gather call per inner iter (1 IFFT + per-source Gaussian stencil),
then injected via the legacy `FrozenEFGPDrift` custom_vjp shim
(linearised B′ gradient structure). Same gradients as the now-deleted
`'gmix_live'` path, ~2× faster. Smoothing controlled by
`qx_v_gather_n_sigma` (default 2.0 → ~5% rel err, absorbed by
ρ-damping).
- Alternative: `'linearised_shim'` (point-eval drift moments at m_t,
  no Gaussian smoothing). Cheaper (no gather) but misses the
  first-order S-correction in the base point.

**`restore_qf_variance='none'` (DEFAULT).** Drop the q(f)-variance
term `V(x) = Σ_r σ_r⁻² φ(x)* A_r⁻¹ φ(x)` (Approximation A). Justified
by the structural argument in `efgp_estep.tex` §6: for SDE-recovery
with diverse ICs and ergodic mixing, ∇V ≈ 0 at every interior `m_i`.
Diagnostic at K=10/T=500 confirms the gradient injection works
(verified numerically) but `‖∇V‖ ≈ 5e-5 · ‖legacy ∂Eff/∂m‖` per
transition — intrinsically tiny on this problem class, washed out by
SING's ρ-damping. The bench's identical recovery between `'none'` and
`'hutch'` reflects this, not a wiring bug.

**Opt-in for problems where the structural argument fails:**

- `'hutch'`: homogeneous-S restoration. Hutchinson estimate of the
  diagonal sums `ω(δ) = Σ_(k,l): ξ_l−ξ_k=δ (DA⁻¹D)_(kl)` (padded-FFT
  cross-correlation) + a single batched Type-2 NUFFT against
  `ω·env(S_homo)`. Per-source S_i replaced by trajectory-mean. ~11%
  wall overhead at K=10/T=500.
- `'hutch_hetS'`: per-source S_i envelope via the gmix-gather primitive
  (`sing/efgp_gmix_gather.py`). ~15% wall overhead.

Use either opt-in when the structural argument is suspect: **anisotropic
source coverage**, sources clustered in a sub-region, **sharp turns
near the data-domain boundary**, **cold-start before q(x) localises**,
or **substantially varying S_i across t** (early/late + steady state).
- `'none'`: drop V (Approximation A baseline). Empirically fine on
  uniformly-distributed data with ergodic SDEs, but `'hutch'` is now
  nearly free so it is preferred.

For all three V modes, `∂V/∂S_i` is set to 0 — first-order Taylor
linearisation of V at m_i, exactly analogous to the linearised B′
approximation already used for f. Bias is `O(‖S_i‖·σ_f²/ℓ²)`, small
for q(x) localised at scale ℓ. See `efgp_estep.tex` §6 for the
explicit Taylor argument.

V-restoration is only wired for `qx_moments_method` in
(`'linearised_shim'`, `'gmix_batched'`) + `estep_method='gmix'`.

## Test / regression cheat sheet

```bash
# Project venv
PY=/Users/colecitrenbaum/myenv/bin/python

# Canonical full EFGP regression (multi-trial + K=1 + primitives + agreement).
# DO NOT include the x64-forcing test files below in this sequence —
# they globally enable jax_enable_x64 which exposes a latent fp32-only
# assumption in test_efgp_jax_primitives.py::test_cg_solve_matches_torch.
$PY -m pytest tests/test_efgp_em_multitrial.py \
    tests/test_efgp_jax_primitives.py \
    tests/test_efgp_jax_recovery.py \
    tests/test_efgp_gmix_vs_mc.py \
    tests/test_efgp_analytic.py \
    tests/test_efgp_sparsegp_drift_agreement.py \
    tests/test_gmix_nufft.py tests/test_gmix_spreader.py

# V-restoration tests (force x64; run separately):
$PY -m pytest tests/test_gmix_gather.py tests/test_efgp_qx_v_hutch.py
```

`test_efgp_em_multitrial.py` covers: pytree round-trip, K-replication σ²/K
equivalence, ragged-vs-concat, K=1 smoke recovery, K-trial smoke fit.
`test_gmix_gather.py` covers the Type-2 gmix-gather primitive (4 tests:
DC, random-c vs explicit sum, eps convergence, autodiff grad).
`test_efgp_qx_v_hutch.py` covers V-Hutch homog + hetS (unbiasedness vs
dense Cholesky, hetS-vs-homog at constant S, hetS-vs-dense-truth with
varying S, complex-ω guard, slow E2E E2E with all 3 modes).

## File map (EFGP-relevant)

- `sing/efgp_em.py` — top-level fit (multi-trial driver)
- `sing/efgp_jax_drift.py` — q(f) primitives (`compute_mu_r_*_jax` incl.
  `compute_mu_r_analytic_jax` (DEFAULT path) + `qf_and_moments_analytic_jax`
  + `_env_taylor_pairs`, `_flatten_stein`, `drift_moments_jax`,
  `m_step_kernel_jax`, `FrozenEFGPDrift`)
- `sing/efgp_jax_primitives.py` — JAX NUFFT, BTTB Toeplitz, spectral grid
- `sing/efgp_gmix_spreader.py` — closed-form Gaussian mixture spreader (Type-1)
- `sing/efgp_gmix_gather.py` — Type-2 analog of spreader (per-source
  Gaussian gather of an inverse-FFT'd spectral coefficient grid). Used
  by `restore_qf_variance='hutch_hetS'`.
- `sing/efgp_qx_v_hutch.py` — V-Hutch restoration (homog + hetS):
  `precompute_omega_per_r` (Hutchinson ω̂ via FFT xcorr),
  `precompute_V_and_grad_E_V_per_t{,_hetS}`, `FrozenEFGPDriftWithVHutch`
  (custom_vjp shim).
- `sing/efgp_emissions.py` — closed-form Gaussian (C, d, R) update
  (multi-trial, aggregates over both K and T)
