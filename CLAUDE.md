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

## Project venv

```
/Users/colecitrenbaum/myenv/bin/python
```
The system Python is missing `jax_finufft` — do not use it.

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

## Test / regression cheat sheet

```bash
# Project venv
PY=/Users/colecitrenbaum/myenv/bin/python

# Full EFGP regression (multi-trial + K=1 + primitives + agreement)
$PY -m pytest tests/test_efgp_em_multitrial.py \
    tests/test_efgp_jax_primitives.py \
    tests/test_efgp_jax_recovery.py \
    tests/test_efgp_gmix_vs_mc.py \
    tests/test_efgp_sparsegp_drift_agreement.py \
    tests/test_gmix_nufft.py tests/test_gmix_spreader.py
```

`test_efgp_em_multitrial.py` covers: pytree round-trip, K-replication σ²/K
equivalence, ragged-vs-concat, K=1 smoke recovery, K-trial smoke fit.

## File map (EFGP-relevant)

- `sing/efgp_em.py` — top-level fit (multi-trial driver)
- `sing/efgp_jax_drift.py` — q(f) primitives (`compute_mu_r_*_jax`,
  `_flatten_stein`, `drift_moments_jax`, `m_step_kernel_jax`,
  `FrozenEFGPDrift`)
- `sing/efgp_jax_primitives.py` — JAX NUFFT, BTTB Toeplitz, spectral grid
- `sing/efgp_gmix_spreader.py` — closed-form Gaussian mixture spreader
- `sing/efgp_emissions.py` — closed-form Gaussian (C, d, R) update
  (multi-trial, aggregates over both K and T)
