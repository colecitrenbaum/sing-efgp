# Keep-all autodiff E-step (experimental branch `exp-keep-all-terms`)

## TL;DR
An alternative EFGP-SING **q(x) update** that keeps *all* transition-term
gradients (no Gauss--Newton / statistical-linearisation drops) by simply
**autodiffing the summed transition ELBO** built from batched gmix moments.
It is exact, comparable-error to production, and (on CPU) faster at T=10k.
Use it via `qx_moments_method='gmix_full_batched'` in `fit_efgp_sing_jax`.

## Why this exists
Production drops the residual-Hessian / covariance terms of
`E_q[ f^T f ]`'s (m,S)-derivatives (the "linearisation"), injecting them
through the `FrozenEFGPDrift` custom_vjp shim. Those dropped terms are
individually large & sign-indefinite, but get swamped by the O(1/Δt)
δ-quadratic in the assembled precision, so production works. This branch
instead keeps everything and lets autodiff produce the gradients.

## The idea (why autodiff is cheap here)
Every q(x)-moment is a linear contraction of the single gmix'd feature
vector  ψ_k(m,S) = D_k · exp(2πi ξ_k·(m-x_c)) · exp(-2π² ξ_k^T S ξ_k):
  E[f]      = μ^T ψ ,   E[J] = μ^T (2πi ξ ⊙ ψ) ,   E[f^T f] = Σ_δ ρ(δ)·env(δ).
All (m,S)-derivatives are just more frequency-weightings of ψ
(∂m → 2πi ξ, ∂S → -2π² ξξ^T). Reverse-mode autodiff regenerates them
automatically: **the VJP of a NUFFT is a NUFFT** (Type-2 ↔ Type-1), and
that adjoint *is* the frequency multiplication. So we build the summed
transition term with batched moments and take ONE `jax.grad` over all
(m_i,S_i); the sum decouples so we still get every per-transition natural
gradient — no shim, no drops, no custom_vjp.

## Files
- `sing/exp_full_moments.py` — differentiable exact closed-form moments:
  `Ef_diff`, `Edfdx_diff` (direct spectral sums), and `FullGmixDrift`
  (a plain SDE shim: f/ff/dfdx differentiable, no custom_vjp — the
  per-source autodiff reference). Frame: phase at (m - grid.xcen).
- `sing/exp_batched_estep.py` — the production-shaped path:
  `make_total_negCE` (summed transition ELBO, batched exact moments),
  `nat_grad_batched` (one `jax.grad` → {J,h,L}, monolithic),
  `nat_grad_batched_chunked` (source-chunked, O(chunk·M) memory, K=1).
- `sing/efgp_em.py` — new `qx_moments_method` branches:
  `'gmix_full'` (per-source autodiff via FullGmixDrift; exact ref, slow),
  `'gmix_full_batched'` (monolithic batched grad-once; **use this**).
  (Env vars `KEEPALL_PSD`, `KEEPALL_FF` are experiment toggles, default off.)
- `demos/bench_keepall_autodiff.py` — runnable head-to-head vs production.

## How to run
```python
fit_efgp_sing_jax(..., estep_method='gmix',
                  qx_moments_method='gmix_full_batched')   # keep-all autodiff
# vs production:
fit_efgp_sing_jax(..., qx_moments_method='gmix_batched')   # drop / shim
```
Always enable fp64 first: `jax.config.update("jax_enable_x64", True)`.
Quick bench: `python demos/bench_keepall_autodiff.py`.

## CPU validation (already done, darwin, fp64)
- Correctness: batched grad-once == per-source exact reference to ~1e-16
  (J,h) and exactly (L).
- T-scaling: linear in T (3.6ms→106ms over T=100→1600, M=121), comparable
  to per-source.
- T=10k Duffing per-E-step call: 150 ms (keep-all) vs 456 ms (production).
- T=10k Duffing end-to-end (15 EM iters, learn_kernel): latent RMSE
  0.0566 (keep-all) vs 0.0576 (production); ell/var identical; 33s vs 36s.
- Memory (monolithic, M=121): peak RSS 0.77 / 1.65 / 4.75 GB at
  N=1e4 / 3e4 / 1e5. Grows O(N·M). Chunked stays flat ~1 GB (exact same
  gradient) if needed.

## What to look for on GPU
This path is dense-einsum + single-grad, which *should* be GPU-friendly,
but verify:
1. **Recovery error parity.** Run the bench (or your problem) with
   `gmix_full_batched` vs `gmix_batched`; latent RMSE and recovered
   (ell, sigma_f^2) should match to ~3 decimals. This is the main check.
2. **Wall clock.** Expect keep-all to avoid (a) cuFINUFFT gather launch
   overhead and (b) the per-transition `vmap(nat_grad_transition)` +
   custom_vjp of production. But note the **shared** cost — the
   block-tridiag smoother `lax.scan` (natural↔marginal, sequential in T)
   — tends to dominate GPU wall for BOTH methods, so totals may be close.
   Time the E-step natgrad call in isolation to see the real difference.
3. **Peak memory.** Monolithic peak is O(N·M) (reverse-mode tape holds
   ~a dozen (N,M)-ish arrays). At M≈121 that's ~4.75 GB at N=1e5 (fine).
   At large M (≳10^3) it grows ~10× and may OOM — if so switch to
   `nat_grad_batched_chunked` (flat ~1 GB, exact) or lower the grid M.
4. **Direct-sum vs NUFFT crossover.** Moments here are O(N·M) direct
   spectral sums (GPU-happy dense einsums, no cuFINUFFT). At very large M
   the production gather's O(M log M + N) wins on FLOPs; on GPU the
   crossover is pushed higher (dense FLOPs are cheap; cuFINUFFT has launch
   overhead). Worth a quick M-sweep if you push M large.
5. **fp64.** Required (see repo CLAUDE.md); fp32 diverges at K·T ≳ few k.

## Known limitations (before using beyond the benchmark)
- **Inputs are ignored.** `make_total_negCE` / `nat_grad_batched` omit the
  linear-input (B u) terms of `compute_neg_CE_single`. Correct only when
  there are no inputs (input_effect = 0). Add them before using with inputs.
- **V term dropped** (drift posterior variance), same as production
  `restore_qf_variance='none'` (Approximation A). Not restored here.
- `nat_grad_batched_chunked` assumes K==1; monolithic handles K>1.
- These are experimental modules (`exp_*`); not integrated into tests.

## Design threads (for the paper / next steps)
- **This (autodiff) thread:** simplest to explain — "differentiate the
  exact transition ELBO; all gradients are NUFFT adjoints; nothing
  dropped." Validated above.
- **Shim thread (alt):** a custom_vjp that *injects* the exact gradients
  as batched gather NUFFTs (O(M log M + N), lowest memory) — more code,
  best asymptotics at large M. Not implemented on this branch.
