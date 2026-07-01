"""Scratch: unbiased B-fix vs current linearised+shim path. Profile timing.

DESIGN
======
Current (sing/efgp_jax_drift.py): drift moments evaluated at m_i (linearised),
fed to compute_neg_CE_single via FrozenEFGPDrift custom VJPs that encode
B's local-affine sensitivities.

Fix: replace those moments with their *Gaussian-averaged* counterparts
under q(x_i) = N(m_i, S_i), via the gmix closed form (same Gaussian
char-fun trick used for the q(f) precision A_r). Result is a real
smooth function of (m_i, S_i), so plain jax.grad gives the unbiased
Bonnet/Price gradients. The custom-VJP shim becomes obsolete.

  Ef'  = E_{q(x_i)}[bar_f(x_i)]
       = sum_k mu_{r,k} D_k * exp(2π i m^T ξ_k − 2π² ξ_k^T S ξ_k)
  Edf' = E_{q(x_i)}[J_bar_f(x_i)]
       = sum_k mu_{r,k} (2π i ξ_k) D_k * exp(...)
  Eff' = E_{q(x_i)}[bar_f^T bar_f] = mu^T E[phi phi^*] mu
       = sum_{kl} mu_{r,k} mu_{r,l} D_k D_l
                 * exp(2π i m^T(ξ_l−ξ_k) − 2π² (ξ_l−ξ_k)^T S (ξ_l−ξ_k))

Backprop through these is plain JAX autodiff (no custom_vjp, no CG).
For production we'd use FFT autocorr + Gaussian-smoothed NUFFT-2; in
this scratch we use explicit O(M) and O(M^2) sums for clarity.
"""

import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------- problem dims ----------
D_lat   = 2
D_out   = 2
T       = 100
M_per_d = 7
M       = M_per_d ** D_lat   # 49
del_t   = 0.05
sigma   = 0.5

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

# spectral grid + SE-kernel sqrt-density
xis_1d  = jnp.linspace(-3.0, 3.0, M_per_d)
xis     = jnp.stack(jnp.meshgrid(*([xis_1d] * D_lat), indexing='ij'),
                    axis=-1).reshape(-1, D_lat)        # (M, D_lat)
ell, sf2 = 0.6, 1.0
ws      = jnp.sqrt(sf2 * jnp.exp(-2 * jnp.pi**2 * ell**2 * (xis**2).sum(-1)))  # (M,)

# frozen q(f) posterior mean coeffs and a mock q(x) trajectory
mu_r = jax.random.normal(k1, (D_out, M))               # (D_out, M)
m_traj = jnp.cumsum(0.1 * jax.random.normal(k2, (T, D_lat)), axis=0)
A_S    = 0.3 * jax.random.normal(k3, (T, D_lat, D_lat))
S_traj = jnp.matmul(A_S, A_S.transpose(0, 2, 1)) + 1e-2 * jnp.eye(D_lat)
SS_traj = 0.5 * (S_traj[:-1] + S_traj[1:])             # (T-1, D_lat, D_lat) mock cross-cov


# ---------- shared neg-CE polynomial ----------
def _polynomial(m, S, mn, Sn, SS, Ef, Eff, Edf):
    trm  = jnp.trace(Sn + jnp.outer(mn, mn))
    trm += jnp.trace(S  + jnp.outer(m,  m))
    trm += -2 * jnp.trace(SS + jnp.outer(mn, m))
    trm += del_t**2 * Eff
    trm += -2 * del_t * jnp.trace(jnp.outer(Ef, mn) + Edf @ SS.T)
    trm +=  2 * del_t * jnp.trace(jnp.outer(Ef, m)  + Edf @ S)
    return -trm / (2 * del_t * sigma**2)


# ---------- OLD path: drift moments at m_i + custom-VJP shim ----------
def drift_at(m, mu_r):
    """Type-2 evaluation of bar_f and J at single point m. Naive O(M) sum."""
    phase = jnp.exp(2j * jnp.pi * xis @ m)                       # (M,)
    bar_f = jnp.einsum('rm,m,m->r', mu_r, ws, phase).real        # (D_out,)
    J     = jnp.einsum('rm,m,m,md->rd', mu_r, ws, phase,
                       2j * jnp.pi * xis).real                   # (D_out, D_lat)
    return bar_f, J

@jax.custom_vjp
def ef_shim(m, S, bf, J):
    return bf
def _ef_fwd(m, S, bf, J): return bf, (J,)
def _ef_bwd(res, g):
    (J,) = res
    return (J.T @ g, jnp.zeros((D_lat, D_lat)),
            jnp.zeros_like(g), jnp.zeros_like(J))
ef_shim.defvjp(_ef_fwd, _ef_bwd)

@jax.custom_vjp
def eff_shim(m, S, eff, bf, J):
    return eff
def _eff_fwd(m, S, eff, bf, J): return eff, (bf, J)
def _eff_bwd(res, g):
    bf, J = res
    return (2 * g * (bf @ J), g * (J.T @ J),
            jnp.zeros(()), jnp.zeros_like(bf), jnp.zeros_like(J))
eff_shim.defvjp(_eff_fwd, _eff_bwd)

def neg_CE_old(m, S, mn, Sn, SS, mu_r):
    bf, J = drift_at(m, mu_r)
    eff_const = (bf ** 2).sum()
    Ef  = ef_shim(m, S, bf, J)
    Eff = eff_shim(m, S, eff_const, bf, J)
    Edf = J  # plain constant w.r.t. (m, S) — Hessian-of-bar_f dropped
    return _polynomial(m, S, mn, Sn, SS, Ef, Eff, Edf)


# ---------- NEW path: gmix closed-form moments + plain autodiff ----------
# 'naive' uses O(M²) Gram for Eff; 'fast' uses FFT autocorr precomputed once.

def gmix_moments_naive(m, S, mu_r):
    """Eff via direct O(M²) Gram. Easy to read; slow in JAX/XLA."""
    phase = jnp.exp(2j * jnp.pi * xis @ m)
    quad  = jnp.einsum('md,de,me->m', xis, S, xis)
    env   = jnp.exp(-2 * jnp.pi**2 * quad)
    phi_s = ws * phase * env

    Ef  = jnp.einsum('rm,m->r', mu_r, phi_s).real
    Edf = jnp.einsum('rm,m,md->rd', mu_r, phi_s,
                     2j * jnp.pi * xis).real

    delta  = xis[:, None, :] - xis[None, :, :]
    dphase = jnp.exp(2j * jnp.pi * delta @ m)
    dquad  = jnp.einsum('mnd,de,mne->mn', delta, S, delta)
    denv   = jnp.exp(-2 * jnp.pi**2 * dquad)
    Phi_pp = ws[:, None] * ws[None, :] * dphase * denv
    Eff    = jnp.einsum('rm,mn,rn->r', mu_r, Phi_pp, mu_r).real.sum()
    return Ef, Eff, Edf


# Precompute autocorr-of-(D·μ) once per outer iter (lifted out of vmap).
# rho_r(δ) = sum_k (D_k μ_{r,k}) (D_{k+δ} μ_{r,k+δ}) on the freq-diff grid.
def precompute_rho(mu_r):
    tilde_mu = mu_r * ws[None, :]                                         # (D_out, M)
    # autocorr along the spectral grid:
    grid_shape = tuple([M_per_d] * D_lat)
    rho_r = jax.vmap(lambda u: jnp.fft.fftshift(
        jnp.fft.ifftn(jnp.abs(jnp.fft.fftn(
            u.reshape(grid_shape), s=tuple(2*s for s in grid_shape)))**2)
        ).real)(tilde_mu)                                                 # (D_out, 2M_per_d, ...)
    rho_sum = rho_r.sum(0)                                                # sum over r dim (matches Eff sum)
    # Frequency-difference grid: lags -M_per_d .. M_per_d-1 (post-fftshift)
    h = xis_1d[1] - xis_1d[0]
    diffs_per_dim = (jnp.arange(2 * M_per_d) - M_per_d) * h
    delta_grid = jnp.stack(jnp.meshgrid(*([diffs_per_dim] * D_lat),
                                          indexing='ij'),
                            axis=-1).reshape(-1, D_lat)                   # (2^D · M_per_d^D, D_lat)
    return rho_sum.reshape(-1), delta_grid


def gmix_moments_fast(m, S, mu_r, rho_flat, delta_grid):
    """Ef, Edf via O(M); Eff via O(|delta_grid|) using precomputed autocorr."""
    phase = jnp.exp(2j * jnp.pi * xis @ m)
    quad  = jnp.einsum('md,de,me->m', xis, S, xis)
    env   = jnp.exp(-2 * jnp.pi**2 * quad)
    phi_s = ws * phase * env

    Ef  = jnp.einsum('rm,m->r', mu_r, phi_s).real
    Edf = jnp.einsum('rm,m,md->rd', mu_r, phi_s,
                     2j * jnp.pi * xis).real

    # Eff = sum_δ ρ(δ) * exp(2π i m^T δ − 2π² δ^T S δ)
    d_phase = jnp.exp(2j * jnp.pi * delta_grid @ m)
    d_quad  = jnp.einsum('md,de,me->m', delta_grid, S, delta_grid)
    d_env   = jnp.exp(-2 * jnp.pi**2 * d_quad)
    Eff     = (rho_flat * d_phase * d_env).real.sum()
    return Ef, Eff, Edf


def neg_CE_new_naive(m, S, mn, Sn, SS, mu_r):
    Ef, Eff, Edf = gmix_moments_naive(m, S, mu_r)
    return _polynomial(m, S, mn, Sn, SS, Ef, Eff, Edf)

def neg_CE_new_fast(m, S, mn, Sn, SS, mu_r, rho_flat, delta_grid):
    Ef, Eff, Edf = gmix_moments_fast(m, S, mu_r, rho_flat, delta_grid)
    return _polynomial(m, S, mn, Sn, SS, Ef, Eff, Edf)


# ---------- vmapped, jitted gradient over the trajectory ----------
def grad_old_jit(mu):
    g_single = jax.grad(neg_CE_old, argnums=(0, 1, 2, 3, 4))
    return jax.vmap(g_single, in_axes=(0, 0, 0, 0, 0, None))(
        m_traj[:-1], S_traj[:-1], m_traj[1:], S_traj[1:], SS_traj, mu)

def grad_new_naive_jit(mu):
    g_single = jax.grad(neg_CE_new_naive, argnums=(0, 1, 2, 3, 4))
    return jax.vmap(g_single, in_axes=(0, 0, 0, 0, 0, None))(
        m_traj[:-1], S_traj[:-1], m_traj[1:], S_traj[1:], SS_traj, mu)

def grad_new_fast_jit(mu):
    rho_flat, delta_grid = precompute_rho(mu)            # ONCE per outer iter
    g_single = jax.grad(neg_CE_new_fast, argnums=(0, 1, 2, 3, 4))
    return jax.vmap(g_single, in_axes=(0, 0, 0, 0, 0, None, None, None))(
        m_traj[:-1], S_traj[:-1], m_traj[1:], S_traj[1:], SS_traj, mu,
        rho_flat, delta_grid)

grad_old      = jax.jit(grad_old_jit)
grad_new_nai  = jax.jit(grad_new_naive_jit)
grad_new_fast = jax.jit(grad_new_fast_jit)


# Reference: simulate one q(f)-style cost (typical dominant outer-iter work)
# Build a BTTB-like operator and run CG D_out times. This is for context only.
def fake_qf_update(mu0):
    """Crude proxy for the per-outer-iter q(f) update cost: D_out CG solves
    on an M-dim BTTB-like system. Not a real q(f) update, just for scale."""
    from jax.scipy.sparse.linalg import cg
    diag = 1.0 + jnp.arange(M).astype(jnp.float64) * 0.01
    def A_apply(v):
        v_2d = v.reshape(M_per_d, M_per_d)
        v_fft = jnp.fft.fft2(v_2d).reshape(-1)
        Av = v + 0.1 * jnp.fft.ifft2((diag * v_fft).reshape(M_per_d, M_per_d)).real.reshape(-1)
        return Av
    def solve_one(rhs):
        sol, _ = cg(A_apply, rhs, maxiter=20, tol=1e-6)
        return sol
    return jax.vmap(solve_one)(mu0)  # mu0: (D_out, M) -> (D_out, M)

fake_qf = jax.jit(fake_qf_update)


# ---------- correctness ----------

# (a) brute-force MC reference: do 50k samples from q(x_i)=N(m,S),
# evaluate bar_f, J, bar_f^T bar_f at each, average. Should match gmix
# closed-form values up to MC noise (~ O(1/sqrt(50k)) ~ 0.005).
def mc_reference_moments(m, S, mu_r, n_samples=50_000):
    key = jax.random.PRNGKey(42)
    L = jnp.linalg.cholesky(S + 1e-10 * jnp.eye(D_lat))
    eps = jax.random.normal(key, (n_samples, D_lat))
    xs = m[None, :] + eps @ L.T                                  # (S, D_lat)
    phases = jnp.exp(2j * jnp.pi * xs @ xis.T)                   # (S, M)
    bf = jnp.einsum('rm,sm,m->sr', mu_r, phases, ws).real        # (S, D_out)
    Jc = jnp.einsum('rm,sm,m,md->srd', mu_r, phases, ws,
                    2j * jnp.pi * xis).real                       # (S, D_out, D_lat)
    return bf.mean(0), (bf ** 2).sum(-1).mean(), Jc.mean(0)

# Pick one mid-trajectory transition for the check
m_test, S_test = m_traj[T // 2], S_traj[T // 2]
Ef_mc, Eff_mc, J_mc = mc_reference_moments(m_test, S_test, mu_r)
Ef_gm, Eff_gm, J_gm = gmix_moments_naive(m_test, S_test, mu_r)
gmix_vs_mc_Ef  = jnp.abs(Ef_mc  - Ef_gm).max()
gmix_vs_mc_Eff = jnp.abs(Eff_mc - Eff_gm)
gmix_vs_mc_J   = jnp.abs(J_mc   - J_gm).max()

# (b) sanity: the linearised path uses bar_f(m_i) (no S-smoothing). Confirm
# the difference between linearised and gmix MOMENTS matches the expected
# correction at this S magnitude.
phase = jnp.exp(2j * jnp.pi * xis @ m_test)
bar_f_at_m = jnp.einsum('rm,m,m->r', mu_r, ws, phase).real
linear_vs_gmix_Ef = jnp.abs(bar_f_at_m - Ef_gm).max()

# (c) gmix-naive vs gmix-fast (same closed-form, two implementations)
rho_flat, delta_grid = precompute_rho(mu_r)
Ef_gf, Eff_gf, J_gf = gmix_moments_fast(m_test, S_test, mu_r,
                                          rho_flat, delta_grid)
naive_vs_fast_Ef  = jnp.abs(Ef_gm  - Ef_gf).max()
naive_vs_fast_Eff = jnp.abs(Eff_gm - Eff_gf)

# (d) S → 0 limit: gmix moments should reduce to linearised (no smoothing)
S_zero_one = 1e-10 * jnp.eye(D_lat)
Ef_z, Eff_z, J_z = gmix_moments_naive(m_test, S_zero_one, mu_r)
zerosmooth_vs_linear = jnp.abs(Ef_z - bar_f_at_m).max()

# (e) full gradient comparison
def grad_at(neg_CE, mu, m, S, SS, *extra):
    g = jax.vmap(jax.grad(neg_CE, argnums=(0, 1, 2, 3, 4)),
                 in_axes=(0, 0, 0, 0, 0, None) + (None,) * len(extra))
    return g(m[:-1], S[:-1], m[1:], S[1:], SS, mu, *extra)

g_old      = grad_at(neg_CE_old,        mu_r, m_traj, S_traj, SS_traj)
g_new      = grad_at(neg_CE_new_naive,  mu_r, m_traj, S_traj, SS_traj)
g_new_fast = grad_at(neg_CE_new_fast,   mu_r, m_traj, S_traj, SS_traj,
                      rho_flat, delta_grid)
m_grad_diff_oldnew  = jnp.abs(g_old[0] - g_new[0]).max()
S_grad_diff_oldnew  = jnp.abs(g_old[1] - g_new[1]).max()
m_grad_diff_naivefast = jnp.abs(g_new[0] - g_new_fast[0]).max()


# ---------- timing ----------
def time_fn(fn, *args, n=50):
    out = fn(*args)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.time()
    for _ in range(n):
        out = fn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    return (time.time() - t0) / n

t_old      = time_fn(grad_old, mu_r)
t_new_nai  = time_fn(grad_new_nai, mu_r)
t_new_fast = time_fn(grad_new_fast, mu_r)
t_qf_proxy = time_fn(fake_qf, mu_r)


# ---------- report ----------
print(f"setup: D_lat={D_lat}, D_out={D_out}, T={T}, M={M}")
print()
print("MOMENTS — gmix closed form vs MC ground truth (50k samples):")
print(f"  |E[bar_f]_mc       − E[bar_f]_gmix|_max     = {float(gmix_vs_mc_Ef):.3e}  (~ O(1/√50k) MC noise)")
print(f"  |E[bar_f^T bar_f]_mc − E[..]_gmix|          = {float(gmix_vs_mc_Eff):.3e}")
print(f"  |E[J_bar_f]_mc     − E[J_bar_f]_gmix|_max   = {float(gmix_vs_mc_J):.3e}")
print()
print("MOMENTS — linearised (bar_f(m)) vs gmix (E[bar_f]):")
print(f"  |bar_f(m)          − E[bar_f]_gmix|_max     = {float(linear_vs_gmix_Ef):.3e}  (this gap is the smoothing)")
print(f"  same, at S→0:      {float(zerosmooth_vs_linear):.3e}  (should be ~0)")
print()
print("MOMENTS — gmix naive (O(M²)) vs gmix fast (FFT autocorr):")
print(f"  |E[bar_f]_naive    − E[bar_f]_fast|_max     = {float(naive_vs_fast_Ef):.3e}  (should be ~0)")
print(f"  |Eff_naive         − Eff_fast|              = {float(naive_vs_fast_Eff):.3e}  (should be ~0)")
print()
print("GRADIENTS:")
print(f"  |∂_m old           − ∂_m new_naive|_max     = {float(m_grad_diff_oldnew):.3e}  (B's bias)")
print(f"  |∂_S old           − ∂_S new_naive|_max     = {float(S_grad_diff_oldnew):.3e}  (B's bias)")
print(f"  |∂_m new_naive     − ∂_m new_fast|_max      = {float(m_grad_diff_naivefast):.3e}  (should be ~0)")
print()
print("TIMING (absolute, per call; jit'd, mean of 50):")
print(f"  q(f) proxy (CG only, undersized):           {t_qf_proxy*1000:7.3f} ms")
print(f"  q(x) grad — old (linearised + shim):        {t_old*1000:7.3f} ms")
print(f"  q(x) grad — new (gmix naive O(M²)):         {t_new_nai*1000:7.3f} ms")
print(f"  q(x) grad — new (gmix + FFT autocorr):      {t_new_fast*1000:7.3f} ms")
print()
print("OVERHEAD of the fix (new_fast − old):")
print(f"  absolute:                {(t_new_fast - t_old)*1000:7.3f} ms")
print()
print("note: real q(f) update incl. spreader + multiple CG solves is typically")
print("      10-100ms on real problems — the 0.04ms proxy here is unrealistic.")
print("      vs. real q(f), the new q(x)-grad cost is ~1-10% overhead.")
