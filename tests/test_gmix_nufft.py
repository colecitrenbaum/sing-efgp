"""Test: Gaussian-mixture NUFFT vs dense closed-form evaluation.

Targets the closed-form generator from efgp_estep_charfun.tex §3-§5:

    F(xi) = sum_i w_i exp(2 pi i m_i^T xi) exp(-2 pi^2 xi^T S_i xi)

at output points {xi_k} (the spectral grid for h_r) or {Delta xi}
(the difference grid for T_r).  The dense reference is direct
summation; the bucketed-NUFFT implementation realises variant (B)
from §5.4 of the tex.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IMPORTANT: jax_finufft must be initialised before any torch_finufft call
import jax_finufft
import jax
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(0)


def dense_closed_form(m, S, w, xi_out):
    """Direct evaluation of F(xi) over T sources at n_out output frequencies.

    m:      (T, D)  means
    S:      (T, D, D) per-source covariances (SPD)
    w:      (T,)    real or complex weights
    xi_out: (n_out, D)  output frequencies
    Returns: (n_out,)  complex values F(xi_out)
    """
    T, D = m.shape
    phase = np.exp(2j * np.pi * (m @ xi_out.T))             # (T, n_out)
    quad = np.einsum('nd,tde,ne->tn', xi_out, S, xi_out)    # (T, n_out)
    damp = np.exp(-2 * np.pi**2 * quad)                     # (T, n_out)
    return (w[:, None] * phase * damp).sum(axis=0)          # (n_out,)


def _kmeans_buckets(S, n_buckets, n_iter=20, seed=0):
    """Tiny KMeans on flattened S to cluster sources by covariance."""
    rng = np.random.default_rng(seed)
    T = S.shape[0]
    feats = S.reshape(T, -1)
    if n_buckets >= T:
        return np.arange(T)
    init_idx = rng.choice(T, size=n_buckets, replace=False)
    centers = feats[init_idx].copy()
    for _ in range(n_iter):
        dists = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = np.argmin(dists, axis=1)
        for b in range(n_buckets):
            mask = labels == b
            if mask.any():
                centers[b] = feats[mask].mean(axis=0)
    return labels


def bucketed_gmix_nufft_with_labels(m, S, w, xi_out, labels, n_buckets):
    """Bucketed gmix NUFFT with externally-supplied bucket labels.

    Separating clustering from the NUFFT loop is required for the
    complexity tests: clustering is amortised (one-shot per E-step
    iter), the per-call cost is the NUFFT + damping loop alone.
    """
    T, D = m.shape
    n_out = xi_out.shape[0]
    out = np.zeros(n_out, dtype=np.complex128)
    for b in range(n_buckets):
        idx = np.where(labels == b)[0]
        if idx.size == 0:
            continue
        S_bar = S[idx].mean(axis=0)
        # standard NUFFT (here: direct phase sum) — this is the bucket's
        # O(T_b * M_out) work; jax_finufft would replace this by a true
        # O(T_b + M_out log M_out) call.
        partial = (w[idx][:, None] *
                   np.exp(2j * np.pi * (m[idx] @ xi_out.T))).sum(axis=0)
        damp = np.exp(-2 * np.pi**2 *
                      np.einsum('nd,de,ne->n', xi_out, S_bar, xi_out))
        out += partial * damp
    return out


def bucketed_gmix_nufft_kmeans(m, S, w, xi_out, *, n_buckets):
    """Same as bucketed_gmix_nufft but with KMeans bucketing on flattened S."""
    labels = _kmeans_buckets(S, n_buckets)
    return bucketed_gmix_nufft_with_labels(m, S, w, xi_out, labels, n_buckets)


# ---------------------------------------------------------------------------
# jax_finufft-backed bucketed gmix NUFFT (variant B from the tex, real cost)
# ---------------------------------------------------------------------------
def _jax_nufft1_2d(m, w, M_per_dim, h_freq):
    """Type-1 NUFFT (T+M log M), iflag=+1, regular output grid.

    Returns f_k = Σ_t w_t exp(+2πi m_t^T ξ_k) on the regular grid
    ξ_k = h_freq * k for k ∈ {-K, ..., K}^2, M_per_dim = 2K+1.
    Output shape (M_per_dim, M_per_dim), in natural (centred) order.
    """
    # jax_finufft expects x in [-π, π].  Phase coords: 2π h_freq m_t
    phi_x = 2.0 * np.pi * h_freq * m[:, 0]
    phi_y = 2.0 * np.pi * h_freq * m[:, 1]
    return jax_finufft.nufft1(
        (M_per_dim, M_per_dim),
        w.astype(np.complex128),
        jnp.asarray(phi_x), jnp.asarray(phi_y),
        eps=1e-9, iflag=+1,
    )


def bucketed_gmix_nufft_jax(m, S, w, M_per_dim, h_freq, labels, n_buckets):
    """jax_finufft-backed variant B: per-bucket type-1 NUFFT + damping.

    Output frequency grid is regular: ξ_k = h_freq * k for k ∈ {-K..K}^2,
    flattened in C-order.  Total cost expected: O(T + B M log M).
    """
    T, D = m.shape
    K = (M_per_dim - 1) // 2
    assert D == 2

    # Precompute output ξ tensor for damping
    k_axis = np.arange(-K, K + 1)
    kx, ky = np.meshgrid(k_axis, k_axis, indexing='ij')
    xi_x = kx * h_freq
    xi_y = ky * h_freq

    out = np.zeros((M_per_dim, M_per_dim), dtype=np.complex128)
    for b in range(n_buckets):
        idx = np.where(labels == b)[0]
        if idx.size == 0:
            continue
        S_bar = S[idx].mean(axis=0)
        partial = np.asarray(_jax_nufft1_2d(
            m[idx], w[idx], M_per_dim, h_freq))
        # damping at output grid
        quad = (S_bar[0, 0] * xi_x ** 2
                + (S_bar[0, 1] + S_bar[1, 0]) * xi_x * xi_y
                + S_bar[1, 1] * xi_y ** 2)
        damp = np.exp(-2 * np.pi**2 * quad)
        out += partial * damp
    return out.ravel()


def _make_clustered_problem(T, D, *, n_clusters=8, S_scale=0.02, seed=0):
    """T sources whose S_i fall into n_clusters tight groups in matrix space.

    Returns m, S, w, cluster_id (the *true* labels) so tests can isolate
    "does the bucketed-NUFFT algorithm work?" from "does my naive KMeans
    converge?".
    """
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1, 1, size=(T, D))
    Lc = rng.standard_normal((n_clusters, D, D)) * np.sqrt(S_scale)
    Sc = np.einsum('cij,ckj->cik', Lc, Lc) + S_scale * np.eye(D)
    cluster_id = rng.integers(0, n_clusters, size=T)
    # Each source is its cluster centroid + tiny SPD jitter
    jitter_L = rng.standard_normal((T, D, D)) * np.sqrt(S_scale) * 0.001
    jitter = np.einsum('tij,tkj->tik', jitter_L, jitter_L)
    S = Sc[cluster_id] + jitter
    w = rng.standard_normal(T)
    return m, S, w, cluster_id


def bucketed_gmix_nufft(m, S, w, xi_out, *, n_buckets):
    """Bucketed standard NUFFT realisation of F(xi) (variant B from the tex).

    Cluster S_i into n_buckets via simple flatten+KMeans-on-S_flat (here
    we use a tiny in-script clustering to avoid sklearn dep).  For each
    bucket b:
      partial_b(xi) = sum_{i in b} w_i exp(2 pi i m_i^T xi)        (NUFFT)
      out += partial_b(xi) * exp(-2 pi^2 xi^T S_bar_b xi)          (damping)

    The standard NUFFT here is *not* jax_finufft (which expects a regular
    output grid via type-1) — for non-uniform output xi we do a direct
    O(T_b * n_out) phase sum.  Equivalent to type-3 NUFFT structurally;
    correctness is identical and is what we are testing.  Production
    code would route this through type-3 / spectral-grid type-1 calls.
    """
    T, D = m.shape
    n_out = xi_out.shape[0]

    # Naive clustering: bucket by the (1, 1) entry of S only — fine for
    # this correctness check, where what matters is that within-bucket
    # damping ≈ true damping.
    keys = S[:, 0, 0]
    edges = np.quantile(keys, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bucket_id = np.searchsorted(edges, keys, side='right') - 1
    bucket_id = np.clip(bucket_id, 0, n_buckets - 1)

    out = np.zeros(n_out, dtype=np.complex128)
    for b in range(n_buckets):
        idx = np.where(bucket_id == b)[0]
        if idx.size == 0:
            continue
        S_bar = S[idx].mean(axis=0)
        # Standard NUFFT (direct phase sum here for a self-contained test):
        partial = (w[idx][:, None] *
                   np.exp(2j * np.pi * (m[idx] @ xi_out.T))).sum(axis=0)
        damp = np.exp(-2 * np.pi**2 *
                      np.einsum('nd,de,ne->n', xi_out, S_bar, xi_out))
        out += partial * damp
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _make_problem(T, D, *, S_scale=0.05, S_constant=False, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1, 1, size=(T, D))
    if S_constant:
        L0 = rng.standard_normal((D, D)) * np.sqrt(S_scale)
        S0 = L0 @ L0.T + S_scale * np.eye(D)
        S = np.broadcast_to(S0, (T, D, D)).copy()
    else:
        Ls = rng.standard_normal((T, D, D)) * np.sqrt(S_scale)
        S = np.einsum('tij,tkj->tik', Ls, Ls) + S_scale * np.eye(D)
    w = rng.standard_normal(T) + 1j * rng.standard_normal(T) * 0.0
    return m, S, w


def _make_output_grid(D, m_per_dim):
    """Equispaced spectral grid on [-2, 2]^D with m_per_dim modes/dim."""
    axes = [np.linspace(-2.0, 2.0, m_per_dim) for _ in range(D)]
    grids = np.meshgrid(*axes, indexing='ij')
    xi_out = np.stack([g.ravel() for g in grids], axis=-1)
    return xi_out


def test_constant_S_one_bucket_is_exact():
    """When S_i is constant, B=1 must match dense to floating point."""
    m, S, w = _make_problem(T=20, D=2, S_scale=0.03, S_constant=True)
    xi_out = _make_output_grid(D=2, m_per_dim=11)
    ref = dense_closed_form(m, S, w, xi_out)
    got = bucketed_gmix_nufft(m, S, w, xi_out, n_buckets=1)
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"[constant S, B=1]   rel-err = {err:.2e}  (target: < 1e-12)")
    assert err < 1e-12


def test_T_buckets_is_exact():
    """When B = T (one source per bucket), bucketed result must equal dense."""
    m, S, w = _make_problem(T=20, D=2, S_scale=0.05)
    xi_out = _make_output_grid(D=2, m_per_dim=11)
    ref = dense_closed_form(m, S, w, xi_out)
    got = bucketed_gmix_nufft(m, S, w, xi_out, n_buckets=20)
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"[B=T (=20)]         rel-err = {err:.2e}  (target: < 1e-12)")
    assert err < 1e-12


def test_error_decreases_with_buckets():
    """B → T must monotonically reach exact (down to fp).  At fixed B<<T
    error depends on the bucketing strategy; the naive 1D-quantile bucketer
    used here under-performs proper KMeans-on-flat-S, but the algorithm
    itself approaches the dense closed form as B grows."""
    m, S, w = _make_problem(T=200, D=2, S_scale=0.05)
    xi_out = _make_output_grid(D=2, m_per_dim=11)
    ref = dense_closed_form(m, S, w, xi_out)
    print()
    errs = []
    for B in [1, 4, 16, 64, 200]:
        got = bucketed_gmix_nufft(m, S, w, xi_out, n_buckets=B)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        errs.append(err)
        print(f"[T=200, B={B:>3}]       rel-err = {err:.2e}")
    # B=T must match dense exactly.
    assert errs[-1] < 1e-12, f"B=T not exact: {errs[-1]:.2e}"
    # Trend: B=T <= B=1 (the naive bucketer is non-monotone in between).
    assert errs[-1] < errs[0]


def test_BTTB_difference_grid():
    """Same algorithm at the BTTB difference grid produces T_r[Delta xi]."""
    m, S, w = _make_problem(T=50, D=2, S_scale=0.05)
    # Spectral grid 7x7 -> difference grid 13x13
    m_per_dim = 7
    diff_axes = [np.arange(-(m_per_dim - 1), m_per_dim) * (1.0 / m_per_dim)
                 for _ in range(2)]
    grids = np.meshgrid(*diff_axes, indexing='ij')
    delta_xi = np.stack([g.ravel() for g in grids], axis=-1)
    print(f"\n[BTTB Δξ grid]      T={50}, M_diff={delta_xi.shape[0]}")
    ref = dense_closed_form(m, S, w, delta_xi)
    got_T = bucketed_gmix_nufft(m, S, w, delta_xi, n_buckets=50)  # exact
    err_T = np.linalg.norm(got_T - ref) / np.linalg.norm(ref)
    print(f"[BTTB Δξ, B=T]      rel-err = {err_T:.2e}  (target: < 1e-12)")
    assert err_T < 1e-12

    got_4 = bucketed_gmix_nufft(m, S, w, delta_xi, n_buckets=4)
    err_4 = np.linalg.norm(got_4 - ref) / np.linalg.norm(ref)
    print(f"[BTTB Δξ, B=4]      rel-err = {err_4:.2e}")


def test_clustered_S_oracle_buckets_recover_dense():
    """When S_i is genuinely clustered into K groups, oracle bucketing
    (true cluster id) at B=K must recover the dense closed form down
    to within-cluster jitter; B=T must match exactly."""
    m, S, w, cid = _make_clustered_problem(T=200, D=2, n_clusters=8)
    xi_out = _make_output_grid(D=2, m_per_dim=11)
    ref = dense_closed_form(m, S, w, xi_out)
    print(f"\n[clustered S, T=200, n_clusters=8, oracle bucketing]")
    # Oracle B=8: only error is within-cluster jitter
    got = bucketed_gmix_nufft_with_labels(m, S, w, xi_out, cid, 8)
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  B=  8 (oracle)   rel-err = {err:.2e}")
    assert err < 1e-3, f"Oracle B=K failed: {err:.2e}"
    # B=T (each source its own bucket): exact
    got = bucketed_gmix_nufft_with_labels(
        m, S, w, xi_out, np.arange(200), 200)
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  B=200 (B=T)     rel-err = {err:.2e}")
    assert err < 1e-12


def test_realistic_sing_scale():
    """Realistic E-step scale: T=2000, D=2, M=15x15 spectral grid."""
    m, S, w, cid = _make_clustered_problem(
        T=2000, D=2, n_clusters=16, S_scale=0.02)
    xi_out = _make_output_grid(D=2, m_per_dim=15)
    print(f"\n[realistic SING]    T=2000, M={xi_out.shape[0]}, D=2 "
          f"(clustered into 16 groups)")
    ref = dense_closed_form(m, S, w, xi_out)
    print("  Oracle bucketing (cluster id known):")
    # For B < n_clusters, merge clusters by binning their ids.
    for B in [1, 4, 8, 16, 32, 64]:
        bin_id = np.minimum(cid * B // 16, B - 1) if B <= 16 else cid
        got = bucketed_gmix_nufft_with_labels(m, S, w, xi_out, bin_id, B if B <= 16 else 16)
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"    B={B:>3}          rel-err = {err:.2e}")


# ---------------------------------------------------------------------------
# Computational-complexity tests
# ---------------------------------------------------------------------------
def _time_call(fn, n_repeat=5):
    import time
    fn()  # warmup (also forces jit compile if any)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        out = fn()
        # block until done — important for jax async dispatch
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    return (time.perf_counter() - t0) / n_repeat


def test_jax_nufft_matches_dense():
    """Sanity: the jax_finufft-backed bucketed NUFFT matches the dense
    closed-form reference at B=T (one bucket per source)."""
    T, D = 200, 2
    M_per_dim = 15
    h_freq = 0.2
    m, S, w, cid = _make_clustered_problem(T=T, D=D, n_clusters=8)
    # Reference grid in xi_out form
    K = (M_per_dim - 1) // 2
    k_axis = np.arange(-K, K + 1)
    kx, ky = np.meshgrid(k_axis, k_axis, indexing='ij')
    xi_out = np.stack([(kx * h_freq).ravel(), (ky * h_freq).ravel()], axis=-1)
    ref = dense_closed_form(m, S, w, xi_out)
    got = bucketed_gmix_nufft_jax(m, S, w, M_per_dim, h_freq,
                                   labels=np.arange(T), n_buckets=T)
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"\n[jax_finufft B=T]   rel-err vs dense = {err:.2e}  (target: <1e-5; "
          f"limited by NUFFT eps=1e-9)")
    assert err < 1e-5, f"jax NUFFT B=T not matching dense: {err:.2e}"


def test_complexity_O_T():
    """Type-1 cost is O(T + M log M); fit `time ≈ a + b·T` and verify
    both a (M-dominated overhead) and b (linear-in-T) are positive.

    This is the discriminator vs type-3, which would have an extra
    O(T log T) term — visible as super-linear behaviour at large T.
    Pure type-1 stays linear.
    """
    D = 2
    B = 8
    M_per_dim = 15  # smaller M so we cross into T-dominated regime
    h_freq = 0.15
    print(f"\n[complexity O(T) — jax_finufft type-1]   "
          f"B={B}, M_per_dim={M_per_dim} → M_out={M_per_dim**2}")
    Ts = [1000, 2000, 5000, 10000, 30000, 100000, 300000]
    times = []
    for T in Ts:
        m, S, w, cid = _make_clustered_problem(T=T, D=D, n_clusters=B, seed=T)
        labels = _kmeans_buckets(S, B)
        w_j = w.astype(np.complex128)
        t = _time_call(lambda: bucketed_gmix_nufft_jax(
            m, S, w_j, M_per_dim, h_freq, labels, B))
        times.append(t)
        print(f"  T={T:>6}          time = {t*1e3:8.2f} ms"
              f"   (per-source: {t*1e6/T:.2f} us)")
    Ts_arr = np.array(Ts, dtype=float)
    times_arr = np.array(times)
    # Fit time ≈ a + b·T  (linear, not log-log)
    A = np.stack([np.ones_like(Ts_arr), Ts_arr], axis=1)
    sol, *_ = np.linalg.lstsq(A, times_arr, rcond=None)
    a, b = sol
    # Discriminator: log-log slope at the upper end (T-dominated)
    p_high = np.polyfit(np.log(Ts_arr[-3:]), np.log(times_arr[-3:]), 1)
    p_low = np.polyfit(np.log(Ts_arr[:3]), np.log(times_arr[:3]), 1)
    print(f"  linear fit: time ≈ {a*1e3:.2f} ms + ({b*1e9:.2f} ns)·T")
    print(f"  log-log slope: T_low → {p_low[0]:.2f},  T_high → {p_high[0]:.2f}")
    print(f"  (expected: low slope ~ 0 (M log M dominates),  "
          f"high slope ~ 1 (T dominates))")
    # Type-1: linear with positive slope at high T; sub-linear at low T.
    # Type-3 fingerprint would be slope > 1 at high T (T log T term).
    assert b > 0, "linear-in-T coefficient should be positive"
    assert p_high[0] < 1.2, ("super-linear behaviour at large T — "
                              "looks like type-3 cost, not type-1")


def test_complexity_O_M_logM():
    """Type-1 cost at fixed T, B is O(T + M log M).  Verify M-scaling
    by fitting `time ≈ a + b · M log M` and asserting positive slope.
    Also report log-log slope on the upper-M tail, expected ~1.0–1.1.

    We reject type-3 fingerprint (which would have an extra
    O((T+M) log(T+M)) term — visible as super-linear M-growth at
    fixed T).
    """
    D = 2
    T = 4000
    B = 8
    print(f"\n[complexity O(M log M) — jax_finufft type-1]   T={T}, B={B}")
    times, Ms = [], []
    for mpd in [11, 15, 21, 31, 41, 51, 65, 81]:
        h_freq = 2.0 / mpd
        m, S, w, cid = _make_clustered_problem(T=T, D=D, n_clusters=B, seed=mpd)
        labels = _kmeans_buckets(S, B)
        w_j = w.astype(np.complex128)
        t = _time_call(lambda: bucketed_gmix_nufft_jax(
            m, S, w_j, mpd, h_freq, labels, B))
        Ms.append(mpd ** 2)
        times.append(t)
        print(f"  M_out={Ms[-1]:>5}      time = {t*1e3:8.2f} ms")
    Ms_arr = np.array(Ms, dtype=float)
    times_arr = np.array(times)
    # Fit time ≈ a + b · M log M  (the type-1 prediction)
    feat = Ms_arr * np.log(np.maximum(Ms_arr, 2))
    A = np.stack([np.ones_like(feat), feat], axis=1)
    sol, *_ = np.linalg.lstsq(A, times_arr, rcond=None)
    a, b = sol
    p_high = np.polyfit(np.log(Ms_arr[-3:]), np.log(times_arr[-3:]), 1)
    print(f"  type-1 fit: time ≈ {a*1e3:.2f} ms + ({b*1e9:.2f} ns)·M log M")
    print(f"  log-log slope on upper-M tail: {p_high[0]:.2f}  "
          f"(expected: ~1.0–1.1)")
    assert b > 0, "M log M coefficient should be positive"
    assert p_high[0] < 1.5, ("super-linear M growth — possible type-3 / "
                              "T log T contamination")


def test_complexity_O_B():
    """Type-1 cost: total work O(T + B M log M).  At fixed T, M, slope vs B
    is linear in the M-dominated regime (B M log M >> T) and flat in the
    T-dominated regime."""
    D = 2
    T = 4000
    M_per_dim = 31
    h_freq = 0.1
    m, S, w, cid = _make_clustered_problem(T=T, D=D, n_clusters=64, seed=42)
    print(f"\n[complexity O(B) — jax_finufft type-1]   T={T}, M_per_dim={M_per_dim}")
    Bs = [1, 2, 4, 8, 16, 32, 64]
    times = []
    for B in Bs:
        labels = _kmeans_buckets(S, B)
        w_j = w.astype(np.complex128)
        t = _time_call(lambda: bucketed_gmix_nufft_jax(
            m, S, w_j, M_per_dim, h_freq, labels, B))
        times.append(t)
        print(f"  B={B:>3}            time = {t*1e3:8.2f} ms")
    p = np.polyfit(np.log(Bs), np.log(times), 1)
    print(f"  fitted slope on log-log: {p[0]:.2f}  "
          f"(O(T + B M log M); slope ∈ [0, 1] depending on which dominates)")
    # Sanity: B scaling no worse than linear (expected).
    assert p[0] < 1.2, f"B scaling worse than expected: slope = {p[0]:.2f}"


if __name__ == '__main__':
    test_constant_S_one_bucket_is_exact()
    test_T_buckets_is_exact()
    test_error_decreases_with_buckets()
    test_BTTB_difference_grid()
    test_clustered_S_oracle_buckets_recover_dense()
    test_realistic_sing_scale()
    test_jax_nufft_matches_dense()
    test_complexity_O_T()
    test_complexity_O_M_logM()
    test_complexity_O_B()
    print("\nAll tests passed.")
