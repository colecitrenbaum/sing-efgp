"""
Closed-form Gaussian emissions update for the EFGP-SING M-step.

Model: ``y_t = C x_t + d + ε_t``,  ``ε_t ~ N(0, diag(R))``.

Given the variational marginals ``m_t = E_q[x_t]``, ``S_t = Var_q[x_t]``, the
closed-form maximizer of ``Σ_t E_q[log p(y_t | x_t)]`` is per output dim n:

    [C_n; d_n] = solve(M, b_n)

with the design Gram

    M = Σ_t [[ m_t m_t^T + S_t,  m_t ],
            [        m_t^T,       1  ]]   ∈ R^{(D+1) × (D+1)}

and per-output rhs

    b_n = Σ_t y_{t, n} [m_t; 1]   ∈ R^{D+1}.

The diagonal observation noise update is then

    R_n = (1/T) Σ_t [(y_{t,n} - C_n m_t - d_n)^2 + C_n S_t C_n^T].
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def update_emissions_gaussian(
    ms: jnp.ndarray,           # (K, T, D)
    Ss: jnp.ndarray,           # (K, T, D, D)
    ys: jnp.ndarray,           # (K, T, N)
    t_mask: jnp.ndarray = None,    # (K, T) bool, True where an observation exists
    update_R: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Closed-form ML update of (C, d, R), aggregated across K trials and T times.

    Sufficient statistics are linear sums over (k, t) — multi-trial
    aggregation is just one extra reduction axis.
    """
    K, T, D = ms.shape
    N = ys.shape[-1]

    if t_mask is None:
        t_mask = jnp.ones((K, T), dtype=bool)
    w = t_mask.astype(ms.dtype)                                # (K, T)

    # Sufficient statistics: reduce over (K, T)
    sum_m = (w[..., None] * ms).sum(axis=(0, 1))                                 # (D,)
    sum_mmT_plus_S = (w[..., None, None] * (
        ms[..., :, None] * ms[..., None, :] + Ss
    )).sum(axis=(0, 1))                                                          # (D, D)
    T_eff = w.sum()                                                              # scalar

    # M (D+1, D+1)
    M = jnp.block([[sum_mmT_plus_S, sum_m[:, None]],
                   [sum_m[None, :], T_eff[None, None]]])
    M = M + 1e-9 * jnp.eye(D + 1)

    # B (D+1, N): b_n stacked as columns
    sum_ym = (w[..., None, None] * ys[..., None, :] * ms[..., :, None]
              ).sum(axis=(0, 1))                                                 # (D, N)
    sum_y = (w[..., None] * ys).sum(axis=(0, 1))                                 # (N,)
    B = jnp.concatenate([sum_ym, sum_y[None, :]], axis=0)                         # (D+1, N)

    Cd = jnp.linalg.solve(M, B)                                  # (D+1, N)
    C_new = Cd[:D, :].T                                          # (N, D)
    d_new = Cd[D, :]                                              # (N,)

    if update_R:
        # Per-time residual variance:  E[(y - Cx - d)^2] = (y - Cm - d)^2 + C S C^T
        Cm = ms @ C_new.T                                         # (K, T, N)
        resid_sq = (ys - Cm - d_new) ** 2                         # (K, T, N)
        # C S C^T  per (k, t, n)
        CSC = jnp.einsum('nd, ktde, ne -> ktn', C_new, Ss, C_new)
        per_kt = (resid_sq + CSC) * w[..., None]                  # (K, T, N)
        R_new = per_kt.sum(axis=(0, 1)) / jnp.maximum(T_eff, 1.0) # (N,)
        R_new = jnp.maximum(R_new, 1e-6)
    else:
        R_new = None

    return C_new, d_new, R_new
