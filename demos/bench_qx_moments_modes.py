"""Subprocess-isolated bench: gmix_batched vs linearised_shim.

Compares the two production ``qx_moments_method`` modes on the canonical
SDE-recovery problem. Each (mode) cell spawns a fresh interpreter so JIT
costs and library init are paid cleanly per mode (no cross-mode cache
sharing → no order confound).

  - 'gmix_batched'    (DEFAULT): Gaussian-smoothed Ef/Edfdx via batched
                       gmix-gather precompute + custom_vjp shim
  - 'linearised_shim': point-eval drift moments + custom_vjp shim

(The earlier 'gmix_live' per-transition path was removed —
``gmix_batched`` strictly subsumes it, same gradients, ~2× faster at
K=10/T=500.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import subprocess
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))


def worker(args):
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr

    import sing.efgp_em as em
    import sing.efgp_jax_primitives as jp
    from sing.likelihoods import Likelihood
    from sing.simulate_data import simulate_sde, simulate_gaussian_obs

    D = 2; SIGMA = 0.4
    LS_TRUE, VAR_TRUE = 0.6, 1.0
    LS_INIT, VAR_INIT = 1.5, 1.0
    N_OBS = 6

    class GLik(Likelihood):
        def ell(self, y, mean, var, output_params):
            R = output_params['R']
            return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                           - 0.5 * ((y - mean) ** 2 + var) / R)

    K, T, T_MAX = args.K, args.T, args.T_max
    seed = args.seed

    X_template = jnp.linspace(-4.0, 4.0, 16)[:, None] * jnp.ones((1, D))
    grid = jp.spectral_grid_se(LS_TRUE, VAR_TRUE, X_template, eps=1e-2)
    M = grid.M
    keys = jr.split(jr.PRNGKey(seed), D)
    eps = jnp.stack([
        ((jax.random.normal(jr.split(k)[0], (M,))
          + 1j * jax.random.normal(jr.split(k)[1], (M,))).astype(grid.ws.dtype)
         / math.sqrt(2)) for k in keys
    ], axis=0)
    fk_draw = grid.ws[None, :] * eps
    def drift(x, t):
        return jnp.stack([
            jp.nufft2(x[None, :], fk_draw[r].reshape(*grid.mtot_per_dim),
                      grid.xcen, grid.h_per_dim, eps=6e-8).real[0]
            for r in range(D)
        ])

    rng = np.random.default_rng(seed + 1)
    x0_K = jnp.asarray(rng.uniform(-2.0, 2.0, (K, D)).astype(np.float64))
    xs_list = [simulate_sde(jr.PRNGKey(seed + 100 + k), x0=x0_K[k], f=drift,
                              t_max=T_MAX, n_timesteps=T,
                              sigma=lambda x, t: SIGMA * jnp.eye(D))
                for k in range(K)]
    xs_K = jnp.clip(jnp.stack(xs_list, axis=0), -3.5, 3.5).astype(jnp.float64)
    out_true = dict(C=jnp.asarray(rng.standard_normal((N_OBS, D)) * 0.5),
                    d=jnp.zeros(N_OBS, dtype=jnp.float64),
                    R=jnp.full((N_OBS,), 0.05, dtype=jnp.float64))
    ys_K = jnp.stack([simulate_gaussian_obs(jr.PRNGKey(seed + 200 + k),
                                              xs_K[k], out_true)
                        for k in range(K)], axis=0).astype(jnp.float64)

    def _fit():
        lik = GLik(ys_K, jnp.ones((K, T), dtype=bool))
        ip = dict(mu0=x0_K, V0=jnp.tile(0.1 * jnp.eye(D, dtype=jnp.float64),
                                          (K, 1, 1)))
        rho = jnp.linspace(0.05, 0.7, args.n_em)
        return em.fit_efgp_sing_jax(
            likelihood=lik,
            t_grid=jnp.linspace(0, T_MAX, T, dtype=jnp.float64),
            output_params=out_true, init_params=ip, latent_dim=D,
            lengthscale=LS_INIT, variance=VAR_INIT, sigma=SIGMA,
            sigma_drift_sq=SIGMA ** 2, eps_grid=1e-3,
            estep_method='gmix', S_marginal=2,
            n_em_iters=args.n_em, n_estep_iters=args.n_estep, rho_sched=rho,
            learn_emissions=False, update_R=False,
            learn_kernel=True, n_mstep_iters=4, mstep_lr=0.01,
            n_hutchinson_mstep=4, kernel_warmup_iters=8,
            pin_grid=True, pin_grid_lengthscale=LS_TRUE * 0.75,
            verbose=False, true_xs=np.asarray(xs_K),
            qx_moments_method=args.mode,
            restore_qf_variance='none',
        )

    t0 = time.time()
    mp1, _, _, _, hist1 = _fit()
    cold_wall = time.time() - t0
    t0 = time.time()
    mp2, _, _, _, hist2 = _fit()
    warm_wall = time.time() - t0

    out = dict(
        K=K, T=T, mode=args.mode, n_em=args.n_em, n_estep=args.n_estep,
        cold_wall=cold_wall, warm_wall=warm_wall,
        jit_cost=cold_wall - warm_wall,
        per_iter_warm=warm_wall / args.n_em,
        ls=float(hist2.lengthscale[-1]),
        var=float(hist2.variance[-1]),
        latent_rmse=float(hist2.latent_rmse[-1]),
    )
    print(json.dumps(out))


def orchestrator():
    sizes = [(2, 100, 1.6), (5, 250, 4.0), (10, 500, 8.0)]
    modes = ['gmix_batched', 'linearised_shim']
    n_em, n_estep = 15, 8

    py = sys.executable
    script = str(Path(__file__).resolve())
    rows = []

    print("=== Subprocess-isolated bench: qx_moments_method modes ===\n")
    print(f"  Each (K, T, mode) cell = fresh subprocess, two successive fits.\n"
          f"  cold = first fit (full JIT), warm = second fit (caches hot)\n")

    for K, T, T_MAX in sizes:
        print(f"--- K={K} T={T} (N_eff={K * (T - 1)}) ---")
        for mode in modes:
            cmd = [py, script, '--worker',
                   '--K', str(K), '--T', str(T), '--T-max', str(T_MAX),
                   '--mode', mode, '--n-em', str(n_em),
                   '--n-estep', str(n_estep), '--seed', '42']
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {mode:>16s}: FAILED")
                print(f"    stderr: {r.stderr[-400:]}")
                continue
            lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
            data = json.loads(lines[-1])
            rows.append(data)
            print(f"  {mode:>16s}: cold={data['cold_wall']:6.1f}s  "
                  f"warm={data['warm_wall']:6.1f}s  "
                  f"jit={data['jit_cost']:5.1f}s  "
                  f"per-iter={data['per_iter_warm']:.3f}s  "
                  f"ℓ={data['ls']:.4f}  σ²={data['var']:.4f}  "
                  f"lat={data['latent_rmse']:.4f}")
        print()

    print("--- warm wall ratios (relative to linearised_shim, isolation-clean) ---")
    for K, T, _ in sizes:
        cells = [r for r in rows if r['K'] == K and r['T'] == T]
        ref = next((r['warm_wall'] for r in cells
                     if r['mode'] == 'linearised_shim'), None)
        if ref is None:
            continue
        parts = []
        for mode in modes:
            cell = next((r for r in cells if r['mode'] == mode), None)
            if cell:
                parts.append(f"{mode}/linearised_shim={cell['warm_wall']/ref:.2f}x")
        print(f"  K={K} T={T}: {'  '.join(parts)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--K', type=int)
    p.add_argument('--T', type=int)
    p.add_argument('--T-max', type=float)
    p.add_argument('--mode', type=str)
    p.add_argument('--n-em', type=int)
    p.add_argument('--n-estep', type=int)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.worker:
        worker(args)
    else:
        orchestrator()


if __name__ == '__main__':
    main()
