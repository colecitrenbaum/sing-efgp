"""
diag_sparsegp_nan_trace.py

At lr=1e-1 SparseGP NaNs at iter 33 with min ℓ_dim=0.82 — not pathological.
Root-cause hunt: capture per-iter (ELBO, ℓ-per-dim, σ², output_scale_raw)
and identify which value goes bad first.

Method: re-run lr=8e-2 (stable) and lr=1e-1 (NaN) with print_interval=1 +
record drift_params_history and ELBO history. Plot side-by-side.

Run:
  ~/myenv/bin/python demos/diag_sparsegp_nan_trace.py
"""
from __future__ import annotations

import io
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import jax.random as jr

from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.sde import SparseGP
from sing.kernels import RBF
from sing.expectation import GaussHermiteQuadrature
from sing.sing import fit_variational_em
from sing.initialization import initialize_zs


D = 2
N_EM = 50
LS_INIT = 0.7
VAR_INIT = 1.0


CFG = dict(
    name='Duffing double-well',
    drift_fn=lambda x, t: jnp.stack([x[1], x[0] - x[0]**3 - 0.5*x[1]]),
    x0=jnp.array([1.2, 0.0]),
    T=400, t_max=15.0, sigma=0.2, N_obs=8, seed=13,
)


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def make_data(cfg):
    sigma_fn = lambda x, t: cfg['sigma'] * jnp.eye(D)
    xs = simulate_sde(jr.PRNGKey(cfg['seed']), x0=cfg['x0'],
                      f=cfg['drift_fn'], t_max=cfg['t_max'],
                      n_timesteps=cfg['T'], sigma=sigma_fn)
    xs = jnp.clip(xs, -3.0, 3.0)
    rng = np.random.default_rng(cfg['seed'])
    C_true = jnp.asarray(rng.standard_normal((cfg['N_obs'], D)) * 0.5)
    out_true = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
                    R=jnp.full((cfg['N_obs'],), 0.05))
    ys = simulate_gaussian_obs(jr.PRNGKey(cfg['seed'] + 1), xs, out_true)
    op = dict(C=C_true, d=jnp.zeros(cfg['N_obs']),
              R=jnp.full((cfg['N_obs'],), 0.1))
    ip = jax.tree_util.tree_map(
        lambda x: x[None], dict(mu0=jnp.zeros(D), V0=jnp.eye(D) * 0.1))
    t_grid = jnp.linspace(0., cfg['t_max'], cfg['T'])
    lik = GLik(ys[None], jnp.ones((1, cfg['T']), dtype=bool))
    return xs, ys, lik, op, ip, t_grid


def fit_with_trace(lik, op, ip, t_grid, sigma, lr):
    quad = GaussHermiteQuadrature(D=D, n_quad=5)
    zs = initialize_zs(D=D, zs_lim=2.5, num_per_dim=8)
    sparse = SparseGP(zs=zs, kernel=RBF(latent_dim=D), expectation=quad)
    drift_params0 = dict(length_scales=jnp.full((D,), float(LS_INIT)),
                          output_scale=jnp.asarray(math.sqrt(VAR_INIT)))
    rho_sched = jnp.linspace(0.05, 0.7, N_EM)
    history = []
    elbo_log = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(elbo_log):
        mp, *_ = fit_variational_em(
            key=jr.PRNGKey(33), fn=sparse, likelihood=lik, t_grid=t_grid,
            drift_params=drift_params0, init_params=ip, output_params=op,
            sigma=sigma, rho_sched=rho_sched, n_iters=N_EM, n_iters_e=10,
            n_iters_m=4, perform_m_step=True, learn_output_params=False,
            learning_rate=jnp.full((N_EM,), lr),
            print_interval=1, drift_params_history=history)
    wall = time.perf_counter() - t0
    elbo_text = elbo_log.getvalue()
    # Parse ELBO from print lines
    elbos = []
    for line in elbo_text.split('\n'):
        if 'elbo' in line.lower():
            for tok in line.replace(',', ' ').split():
                try:
                    if '.' in tok or 'e' in tok or '-' in tok or '+' in tok:
                        v = float(tok)
                        elbos.append(v)
                        break
                except ValueError:
                    continue
    # Per-iter raw values
    ls_per = np.stack([np.asarray(d['length_scales']) for d in history])
    var_per = np.array([float(np.asarray(d['output_scale'])) ** 2
                          for d in history])
    output_scale_raw = np.array([float(np.asarray(d['output_scale']))
                                   for d in history])
    return dict(
        ls_per=ls_per, var_per=var_per, output_scale_raw=output_scale_raw,
        elbos=np.array(elbos), wall=wall, elbo_text=elbo_text,
    )


def main():
    print(f"[trace] SparseGP NaN-trace on {CFG['name']} (oracle C frozen)",
          flush=True)
    xs, ys, lik, op, ip, t_grid = make_data(CFG)

    LRS = [8e-2, 1e-1]
    results = {}
    for lr in LRS:
        print(f"\n[trace] lr={lr:.0e} (n_em={N_EM})...", flush=True)
        r = fit_with_trace(lik, op, ip, t_grid, CFG['sigma'], lr)
        # Find first NaN iter
        first_nan = None
        for t in range(len(r['ls_per'])):
            if (np.any(np.isnan(r['ls_per'][t])) or
                    np.isnan(r['var_per'][t]) or
                    np.isnan(r['output_scale_raw'][t])):
                first_nan = t
                break
        # Per-iter ELBO (not always 1:1 with iters; print sketchy)
        n_elbo = len(r['elbos'])
        print(f"  wall={r['wall']:.1f}s, history len={len(r['ls_per'])}, "
              f"elbos len={n_elbo}", flush=True)
        if first_nan is not None:
            print(f"  *** NaN at iter {first_nan} ***", flush=True)
            # Print 3 iters before NaN
            for t in range(max(0, first_nan-3), min(len(r['ls_per']), first_nan+1)):
                print(f"    iter {t:3d}: ℓ_per={r['ls_per'][t]}, "
                      f"σ²={r['var_per'][t]:.4g}, "
                      f"output_scale_raw={r['output_scale_raw'][t]:.4g}",
                      flush=True)
        else:
            print(f"  no NaN. final ℓ={r['ls_per'][-1]}, "
                  f"σ²={r['var_per'][-1]:.4g}", flush=True)
            print(f"  σ² range: [{r['var_per'].min():.4g}, "
                  f"{r['var_per'].max():.4g}]", flush=True)
            print(f"  ℓ_per per dim: dim0 [{r['ls_per'][:,0].min():.4g}, "
                  f"{r['ls_per'][:,0].max():.4g}], "
                  f"dim1 [{r['ls_per'][:,1].min():.4g}, "
                  f"{r['ls_per'][:,1].max():.4g}]", flush=True)
        results[lr] = r

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for row, lr in enumerate(LRS):
        r = results[lr]
        iters = np.arange(len(r['ls_per']))

        ax = axes[row, 0]
        ax.plot(iters, r['ls_per'][:, 0], '-o', label='ℓ_dim0',
                 markersize=2, color='C0')
        ax.plot(iters, r['ls_per'][:, 1], '-o', label='ℓ_dim1',
                 markersize=2, color='C1')
        ax.set_yscale('log')
        ax.set_xlabel('EM iter'); ax.set_ylabel('ℓ_per_dim')
        ax.set_title(f"lr={lr:.0e} | length_scales per dim")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(iters, r['var_per'], '-o', markersize=2, color='C2')
        ax.set_yscale('log')
        ax.set_xlabel('EM iter'); ax.set_ylabel('σ² (output_scale²)')
        ax.set_title(f"lr={lr:.0e} | output variance")
        ax.grid(alpha=0.3)

        ax = axes[row, 2]
        ax.plot(iters, r['output_scale_raw'], '-o', markersize=2, color='C3')
        ax.axhline(0, color='red', linestyle=':', linewidth=1.5,
                    label='zero (sign flip)')
        ax.set_xlabel('EM iter'); ax.set_ylabel('output_scale (raw)')
        ax.set_title(f"lr={lr:.0e} | output_scale (raw, signed)")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[row, 3]
        if len(r['elbos']) > 0:
            ax.plot(np.arange(len(r['elbos'])), r['elbos'],
                     '-o', markersize=2, color='C4')
        ax.set_xlabel('print idx'); ax.set_ylabel('ELBO (parsed)')
        ax.set_title(f"lr={lr:.0e} | ELBO (parsed from print)")
        ax.grid(alpha=0.3)

    fig.suptitle(f"SparseGP per-iter trace, oracle C frozen, "
                 f"on {CFG['name']}\n"
                 f"What goes wrong first at lr=1e-1?",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/tmp/diag_sparsegp_nan_trace.png', dpi=110)
    print('\n[trace] Saved /tmp/diag_sparsegp_nan_trace.png', flush=True)

    # Save text logs
    for lr, r in results.items():
        with open(f'/tmp/diag_sparsegp_nan_trace_lr{lr:.0e}.log', 'w') as f:
            f.write(r['elbo_text'])
    print('[trace] Saved per-LR text logs to /tmp/diag_sparsegp_nan_trace_*.log',
          flush=True)


if __name__ == '__main__':
    main()
