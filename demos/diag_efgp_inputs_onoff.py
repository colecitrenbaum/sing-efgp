"""Is the EFGP neural drift offset caused by the inputs wiring?

Fits EFGP on the neural data TWICE: inputs ON (learned B) and inputs OFF
(input_signals=None).  Reports the mean drift over the data and the
autonomous residual using the EM's ACTUAL converged q(f) for each.

  - If the offset (mean f) vanishes with inputs OFF -> inputs are wired wrong.
  - If it persists -> inputs are exonerated; the bias is elsewhere
    (drift fit / fixed-emissions interaction on this data).

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_inputs_onoff.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

import sing.efgp_em as em
from sing.likelihoods import Likelihood
import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def fit_no_inputs(ys, output_params, x0, t_grid, X_template, ls_init):
    T = ys.shape[1]
    lik = GLik(ys, jnp.ones((1, T), dtype=bool))
    ip = dict(mu0=x0, V0=jnp.eye(b.D)[None])
    rho = jnp.linspace(0.05, 0.7, b.N_EM)
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(output_params), init_params=ip, latent_dim=b.D,
        lengthscale=ls_init, variance=b.VAR_INIT, sigma=b.SIGMA,
        sigma_drift_sq=b.SIGMA ** 2, eps_grid=1e-2, estep_method='gmix',
        n_em_iters=b.N_EM, n_estep_iters=b.N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=b.N_MSTEP_INNER, mstep_lr=b.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8,
        X_template=X_template, verbose=True,
    )
    return mp, hist


def report(tag, hist, mp, t_grid, inputs=None):
    m = np.asarray(mp['m'][0])
    dt = float(t_grid[1] - t_grid[0])
    if inputs is not None and hist.input_effect is not None:
        Bv = (np.asarray(hist.input_effect) @ np.asarray(inputs[0]).T).T
    else:
        Bv = np.zeros_like(m)
    u_auto = (m[1:] - m[:-1]) / dt - Bv[:-1]
    f = em.posterior_drift_mean(hist, m)
    fmean = f.mean(0)
    resid = (u_auto - f[:-1]).mean(0)
    print(f"  [{tag}]  ell={hist.lengthscale[-1]:.2f} var={hist.variance[-1]:.3f}"
          f"  mean f = {np.round(fmean,3)} (||{np.linalg.norm(fmean):.3f}||)"
          f"  resid = {np.round(resid,3)} (||{np.linalg.norm(resid):.3f}||)")
    return fmean


def main():
    norm_neural = b.load_neural_data()[::b.SUBSAMPLE_T]
    n_timesteps = norm_neural.shape[0]
    t_grid = jnp.arange(n_timesteps) * (b.DT * b.SUBSAMPLE_T)
    ys = jnp.asarray(norm_neural[None])
    inputs, _, _ = b.build_inputs(n_timesteps)

    output_params, x0 = initialize_params_pca(b.D, ys)
    xs_pca = np.asarray((ys[0] - output_params['d']) @ output_params['C'])
    lo = xs_pca.min(0) - 1.0; hi = xs_pca.max(0) + 1.0
    ls_init = float(np.max(hi - lo)) / 8.0
    X_template = (jnp.linspace(lo.min(), hi.max(), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    print("  fitting EFGP inputs ON ...")
    mp_on, hist_on, _ = b.fit_efgp(ys, inputs, output_params, x0, t_grid,
                                   X_template, ls_init)
    print("  fitting EFGP inputs OFF ...")
    mp_off, hist_off = fit_no_inputs(ys, output_params, x0, t_grid,
                                     X_template, ls_init)

    print()
    f_on = report("inputs ON ", hist_on, mp_on, t_grid, inputs=inputs)
    f_off = report("inputs OFF", hist_off, mp_off, t_grid, inputs=None)
    print(f"\n  mean-f difference (ON - OFF) = {np.round(f_on - f_off, 3)}")
    if np.linalg.norm(f_off) < 0.15:
        print("  => offset VANISHES without inputs -> inputs wiring is the cause.")
    else:
        print("  => offset PERSISTS without inputs -> inputs exonerated; "
              "bias is in the drift/emissions fit.")


if __name__ == "__main__":
    main()
