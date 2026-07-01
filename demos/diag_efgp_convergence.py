"""Is the EFGP-SING neural fit converged?  Run it much longer and inspect.

Runs EFGP-SING for N_EM=100 (rho ramps 0.05->0.7 over the first 40 iters, then
holds at 0.7) and reports/plots the ell and sigma_f^2 trajectories plus the
last-20-iter drift, so we can see whether the hypers (esp. sigma_f^2, which was
still declining at iter 40) have actually plateaued.  Also compares the final
q(x) latents to the 40-iter bench fit to see if the latents themselves moved.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_convergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SING = Path(__file__).resolve().parent.parent
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sing.efgp_em as em
from sing.likelihoods import Likelihood
from sing.inputs import InputSignals
import demos.bench_neural_efgp_vs_sparsegp as b
from sing.initialization import initialize_params_pca

OUT = b.OUT_DIR / "efgp_convergence.png"
N_EM_LONG = 100


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


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
    X_template = (jnp.linspace(float(lo.min()), float(hi.max()), max(n_timesteps, 64))[:, None]
                  * jnp.ones((1, b.D)))

    # rho: same ramp as the bench for the first 40, then hold at 0.7.
    rho = jnp.concatenate([jnp.linspace(0.05, 0.7, 40),
                           jnp.full(N_EM_LONG - 40, 0.7)])

    T = ys.shape[1]
    lik = GLik(ys, jnp.ones((1, T), dtype=bool))
    ip = dict(mu0=x0, V0=jnp.eye(b.D)[None])
    print(f"  fitting EFGP-SING for {N_EM_LONG} EM iters ...")
    mp, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(output_params), init_params=ip, latent_dim=b.D,
        lengthscale=ls_init, variance=b.VAR_INIT, sigma=b.SIGMA,
        sigma_drift_sq=b.SIGMA ** 2, eps_grid=1e-2, estep_method='gmix',
        n_em_iters=N_EM_LONG, n_estep_iters=b.N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=b.N_MSTEP_INNER, mstep_lr=b.MSTEP_LR,
        kernel_warmup_iters=8,
        input_signals=InputSignals(inputs), learn_input_effect=True,
        input_effect_warmup_iters=8, X_template=X_template, verbose=True)

    ls = np.asarray(hist.lengthscale); var = np.asarray(hist.variance)
    m100 = np.asarray(mp['m'][0])

    def tail_stats(x, k=20):
        t = x[-k:]
        return float(t.mean()), float(t.max() - t.min()), float(x[-1] - x[-k])
    lm, lr_, ld = tail_stats(ls); vm, vr, vd = tail_stats(var)
    print(f"\n  ell   @40={ls[39]:.3f}  @{N_EM_LONG}={ls[-1]:.3f}  "
          f"last20: mean={lm:.3f} band={lr_:.3f} drift={ld:+.3f}")
    print(f"  var   @40={var[39]:.3f}  @{N_EM_LONG}={var[-1]:.3f}  "
          f"last20: mean={vm:.3f} band={vr:.3f} drift={vd:+.3f}")

    # Did the latents move from the 40-iter bench fit?
    try:
        d = np.load(b.OUT_DIR / "bench.npz")
        m40 = d['m_efgp']
        dm = float(np.sqrt(np.mean((m100 - m40) ** 2)))
        print(f"  latent RMS change q(x) 40->{N_EM_LONG}: {dm:.4f} "
              f"(latent scale ~{np.std(m100):.1f})")
    except Exception as e:
        print(f"  (couldn't load bench.npz for latent comparison: {e})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    it = np.arange(len(ls))
    ax[0].plot(it, ls, '-o', ms=3, color='tab:blue'); ax[0].axvline(40, ls='--', c='gray', lw=1)
    ax[0].set_title(f"lengthscale ell  (@40={ls[39]:.2f} -> @{N_EM_LONG}={ls[-1]:.2f})")
    ax[0].set_xlabel("EM iter"); ax[0].set_ylabel("ell")
    ax[1].plot(it, var, '-o', ms=3, color='tab:red'); ax[1].axvline(40, ls='--', c='gray', lw=1)
    ax[1].set_title(f"variance sigma_f^2  (@40={var[39]:.3f} -> @{N_EM_LONG}={var[-1]:.3f})")
    ax[1].set_xlabel("EM iter"); ax[1].set_ylabel("sigma_f^2")
    fig.suptitle(f"EFGP-SING convergence over {N_EM_LONG} EM iters "
                 "(dashed = bench's stopping point, iter 40)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
