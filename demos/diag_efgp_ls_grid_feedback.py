"""Is the choppy EFGP lengthscale trace caused by the adaptive-h moving mesh?

EFGP's default grid policy rebuilds the spectral grid spacing h(ell) from the
current ell every outer EM iter (adaptive-h, for jittability).  Hypothesis:
that ell->grid->objective->ell feedback produces the ~2-iter sawtooth in the
ell trace that SparseGP (fixed inducing points) doesn't show.

Test: fit EFGP twice on the neural data -- (A) adaptive-h (default) and
(B) pin_grid=True (grid frozen, no ell-dependent h) -- and overlay the
lengthscale traces.  If (B) is smooth, the sawtooth is the moving mesh.

Usage:
    /Users/colecitrenbaum/myenv/bin/python demos/diag_efgp_ls_grid_feedback.py
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

OUT = b.OUT_DIR / "efgp_ls_grid_feedback.png"


class GLik(Likelihood):
    def ell(self, y, mean, var, output_params):
        R = output_params['R']
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi * R)
                       - 0.5 * ((y - mean) ** 2 + var) / R)


def fit(ys, inputs, output_params, x0, t_grid, X_template, ls_init, pin):
    T = ys.shape[1]
    lik = GLik(ys, jnp.ones((1, T), dtype=bool))
    ip = dict(mu0=x0, V0=jnp.eye(b.D)[None])
    rho = jnp.linspace(0.05, 0.7, b.N_EM)
    kw = {}
    if pin:
        kw = dict(pin_grid=True, pin_grid_lengthscale=float(ls_init))
    _, _, _, _, hist = em.fit_efgp_sing_jax(
        likelihood=lik, t_grid=t_grid,
        output_params=dict(output_params), init_params=ip, latent_dim=b.D,
        lengthscale=ls_init, variance=b.VAR_INIT, sigma=b.SIGMA,
        sigma_drift_sq=b.SIGMA ** 2, eps_grid=1e-2, estep_method='gmix',
        n_em_iters=b.N_EM, n_estep_iters=b.N_ESTEP, rho_sched=rho,
        learn_emissions=False, update_R=False,
        learn_kernel=True, n_mstep_iters=b.N_MSTEP_INNER, mstep_lr=b.MSTEP_LR,
        kernel_warmup_iters=8,
        input_signals=InputSignals(inputs), learn_input_effect=True,
        input_effect_warmup_iters=8, X_template=X_template, verbose=False,
        **kw)
    return np.asarray(hist.lengthscale), np.asarray(hist.variance)


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

    print("  fitting EFGP adaptive-h (default) ...")
    ls_a, var_a = fit(ys, inputs, output_params, x0, t_grid, X_template, ls_init, pin=False)
    print("  fitting EFGP pin_grid=True ...")
    ls_p, var_p = fit(ys, inputs, output_params, x0, t_grid, X_template, ls_init, pin=True)

    # roughness metric: mean |2nd difference| over the post-warmup tail
    def rough(x):
        return float(np.mean(np.abs(np.diff(np.diff(x[8:])))))
    print(f"  ell roughness (mean |2nd diff|, post-warmup):"
          f"  adaptive-h={rough(ls_a):.4f}   pinned={rough(ls_p):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ls_a, '-o', ms=3, label=f'adaptive-h (default)  rough={rough(ls_a):.3f}',
            color='tab:blue')
    ax.plot(ls_p, '-s', ms=3, label=f'pin_grid=True  rough={rough(ls_p):.3f}',
            color='tab:green')
    ax.set_xlabel("EM iter"); ax.set_ylabel("lengthscale ell")
    ax.set_title("EFGP lengthscale: adaptive-h moving mesh vs pinned grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
