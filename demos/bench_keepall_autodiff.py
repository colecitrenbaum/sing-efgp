"""Head-to-head: production drop (gmix_batched) vs keep-all autodiff
(gmix_full_batched) on a Duffing latent SDE. Reports latent RMSE,
recovered (ell, sigma_f^2), and wall clock. See KEEPALL_AUTODIFF_NOTES.md.

Usage:  python demos/bench_keepall_autodiff.py [T] [n_em]
On GPU: check (1) RMSE parity, (2) per-fit wall, (3) peak memory.
"""
import sys, time
import jax; jax.config.update("jax_enable_x64", True)   # MUST precede jax use
import jax.numpy as jnp, jax.random as jr, numpy as np
from sing.likelihoods import Likelihood
from sing.simulate_data import simulate_sde, simulate_gaussian_obs
from sing.efgp_em import fit_efgp_sing_jax

T     = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
n_em  = int(sys.argv[2]) if len(sys.argv) > 2 else 15
D, K, sigma, t_max = 2, 1, 0.4, T * 0.01

def drift(x, t):
    return jnp.array([x[1], x[0] - x[0]**3 - 0.3 * x[1]])   # Duffing
sig = lambda x, t: sigma * jnp.eye(D)
xs = simulate_sde(jr.PRNGKey(0), x0=jnp.array([1.0, 0.0]), f=drift,
                  t_max=t_max, n_timesteps=T, sigma=sig)[None]
N = 6; rng = np.random.default_rng(2)
C = jnp.asarray(rng.standard_normal((N, D)) * 0.6)
op_t = dict(C=C, d=jnp.zeros(N), R=jnp.full((N,), 0.05))
ys = simulate_gaussian_obs(jr.PRNGKey(9), xs[0], op_t)[None]

class GLik(Likelihood):
    def ell(self, y, m, v, op):
        R = op['R']
        return jnp.sum(-0.5*jnp.log(2*jnp.pi*R) - 0.5*((y-m)**2+v)/R)

tm  = jnp.ones((K, T), bool)
lik = GLik(ys, tm)
op  = dict(C=C, d=jnp.zeros(N), R=jnp.full((N,), 0.05))
ip  = jax.tree_util.tree_map(lambda z: jnp.broadcast_to(z, (K,)+z.shape),
                             dict(mu0=jnp.zeros(D), V0=jnp.eye(D)*0.3))
tg  = jnp.linspace(0., t_max, T)
rho = jnp.linspace(0.05, 0.8, n_em)

def run(method):
    t0 = time.time()
    mp, _, _, _, h = fit_efgp_sing_jax(
        likelihood=lik, t_grid=tg, output_params=op, init_params=ip,
        latent_dim=D, lengthscale=0.8, variance=1.0, sigma=sigma,
        sigma_drift_sq=sigma**2, eps_grid=1e-2, n_em_iters=n_em,
        n_estep_iters=6, rho_sched=rho, learn_emissions=False,
        learn_kernel=True, kernel_warmup_iters=6, mstep_lr=0.01,
        verbose=False, estep_method='gmix', qx_moments_method=method)
    m = np.asarray(mp['m']); fin = bool(np.all(np.isfinite(m)))
    rmse = float(np.sqrt(np.mean((m - np.asarray(xs))**2))) if fin else np.nan
    return dict(rmse=rmse, ell=float(h.lengthscale[-1]),
                var=float(h.variance[-1]), wall=time.time()-t0, fin=fin)

print(f"Duffing T={T} K={K} N={K*(T-1)} n_em={n_em}  backend={jax.default_backend()}")
for method in ['gmix_batched', 'gmix_full_batched']:
    r = run(method)
    print(f"  {method:18s} rmse={r['rmse']:.4f} ell={r['ell']:.3f} "
          f"var={r['var']:.3f} wall={r['wall']:.0f}s finite={int(r['fin'])}",
          flush=True)
