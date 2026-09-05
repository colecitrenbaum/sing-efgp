"""Profile where SING-EFGP wall time actually goes, on GPU.

Three complementary attributions:
  (1) COMPILE vs RUN — full-fit cold (incl. JIT) vs warm (jit cache reused).
  (2) COMPONENT — isolated warm per-call timing of each recurring op:
        smoother (natural_to_marginal_params = assoc-scan log-Z + its grad),
        forward-only log-normalizer (isolates the autodiff/backward cost),
        q(f) update (compute_mu_r_gmix), drift moments (gmix gather),
        nat_grad_transition, nat_grad_likelihood, kernel M-step.
  (3) DIFFERENTIAL — warm full fits at two n_estep_iters; the slope isolates
      the true per-inner-iter cost via the real code path (cross-checks (2)).

Run on GPU: python demos/profile_sing.py
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_jax_drift import FrozenEFGPDrift
from sing.utils.sing_helpers import (
    natural_to_marginal_params, natural_to_mean_params,
    compute_log_normalizer_parallel, dynamics_to_natural_params)
from sing.sing import nat_grad_likelihood, nat_grad_transition
from sing.efgp_gmix_spreader import stencil_radius_for, pick_grid_size
import demos.bench_gpdrift_scaling as run
import demos.bench_gpdrift_x64 as base

N_REP = 7


def warm(fn, *args, n=N_REP):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def build(T, seed=0):
    xs, lik, op, ip, t_grid, sigma, *_ = run.make_data(T, seed)
    D = base.D
    xt = jnp.asarray(run.data_aware_template(np.asarray(xs)))
    grid = jp.spectral_grid_se(0.7, base.VAR_INIT, xt, eps=1e-3)
    tm = jnp.ones((1, T), dtype=bool)
    # init natural params (diffusion seed), K=1
    def _init_one(mu0_, V0_, tm_):
        A = jnp.zeros((T - 1, D, D)); b = jnp.zeros((T, D)).at[0].set(mu0_)
        Q = jnp.tile((sigma ** 2) * jnp.eye(D), (T, 1, 1)).at[0].set(V0_)
        return dynamics_to_natural_params(A, b, Q, tm_)
    nat_p = vmap(_init_one)(ip['mu0'], ip['V0'], tm)
    return dict(xs=xs, lik=lik, op=op, ip=ip, t_grid=t_grid, sigma=sigma,
                D=D, xt=xt, grid=grid, tm=tm, nat_p=nat_p)


def fit_once(b, n_em, n_estep):
    rho = jnp.linspace(0.05, 0.7, n_em)
    t0 = time.perf_counter()
    mp, _, _, _, hist = base.efgp_em.fit_efgp_sing_jax(
        likelihood=b['lik'], t_grid=b['t_grid'], output_params=b['op'],
        init_params=b['ip'], latent_dim=b['D'], lengthscale=0.7,
        variance=base.VAR_INIT, sigma=b['sigma'], sigma_drift_sq=b['sigma'] ** 2,
        eps_grid=1e-3, S_marginal=2, n_em_iters=n_em, n_estep_iters=n_estep,
        rho_sched=rho, learn_emissions=False, update_R=False, learn_kernel=True,
        n_mstep_iters=base.N_M_INNER, mstep_lr=base.MSTEP_LR,
        n_hutchinson_mstep=4, kernel_warmup_iters=8, X_template=b['xt'],
        estep_method='gmix', verbose=False)
    jax.block_until_ready(mp['m'])
    return time.perf_counter() - t0


def main():
    import os
    T_LIST = [int(x) for x in os.environ.get('PROFILE_T', '2000,10000').split(',')]
    print(f"device={jax.devices()} backend={jax.default_backend()}", flush=True)
    for T in T_LIST:
        print(f"\n############### T={T} (K=1, D=2) ###############", flush=True)
        b = build(T)
        grid, tm, nat_p, D = b['grid'], b['tm'], b['nat_p'], b['D']
        sds = b['sigma'] ** 2
        del_t = b['t_grid'][1:] - b['t_grid'][:-1]
        M = grid.M
        print(f"grid M={M} mtot={grid.mtot_per_dim[0]}", flush=True)

        # gmix sizing (mirror fit startup)
        h_spec = float(grid.h_per_dim[0])
        V0 = np.asarray(b['ip']['V0'][0]); sig0 = float(np.sqrt(np.linalg.eigvalsh(V0).max()))
        m_ext = float((np.asarray(b['xt']).max(0) - np.asarray(b['xt']).min(0)).max())
        fine_N = pick_grid_size(h_spec=h_spec, m_extent=m_ext, sigma_max=sig0)
        h_grid = 1.0 / (fine_N * h_spec)
        stencil_r = max(8, int(stencil_radius_for(b['ip']['V0'][0][None], h_grid, n_sigma=1.5)))
        gather_N = 1 << (int(2 * grid.mtot_per_dim[0]) - 1).bit_length()

        # ---------- (2) COMPONENTS (warm per-call ms) ----------
        comp = {}
        # smoother: marginals (assoc-scan logZ + reverse-mode grad)
        f_marg = jax.jit(vmap(natural_to_marginal_params))
        comp['smoother_marg(+grad)'] = warm(f_marg, nat_p, tm)
        f_mean = jax.jit(vmap(natural_to_mean_params))
        comp['smoother_mean(+grad)'] = warm(f_mean, nat_p, tm)
        # forward-only log-normalizer (no grad) — isolates backward cost
        def _lognorm(np_, tm_):
            return vmap(lambda p, m: compute_log_normalizer_parallel(
                (-2) * p['J'], (-1) * p['L'], p['h'], m))(np_, tm_)
        f_lz = jax.jit(_lognorm)
        comp['lognorm_fwd_only'] = warm(f_lz, nat_p, tm)

        # marginals for downstream inputs
        mp_b, _ = f_marg(nat_p, tm)
        ms, Ss, SSs = mp_b['m'], mp_b['S'], mp_b['SS']

        # q(f): compute_mu_r_gmix (full multi-trial wrapper w/o moments)
        f_qf = jax.jit(lambda ms, Ss, SSs: jpd.qf_and_moments_gmix_jax(
            ms, Ss, SSs, del_t, tm, grid, sigma_drift_sq=sds, D_lat=D, D_out=D,
            fine_N=fine_N, stencil_r=stencil_r, return_top=True))
        comp['qf_gmix(+moments_nufft)'] = warm(f_qf, ms, Ss, SSs)
        mu_r, Ef, Eff, Edfdx, top = f_qf(ms, Ss, SSs)

        # drift moments via gmix gather (the DEFAULT qx_moments override)
        f_dm = jax.jit(lambda mu_r, ms, Ss: jpd.drift_moments_gmix_jax(
            mu_r, grid, ms, Ss, D_lat=D, D_out=D, gather_N=gather_N, stencil_r=8))
        comp['drift_moments_gmix'] = warm(f_dm, mu_r, ms, Ss)
        Ef, Eff, Edfdx = f_dm(mu_r, ms, Ss)

        # nat_grad_transition (vmap grad of neg-CE via frozen shim)
        def _ngt(nat_p_, Ef, Eff, Edfdx):
            mean_b, _ = vmap(natural_to_mean_params)(nat_p_, tm)
            shim = FrozenEFGPDrift(latent_dim=D, t_grid=b['t_grid'],
                                   Ef_per_t=Ef, Eff_per_t=Eff, Edfdx_per_t=Edfdx)
            inputs = jnp.zeros((1, ms.shape[1], 1))
            ie = jnp.zeros((D, 1))
            return vmap(lambda k, fr, mp_, tm_, ip_, inp_: nat_grad_transition(
                k, fr, None, {}, tm_, ip_, b['t_grid'], mp_, inp_, ie, b['sigma']))(
                jr.split(jr.PRNGKey(0), 1), shim, mean_b, tm, b['ip'], inputs)
        f_ngt = jax.jit(_ngt)
        comp['nat_grad_transition'] = warm(f_ngt, nat_p, Ef, Eff, Edfdx)

        # nat_grad_likelihood
        def _ngl(nat_p_):
            mean_b, _ = vmap(natural_to_mean_params)(nat_p_, tm)
            return vmap(lambda mp_, tm_, ys_: nat_grad_likelihood(
                mp_, tm_, ys_, b['lik'], b['op']))(mean_b, b['lik'].t_mask, b['lik'].ys_obs)
        try:
            f_ngl = jax.jit(_ngl)
            comp['nat_grad_likelihood'] = warm(f_ngl, nat_p)
        except Exception as e:
            comp['nat_grad_likelihood'] = float('nan')
            print(f"  ngl failed: {type(e).__name__}: {str(e)[:70]}", flush=True)

        # M-step (dense M×M Cholesky Adam loop)
        try:
            ws = grid.ws
            z_r = jpd.drift_moments_jax  # placeholder; build z_r from h_r
            ws_safe = jnp.where(jnp.abs(ws.real) < 1e-30, 1e-30, ws.real)
            # h_r ≈ ws * mu_r for the collapsed M-step RHS summary
            z_r_val = (ws.real[None, :] * mu_r) / ws_safe[None, :]
            def _mstep():
                return jpd.m_step_kernel_jax(
                    math.log(0.7), math.log(base.VAR_INIT), mu_r_fixed=mu_r,
                    z_r=z_r_val, top=top, xis_flat=grid.xis_flat,
                    h_per_dim=grid.h_per_dim, D_lat=D, D_out=D,
                    n_inner=base.N_M_INNER, lr=base.MSTEP_LR)
            comp['m_step_kernel(4 adam)'] = warm(lambda: _mstep(), n=3)
        except Exception as e:
            comp['m_step_kernel(4 adam)'] = float('nan')
            print(f"  mstep failed: {type(e).__name__}: {str(e)[:70]}", flush=True)

        print("\n  -- COMPONENT warm per-call (ms) --", flush=True)
        for k, v in comp.items():
            print(f"    {k:28s} {v:9.3f}", flush=True)
        per_inner_est = (comp['smoother_marg(+grad)'] + comp['smoother_mean(+grad)']
                         + comp['qf_gmix(+moments_nufft)'] + comp['drift_moments_gmix']
                         + comp['nat_grad_transition'] + comp.get('nat_grad_likelihood', 0))
        print(f"    {'~per-inner-iter (sum)':28s} {per_inner_est:9.3f}", flush=True)

        # ---------- (1)+(3) COMPILE + DIFFERENTIAL ----------
        n_em = 6
        for n_estep in [6, 12]:
            t_cold = fit_once(b, n_em, n_estep)      # incl compile
            t_warm = fit_once(b, n_em, n_estep)      # cache reused
            print(f"\n  fit n_em={n_em} n_estep={n_estep}: "
                  f"cold={t_cold:6.1f}s warm={t_warm:6.1f}s "
                  f"compile≈{t_cold - t_warm:6.1f}s", flush=True)
            if n_estep == 6:
                warm6 = t_warm
            else:
                warm12 = t_warm
        per_inner_meas = (warm12 - warm6) / (n_em * 6) * 1e3
        print(f"\n  DIFFERENTIAL per-inner-iter (measured, real path) = "
              f"{per_inner_meas:.3f} ms  vs component-sum {per_inner_est:.3f} ms",
              flush=True)


if __name__ == '__main__':
    main()
