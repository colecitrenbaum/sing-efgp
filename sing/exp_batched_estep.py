"""EXPERIMENT: batched grad-once E-step transition natural gradient.
Compute the SUMMED transition ELBO with batched exact gmix moments, then
jax.grad ONCE w.r.t. mean params -> all per-transition natural gradients,
no drops, no custom_vjp. (Ignores V.)
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from sing.utils.sing_helpers import mean_to_marginal_params, compute_neg_CE_initial
from sing.exp_full_moments import Ef_diff, Edfdx_diff
from sing.efgp_gmix_qx_moments import gmix_E_full_Eff, precompute_aux

def make_total_negCE(mu_r, grid, t_grid, trial_mask, init_params, sigma,
                     moment='exact'):
    xcen = grid.xcen; aux = precompute_aux(mu_r, grid)
    dt = t_grid[1:] - t_grid[:-1]                       # (T-1,)
    def _tr(A): return jnp.trace(A, axis1=-2, axis2=-1)
    def total(mean_params_b):                            # dict of (K,T,..)
        marg = jax.vmap(mean_to_marginal_params)(mean_params_b)
        m, S, SS = marg['m'], marg['S'], marg['SS']
        K, T, D = m.shape
        mf = m.reshape(-1, D); Sf = S.reshape(-1, D, D)
        Ef  = jax.vmap(lambda a,b: Ef_diff(a,b,mu_r,grid))(mf,Sf).reshape(K,T,D)
        Edf = jax.vmap(lambda a,b: Edfdx_diff(a,b,mu_r,grid))(mf,Sf).reshape(K,T,D,D)
        if moment == 'exact':
            Eff = jax.vmap(lambda a,b: gmix_E_full_Eff(a-xcen,b,mu_r,grid,aux))(mf,Sf).reshape(K,T)
        else:  # 'sq' = ||E[f]||^2 (linearised value)
            Eff = (Ef*Ef).sum(-1)
        mt,mtn = m[:,:-1], m[:,1:]; St,Stn = S[:,:-1], S[:,1:]; SSt = SS
        Ef_i,Edf_i,Eff_i = Ef[:,:-1], Edf[:,:-1], Eff[:,:-1]
        outer=lambda a,b: a[...,:,None]*b[...,None,:]
        trm  = _tr(Stn + outer(mtn,mtn))
        trm += _tr(St  + outer(mt,mt))
        trm += -2*_tr(SSt + outer(mtn,mt))
        trm += dt[None,:]**2 * Eff_i
        trm += -2*dt[None,:]*_tr(outer(Ef_i,mtn) + jnp.einsum('...ij,...kj->...ik',Edf_i,SSt))
        trm += 2*dt[None,:]*_tr(outer(Ef_i,mt)  + jnp.einsum('...ij,...jk->...ik',Edf_i,St))
        const = -0.5*D*jnp.log(2*jnp.pi*dt[None,:]*sigma**2)
        negCE = const + trm*(-1.0/(2*dt[None,:]*sigma**2))
        tmask = trial_mask[:,:-1] & trial_mask[:,1:]
        total = jnp.where(tmask, negCE, 0.0).sum()
        init = jax.vmap(lambda a,b,c,d: compute_neg_CE_initial(a,b,c,d))(
            m[:,0], S[:,0], init_params['mu0'], init_params['V0'])
        total += jnp.where(trial_mask[:,0], init, 0.0).sum()
        return total
    return total

def nat_grad_batched(mean_params_b, mu_r, grid, t_grid, trial_mask,
                     init_params, sigma, moment='exact'):
    total = make_total_negCE(mu_r, grid, t_grid, trial_mask, init_params, sigma, moment)
    g = jax.grad(total)(mean_params_b)
    symm = lambda A: 0.5*(A + jnp.swapaxes(A,-1,-2))
    return {'J': symm(g['ExxT']), 'h': g['Ex'], 'L': g['ExxnT']}

# ---- memory-bounded chunked grad (keeps direct sums; K=1) -----------------
def nat_grad_batched_chunked(mean_params_b, mu_r, grid, t_grid, trial_mask,
                             init_params, sigma, chunk=2000, moment='exact'):
    """Same exact gradient as nat_grad_batched but peak memory ~O(chunk*M)
    instead of O(N*M): process transitions in chunks, grad each chunk's
    partial total w.r.t. only its source slice, scatter-add. K=1."""
    import jax.numpy as jnp
    from sing.utils.sing_helpers import mean_to_marginal_params, compute_neg_CE_initial
    from sing.exp_full_moments import Ef_diff, Edfdx_diff
    from sing.efgp_gmix_qx_moments import gmix_E_full_Eff, precompute_aux
    xcen=grid.xcen; aux=precompute_aux(mu_r,grid); dt_all=t_grid[1:]-t_grid[:-1]
    K,T,D=mean_params_b['Ex'].shape
    assert K==1
    Ex=mean_params_b['Ex'][0]; ExxT=mean_params_b['ExxT'][0]; ExxnT=mean_params_b['ExxnT'][0]
    def _tr(A): return jnp.trace(A,axis1=-2,axis2=-1)
    def slice_total(Ex_s, ExxT_s, ExxnT_s, dt_s, a):
        mp={'Ex':Ex_s,'ExxT':ExxT_s,'ExxnT':ExxnT_s}
        marg=mean_to_marginal_params(mp); m,S,SS=marg['m'],marg['S'],marg['SS']
        Ef=jax.vmap(lambda x,y: Ef_diff(x,y,mu_r,grid))(m,S)
        Edf=jax.vmap(lambda x,y: Edfdx_diff(x,y,mu_r,grid))(m,S)
        if moment=='exact':
            Eff=jax.vmap(lambda x,y: gmix_E_full_Eff(x-xcen,y,mu_r,grid,aux))(m,S)
        else:
            Eff=(Ef*Ef).sum(-1)
        mt,mtn=m[:-1],m[1:]; St,Stn=S[:-1],S[1:]
        Ef_i,Edf_i,Eff_i=Ef[:-1],Edf[:-1],Eff[:-1]
        outer=lambda u,v: u[...,:,None]*v[...,None,:]
        trm=_tr(Stn+outer(mtn,mtn))+_tr(St+outer(mt,mt))-2*_tr(SS+outer(mtn,mt))
        trm=trm+dt_s**2*Eff_i
        trm=trm-2*dt_s*_tr(outer(Ef_i,mtn)+jnp.einsum('...ij,...kj->...ik',Edf_i,SS))
        trm=trm+2*dt_s*_tr(outer(Ef_i,mt)+jnp.einsum('...ij,...jk->...ik',Edf_i,St))
        const=-0.5*D*jnp.log(2*jnp.pi*dt_s*sigma**2)
        tot=(const+trm*(-1.0/(2*dt_s*sigma**2))).sum()
        tot=tot+jax.lax.cond(a==0,
            lambda: compute_neg_CE_initial(m[0],S[0],init_params['mu0'][0],init_params['V0'][0]),
            lambda: 0.0)
        return tot
    gExxT=jnp.zeros_like(ExxT); gEx=jnp.zeros_like(Ex); gExxnT=jnp.zeros_like(ExxnT)
    gfun=jax.jit(jax.grad(slice_total,argnums=(0,1,2)),static_argnums=(4,))
    a=0
    while a < T-1:
        b=min(a+chunk, T-1)                       # transitions [a,b)
        sl=slice(a,b+1)                            # sources [a,b]
        gx,gxx,gxxn=gfun(Ex[sl],ExxT[sl],ExxnT[a:b],dt_all[a:b],a)
        gEx=gEx.at[sl].add(gx); gExxT=gExxT.at[sl].add(gxx); gExxnT=gExxnT.at[a:b].add(gxxn)
        a=b
    symm=lambda A: 0.5*(A+jnp.swapaxes(A,-1,-2))
    return {'J':symm(gExxT)[None],'h':gEx[None],'L':gExxnT[None]}
