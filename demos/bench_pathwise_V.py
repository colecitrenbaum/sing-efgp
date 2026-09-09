"""Validate the Matheron/pathwise V-restoration for the keep-all autodiff
E-step: (1) sampler Cov(dw) vs dense A^-1, (2) rho+omega -> dense-exact GT
as S grows, (3) natural gradient -> dense-GT gradient as S grows.
See KEEPALL_AUTODIFF_NOTES.md. Run from the repo root.
"""
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr, numpy as np
import sing.efgp_jax_primitives as jp
import sing.efgp_jax_drift as jpd
from sing.efgp_gmix_qx_moments import _autocorr_per_r, precompute_aux, GmixQxAux, _delta_grid_for_autocorr
from sing.exp_pathwise_v import sample_dw, rho_plus_omega_aux
from sing.exp_batched_estep import make_total_negCE

X=jnp.array([[-1.5,-1.0],[1.5,1.0]]); grid=jp.spectral_grid_se(0.7,1.0,X,eps=1.2e-1)
M=grid.M; D_out=2; ws=grid.ws.real; cdtype=grid.ws.dtype
rng=np.random.default_rng(0); Ns=60; sig2=0.16
ms=jnp.asarray(rng.uniform(-1,1,(Ns,2))); Ss=jnp.broadcast_to(jnp.eye(2)*0.06,(Ns,2,2))
weights=jnp.ones(Ns)*(0.05/sig2)
mu_r,_,top=jpd.compute_mu_r_gmix_jax(ms,Ss,jnp.zeros((Ns,2)),jnp.zeros((Ns,2,2)),
    jnp.ones(Ns)*0.05,grid,sigma_drift_sq=sig2,D_lat=2,D_out=2,fine_N=64,stencil_r=6)
A_apply=jp.make_A_apply(grid.ws,top,sigmasq=1.0)
A=np.asarray(jax.vmap(A_apply)(jnp.eye(M,dtype=cdtype))); Ainv=np.linalg.inv(A)
R=np.linalg.cholesky(Ainv+1e-12*np.eye(M))
rho=precompute_aux(mu_r,grid).rho_summed
omega=_autocorr_per_r(jnp.asarray(R.T),ws,grid.mtot_per_dim).sum(0)*D_out
GT=np.asarray(rho+omega.reshape(-1))
GT_aux=GmixQxAux(rho_summed=jnp.asarray(GT),delta_flat=_delta_grid_for_autocorr(grid))
# (1) sampler Cov
dw=np.asarray(sample_dw(mu_r,grid,top,ms,Ss,weights,jr.PRNGKey(0),8000,cg_tol=1e-8,max_iter=4000))
covw=(dw.conj().T@dw/dw.shape[0]); print(f"(1) Cov(dw) vs A^-1: rel={np.linalg.norm(covw-Ainv)/np.linalg.norm(Ainv):.3f}")
# (2) rho+omega and (3) gradient convergence
K,T=1,30; tg=jnp.linspace(0.,3.,T); tm=jnp.ones((K,T),bool)
ip=jax.tree_util.tree_map(lambda z:jnp.broadcast_to(z,(K,)+z.shape),dict(mu0=jnp.zeros(2),V0=jnp.eye(2)*0.2))
mm=jr.normal(jr.PRNGKey(7),(K,T,2))*0.4;Lc=jr.normal(jr.PRNGKey(8),(K,T,2,2))*0.1
Sx=jnp.einsum('...ij,...kj->...ik',Lc,Lc)+jnp.eye(2)*0.05;ExxT=Sx+jnp.einsum('...i,...j->...ij',mm,mm)
SS=jnp.zeros((K,T-1,2,2))+0.01;ExxnT=SS+jnp.einsum('...i,...j->...ij',mm[:,1:],mm[:,:-1]);mp={'Ex':mm,'ExxT':ExxT,'ExxnT':ExxnT}
gGT=jax.grad(make_total_negCE(mu_r,grid,tg,tm,ip,jnp.sqrt(sig2),'exact',quad_aux=GT_aux))(mp)
print("(2) rho+omega rel-err   (3) gradient rel-err   vs dense GT:")
for S in [1,4,16,64]:
    dw=sample_dw(mu_r,grid,top,ms,Ss,weights,jr.PRNGKey(S),S,cg_tol=1e-7,max_iter=3000)
    aux=rho_plus_omega_aux(mu_r,dw,grid)
    r2=float(np.linalg.norm(np.asarray(aux.rho_summed)-GT)/np.linalg.norm(GT))
    g=jax.grad(make_total_negCE(mu_r,grid,tg,tm,ip,jnp.sqrt(sig2),'exact',quad_aux=aux))(mp)
    num=sum(float(np.linalg.norm(np.asarray(g[k])-np.asarray(gGT[k])))**2 for k in g)**.5
    den=sum(float(np.linalg.norm(np.asarray(gGT[k])))**2 for k in g)**.5
    print(f"  S={S:3d}:  rho+omega {r2:.4f}    grad {num/den:.4f}")
