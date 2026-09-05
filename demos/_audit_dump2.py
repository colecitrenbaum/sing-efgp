import numpy as np, glob, os
# 1) full-precision gather vs direct keep-all at T=10000
a=np.load('demos/_bench_duffing_keepall_out/cell_T10000_efgp_keepall_seed0.npz',allow_pickle=True)
b=np.load('demos/_bench_duffing_keepall_out/cell_T10000_efgp_keepall_gather_gN512r64_seed0.npz',allow_pickle=True)
for k in ['ls_final','var_final','drift_nrmse','lat_pc','wall']:
    print(f"{k:14s} direct={a[k].item()!r:24s} gather={b[k].item()!r}")
for k in ['f_pred_states_pc','ls_traj','var_traj','m_inf']:
    x,y=a[k],b[k]
    print(f"{k:18s} shape={x.shape} max|diff|={np.abs(x-y).max():.3e}  bitwise_equal={np.array_equal(x,y)}")
print()
# 2) sp100 T=100000 failure
z=np.load('demos/_bench_duffing_scaling_out/cell_T100000_sp100_seed0.npz',allow_pickle=True)
print("sp100 T=100000 err:\n", str(z['err'].item())[:1500])
