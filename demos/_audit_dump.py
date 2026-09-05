import numpy as np, glob, os
rows=[]
for d in ['demos/_bench_duffing_scaling_out','demos/_bench_duffing_keepall_out','demos/_bench_duffing_keepall_prof','demos/_bench_duffing_highM_out','demos/_bench_duffing_scaling_newcanon']:
    for f in sorted(glob.glob(d+'/*.npz')):
        z=np.load(f, allow_pickle=True)
        g=lambda k,dflt=0: (z[k].item() if k in z.files else dflt)
        rows.append((os.path.basename(d), os.path.basename(f), str(g('status','?'))[:2], g('T'), str(g('method','?')), g('M'),
                     g('wall',float('nan')), g('drift_nrmse',float('nan')), g('ls_final',float('nan')), g('var_final',float('nan')), g('lat_pc',float('nan')), g('peak_gb',0.0)))
print(f"{'dir':32s} {'file':50s} {'st':2s} {'T':>7s} {'method':20s} {'M':>4s} {'wall':>8s} {'nrmse':>8s} {'ls':>7s} {'var':>7s} {'latpc':>7s} {'peakGB':>7s}")
for r in rows:
    print(f"{r[0]:32s} {r[1]:50s} {r[2]:2s} {r[3]:>7} {r[4]:20s} {r[5]:>4} {r[6]:8.1f} {r[7]:8.4f} {r[8]:7.3f} {r[9]:7.3f} {r[10]:7.4f} {r[11]:7.2f}")
