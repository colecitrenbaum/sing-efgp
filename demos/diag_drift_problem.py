"""
diag_drift_problem.py -- visualize WHY drift rel-MSE looked broken.

Problem: the GP-drift SDE (canonical seeds) pushes the particle to the domain
edge; jnp.clip(xs,-3,3) then PINS it to the wall (~96-98% of steps). At the
wall the particle's velocity is ~0 while the true GP drift EXTRAPOLATES to large
values just past the sampling grid -> f_true blows up, the trajectory never
explores a 2D region, and any drift metric is dominated by this degenerate wall
state. The inference is actually fine: EFGP/SparseGP drift correlates ~0.8 with
the empirical pseudo-velocity at INTERIOR states (near the diffusion noise floor).

Produces a 2x2 diagnostic from a saved cell npz (needs f_true_states /
f_pred_states_raw / xs_true / dt).

  python demos/diag_drift_problem.py <cell.npz> [out.png]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else \
        'demos/_bench_gpdrift_scaling_smoke/cell_T1000_efgp_seed0.npz'
    out = sys.argv[2] if len(sys.argv) > 2 else \
        'demos/_bench_gpdrift_scaling_smoke/diag_drift_problem.png'
    z = np.load(npz, allow_pickle=True)
    xs = z['xs_true']; dt = float(z['dt'])
    ft = z['f_true_states']; fp = z['f_pred_states_raw']
    T = xs.shape[0]
    v_full = (xs[1:] - xs[:-1]) / dt               # empirical E[v|x]=f(x), (T-1,D)
    step = max(1, T // ft.shape[0])
    n = min(ft.shape[0], v_full[::step].shape[0])
    pts = xs[::step][:n]
    ft = ft[:n]; fp = fp[:n]
    v = v_full[::step][:n]

    at_wall = (np.abs(pts) >= 2.99).any(1)
    interior = ~at_wall
    frac_wall = at_wall.mean()

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # (a) trajectory in state space
    a = ax[0, 0]
    sc = a.scatter(xs[:, 0], xs[:, 1], c=np.arange(T), s=3, cmap='viridis')
    a.axvline(3, ls='--', c='r', lw=1); a.axvline(-3, ls='--', c='r', lw=1)
    a.axhline(3, ls='--', c='r', lw=1); a.axhline(-3, ls='--', c='r', lw=1)
    a.set_title(f'(a) trajectory in state space\n{frac_wall*100:.0f}% of steps '
                f'PINNED at clip wall |x|=3', fontsize=10)
    a.set_xlabel('x0'); a.set_ylabel('x1')
    plt.colorbar(sc, ax=a, label='time step')

    # (b) time series with clip
    b = ax[0, 1]
    tt = np.arange(T)
    b.plot(tt, xs[:, 0], lw=0.7, label='x0')
    b.plot(tt, xs[:, 1], lw=0.7, label='x1')
    b.axhline(3, ls='--', c='r', lw=1, label='clip |x|=3')
    b.axhline(-3, ls='--', c='r', lw=1)
    b.set_title('(b) state vs time — hits wall almost immediately', fontsize=10)
    b.set_xlabel('time step'); b.legend(fontsize=8)

    # (c) inferred vs true drift, ALL states (polluted by wall)
    c = ax[1, 0]
    lim = max(np.abs(ft).max(), np.abs(fp).max()) * 1.05
    c.scatter(ft[at_wall, 0], fp[at_wall, 0], s=6, c='r', alpha=0.3,
              label='at wall')
    c.scatter(ft[interior, 0], fp[interior, 0], s=6, c='b', alpha=0.5,
              label='interior')
    c.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    c.set_xlim(-lim, lim); c.set_ylim(-lim, lim)
    c.set_title('(c) inferred vs TRUE drift (x0)\nwall pts have huge f_true, '
                'tiny inferred', fontsize=10)
    c.set_xlabel('f_true(x0)'); c.set_ylabel('f_inferred(x0)'); c.legend(fontsize=8)

    # (d) inferred drift vs EMPIRICAL velocity, interior only (the real test)
    d = ax[1, 1]
    if interior.sum() > 2:
        cc0 = np.corrcoef(v[interior, 0], fp[interior, 0])[0, 1]
        cc1 = np.corrcoef(v[interior, 1], fp[interior, 1])[0, 1]
    else:
        cc0 = cc1 = float('nan')
    vlim = max(np.abs(v[interior]).max(), np.abs(fp[interior]).max()) * 1.05 \
        if interior.any() else 1.0
    d.scatter(v[interior, 0], fp[interior, 0], s=8, c='C0', alpha=0.5,
              label=f'x0 (corr={cc0:.2f})')
    d.scatter(v[interior, 1], fp[interior, 1], s=8, c='C3', alpha=0.5,
              label=f'x1 (corr={cc1:.2f})')
    d.plot([-vlim, vlim], [-vlim, vlim], 'k--', lw=1)
    d.set_xlim(-vlim, vlim); d.set_ylim(-vlim, vlim)
    d.set_title('(d) inferred drift vs EMPIRICAL velocity\n(interior only) — '
                'inference IS good where data is valid', fontsize=10)
    d.set_xlabel('empirical v=(dx)/dt'); d.set_ylabel('f_inferred'); d.legend(fontsize=8)

    fig.suptitle('Drift-recovery diagnostic: trajectory collapses to clip wall',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"  frac at wall = {frac_wall*100:.1f}%   interior pts = {interior.sum()}")
    print(f"  f_true rms all={np.sqrt((ft**2).mean()):.3f}  "
          f"interior={np.sqrt((ft[interior]**2).mean()) if interior.any() else float('nan'):.3f}")
    print(f"  interior corr(v, f_inferred): x0={cc0:.3f}  x1={cc1:.3f}")


if __name__ == '__main__':
    main()
