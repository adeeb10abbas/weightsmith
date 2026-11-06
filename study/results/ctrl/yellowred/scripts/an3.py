import json

def fk(q, tip=0.257):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, tip]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -d[i]*sa],
                      [st*sa, ct*sa, ca, d[i]*ca],
                      [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

idx, arr = acts("a2-phases")
items = idx['items']
N = len(items)
Yi = list(range(0, 21)); Ri = list(range(21, 33)); Ci = list(range(33, N))
A = arr['actions']
print('parity', idx.get('base_parity_max_abs_action_error'))
for i, it in enumerate(items):
    o = obs(it['episode'])
    k = it['k']
    cur = fk(o['joint'][k]); end = fk(A[i, 14, :7])
    grp = 'Y' if i in Yi else ('R' if i in Ri else 'C')
    print(grp, it['episode'][-2:], k, 'cur', cur.round(2), 'end', end.round(2), 'g', round(float(o['gripper'][k, 0]), 2), 'gcmd', A[i, :, 7].mean().round(2))

R = arr['action.resid_out'][:, :, :, 0, :]  # N,S,La,1024
H = arr['action.mlp_hidden'][:, :, :, 0, :]  # N,S,La,4096
La = idx['action_layers']
S = R.shape[1]
out = {}
for s in range(S):
    print('--- denoise idx', s, 't', idx['denoise_t'][s])
    for li, L in enumerate(La):
        r = R[:, s, li]
        mY, mR, mC = r[Yi].mean(0), r[Ri].mean(0), r[Ci].mean(0)
        d = mY - mR
        dn = d / np.linalg.norm(d)
        pY, pR, pC = r[Yi] @ dn, r[Ri] @ dn, r[Ci] @ dn
        sd = np.sqrt(0.5 * (pY.var() + pR.var()))
        print(f"L{L:2d} |d|={np.linalg.norm(d):.2f} |r|={np.linalg.norm(r, axis=1).mean():.2f} dprime={(pY.mean()-pR.mean())/sd:.2f} "
              f"projY={pY.mean():.2f} projR={pR.mean():.2f} projC={pC.mean():.2f}  minY={pY.min():.2f} maxR={pR.max():.2f}")
# gate analysis on hidden at s=0
s = 0
for li, L in enumerate(La):
    h = H[:, s, li]
    app = h[Yi + Ri]; car = h[Ci]
    ma, mc = app.mean(0), car.mean(0)
    # units: consistently positive in approach, ~0 in carry
    score = (ma - np.abs(mc)) / (app.std(0) + 1e-3)
    top = np.argsort(score)[::-1][:3]
    print(f"L{L:2d} gate units", [(int(u), round(float(ma[u]), 3), round(float(mc[u]), 3), round(float(app.min(0)[u]), 3), round(float(np.abs(car).max(0)[u]), 3)) for u in top])
