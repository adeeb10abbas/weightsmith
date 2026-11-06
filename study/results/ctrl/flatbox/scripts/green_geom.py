import numpy as np, json
eps = episodes(cand='base')
res = {}
for e in eps:
    eid = e['eid']
    o = obs(eid)
    W = o['wrist_img'].astype(np.int32); B = o['base_img'].astype(np.int32)
    J = o['joint']
    rows = []
    for i in range(len(W)):
        im = W[i]
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        m = (g > r + 15) & (g > b + 40)
        ys, xs = np.nonzero(m)
        if len(xs) > 50:
            cx, cy = xs.mean(), ys.mean()
            C = np.cov(np.stack([xs - cx, ys - cy]))
            w, v = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            rows.append((i, len(xs), round(cx), round(cy), xs.min(), xs.max(), ys.min(), ys.max(), round(float(ang)), round(float(np.sqrt(w[1] / max(w[0], 1e-6))), 2)))
        else:
            rows.append((i, len(xs)))
    res[eid] = rows
    q6 = J[:, 6]
    print(eid, 'q6 start %.2f min %.2f max %.2f' % (q6[0], q6.min(), q6.max()))
    for r in rows[0:30:3]:
        print('   ', r)
json.dump({k: [list(map(lambda x: int(x) if not isinstance(x, float) else x, r)) for r in v] for k, v in res.items()}, open('/work/notes/green_geom.json', 'w'))
