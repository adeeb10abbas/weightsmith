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

AID = "v1-p1"
idx, arr = acts(AID)
items = idx['items']
A = arr['actions']
grp = ['R'] * 20 + ['C'] * 14 + ['G'] * 8 + ['K0'] * 2
stats = {}
for i, it in enumerate(items):
    o = obs(it['episode']); k = it['k']
    cur = fk(o['joint'][k])
    endb = fk(o['actions'][k][14, :7])  # recorded base action
    endp = fk(A[i, 14, :7])
    gb = o['actions'][k][:, 7].mean(); gp = A[i, :, 7].mean()
    sh = endp - endb
    stats.setdefault(grp[i], []).append(sh)
    print(grp[i], it['episode'][-2:], k, 'cur', cur.round(2), 'base_end', endb.round(2), 'p_end', endp.round(2),
          'shift', sh.round(3), 'g', round(float(gb), 2), '->', round(float(gp), 2))
for g, v in stats.items():
    v = np.array(v)
    print(g, 'mean shift', v.mean(0).round(3), 'mean |shift|', np.linalg.norm(v, axis=1).mean().round(3))
