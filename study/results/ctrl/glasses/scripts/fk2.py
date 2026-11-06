import json
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi/2), (0.0, 0.316, np.pi/2), (0.0825, 0.0, np.pi/2),
      (-0.0825, 0.384, -np.pi/2), (0.0, 0.0, np.pi/2), (0.088, 0.0, np.pi/2)]
def tf(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])
def fk(q):
    T = np.eye(4)
    for i, (a, d, al) in enumerate(DH):
        T = T @ tf(a, d, al, q[i])
    T = T @ tf(0, 0.107, 0, 0)
    T = T @ tf(0, 0.1034 + 0.0, 0, -np.pi/4)
    return T
import sys
cand = 'base'
eps = episodes(cand=cand)
out = {}
for e in eps:
    eid = e['eid']
    o = obs(eid)
    J = o['joint']; G = o['gripper'][:, 0]; S = o['step']; A = o['actions']
    rows = []
    for k in range(len(S)):
        p = fk(J[k])[:3, 3]
        rows.append([int(S[k]), round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                     round(float(G[k]), 2), round(float(A[k, 0, 7]), 2), round(float(A[k, :, 7].mean()), 2)])
    out[eid] = dict(success=e.get('success'), rows=rows)
json.dump(out, open('/work/notes/fk_%s.json' % cand, 'w'))
print('keys', list(eps[0].keys()))
print('obs keys', {k: np.asarray(v).shape for k, v in obs(eps[0]['eid']).items()})
for eid, v in sorted(out.items()):
    print(eid, 'OK' if v['success'] else 'FAIL')
    # compact: step:z:g:a0
    print('   ' + ' '.join('%d:%.2f/%.2f/%.1f' % (r[0], r[3], r[4], r[5]) for r in v['rows']))
