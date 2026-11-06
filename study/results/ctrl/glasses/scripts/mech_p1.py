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
    T = T @ tf(0, 0.1034, 0, -np.pi/4)
    return T
def summ(cand):
    out = {}
    for e in episodes(cand=cand):
        o = obs(e['eid'])
        J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']
        hz = []; tz = []
        for k in range(len(G)):
            if 0.6 <= G[k] <= 0.95:
                hz.append(fk(J[k])[2, 3])
                tz.append(fk(A[k, 14, :7])[2, 3] - fk(J[k])[2, 3])  # planned rise within chunk
        out[e['slot']] = dict(ok=bool(e['success']), nhold=len(hz), maxz=max(hz) if hz else None,
                              rise=float(np.mean(tz)) if tz else None, steps=e['steps'])
    return out
b = summ('base'); p = summ('p1')
rb = [v['rise'] for v in b.values() if v['rise'] is not None]
rp = [v['rise'] for v in p.values() if v['rise'] is not None]
print('planned chunk rise while holding (m): base mean %.3f (n=%d eps)  p1 mean %.3f (n=%d eps)' % (np.mean(rb), len(rb), np.mean(rp), len(rp)))
print('hold replans total: base %d  p1 %d' % (sum(v['nhold'] for v in b.values()), sum(v['nhold'] for v in p.values())))
for s in sorted(b):
    x, y = b[s], p[s]
    print(s, 'base', 'OK ' if x['ok'] else 'FAIL', 'nhold=%2d rise=%s' % (x['nhold'], 'NA' if x['rise'] is None else '%+.3f' % x['rise']),
          '| p1', 'OK ' if y['ok'] else 'FAIL', 'nhold=%2d rise=%s' % (y['nhold'], 'NA' if y['rise'] is None else '%+.3f' % y['rise']))
