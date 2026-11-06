import json, sys

def fk(q):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
    T = np.eye(4)
    for i in range(7):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(q[i]), np.sin(q[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -sa*d[i]],
                      [st*sa, ct*sa, ca, ca*d[i]],
                      [0, 0, 0, 1]])
        T = T @ A
    tip = T @ np.array([0, 0, 0.257, 1.0])
    return tip[:3]

eps = episodes(cand='base')
print('n episodes', len(eps))
e = eps[0]
print({k: (v if not isinstance(v, (list, dict)) else str(v)[:150]) for k, v in e.items()})
o = obs(e['eid'])
print({k: (np.shape(v)) for k, v in o.items()})
out = {}
for e in eps:
    eid = e['eid']
    o = obs(eid)
    J = np.array(o['joint'])
    G = np.array(o['gripper']).reshape(len(J), -1)[:, 0]
    steps = np.array(o['step'])
    A = np.array(o['actions'])
    xyz = np.array([fk(j) for j in J])
    rows = []
    for i in range(len(steps)):
        rows.append([int(steps[i]), round(float(G[i]), 3), [round(float(x), 3) for x in xyz[i]],
                     round(float(A[i][:, 7].mean()), 2)])
    out[eid] = rows
json.dump(out, open('/work/notes/base_traj.json', 'w'))
for eid in sorted(out):
    rows = out[eid][:12]
    s = ' '.join('%d:%s%s' % (r[0], ''.join('%+.2f' % x for x in r[2][:3]), 'C' if r[3] > 0.5 else '') for r in rows)
    print(eid[-6:], 'OK ' if [x for x in eps if x['eid'] == eid][0].get('success') else 'F  ', s)
