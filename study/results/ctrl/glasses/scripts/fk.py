import json, sys
# Franka Panda modified DH (Craig)
DH = [  # a, d, alpha
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi/2),
    (0.0, 0.316, np.pi/2),
    (0.0825, 0.0, np.pi/2),
    (-0.0825, 0.384, -np.pi/2),
    (0.0, 0.0, np.pi/2),
    (0.088, 0.0, np.pi/2),
]
def tf(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a],
                     [st*ca, ct*ca, -sa, -d*sa],
                     [st*sa, ct*sa, ca, d*ca],
                     [0, 0, 0, 1]])
def fk(q):
    T = np.eye(4)
    for i, (a, d, al) in enumerate(DH):
        T = T @ tf(a, d, al, q[i])
    T = T @ tf(0, 0.107, 0, 0)
    T = T @ tf(0, 0.1034, 0, -np.pi/4)
    return T

def ee(q):
    T = fk(q)
    return T[:3, 3], T[:3, :3]

eps = episodes(cand='base')
out = {}
for e in eps:
    eid = e['eid'] if 'eid' in e else e.get('id')
    o = obs(eid)
    J = o['joint']; G = o['gripper'][:, 0]; S = o['step']; A = o['actions']
    rows = []
    for k in range(len(S)):
        p, R = ee(J[k])
        rows.append([int(S[k]), round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                     round(float(G[k]), 3), round(float(A[k, :, 7].max()), 2), round(float(A[k, :, 7].min()), 2),
                     round(float(R[2, 2]), 2)])
    out[eid] = dict(success=e.get('success'), rows=rows)
json.dump(out, open('/work/notes/fk_base.json', 'w'))
for eid, v in out.items():
    print(eid, v['success'])
    for r in v['rows']:
        print('  ', r)
