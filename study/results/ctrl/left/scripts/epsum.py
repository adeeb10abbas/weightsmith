import json
import numpy as np

def dh(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1]])

P = [(0, 0.333, 0), (0, 0, -np.pi/2), (0, 0.316, np.pi/2), (0.0825, 0, np.pi/2),
     (-0.0825, 0.384, -np.pi/2), (0, 0, np.pi/2), (0.088, 0, np.pi/2)]

def fk(q, tip=0.107 + 0.1034):
    T = np.eye(4)
    for (a, d, al), th in zip(P, q[:7]):
        T = T @ dh(a, d, al, th)
    T = T @ dh(0, tip, 0, 0)
    return T[:3, 3]

eps = episodes(cand='base')
print('n', len(eps))
e0 = eps[0]
print({k: (repr(v)[:200]) for k, v in (e0.items() if isinstance(e0, dict) else [])})
out = {}
for e in eps:
    eid = e['eid'] if 'eid' in e else e.get('id')
    o = obs(eid)
    J = np.array(o['joint']); G = np.array(o['gripper']).reshape(len(J), -1)[:, 0]
    st = np.array(o['step'])
    xyz = np.array([fk(q) for q in J])
    out[eid] = dict(success=e.get('success'), step=st.tolist(), xyz=xyz.round(3).tolist(), grip=G.round(3).tolist())
    line = ' '.join(f"{int(s)}:{x[0]:+.2f},{x[1]:+.2f},{x[2]:.2f},g{g:.2f}" for s, x, g in list(zip(st, xyz, G))[::3])
    print(eid, e.get('success'))
    print('   ', line)
json.dump(out, open('/work/notes/base_traj.json', 'w'))
