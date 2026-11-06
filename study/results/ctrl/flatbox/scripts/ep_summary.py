import json
import numpy as np

_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
_AL = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
def _tf(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
def fk(q, tip=0.257):
    T = np.eye(4)
    for i in range(7):
        T = T @ _tf(_A[i], _D[i], _AL[i], q[i])
    T = T @ _tf(0, tip, 0, 0)
    return T[:3, 3]

eps = episodes(cand='base')
print('n episodes', len(eps))
print('record keys', list(eps[0].keys()) if isinstance(eps[0], dict) else type(eps[0]))
print(eps[0])
e0 = eps[0]
eid = e0['eid'] if 'eid' in e0 else e0.get('id')
o = obs(eid)
print('obs type', type(o))
if isinstance(o, dict):
    for k, v in o.items():
        print(k, getattr(v, 'shape', type(v)), getattr(v, 'dtype', ''))
out = {}
for e in eps:
    eid = e['eid'] if 'eid' in e else e.get('id')
    o = obs(eid)
    J = np.asarray(o['joint']); G = np.asarray(o['gripper']).reshape(len(J), -1)[:, 0]
    steps = np.asarray(o['step'])
    acts = np.asarray(o['actions'])
    xyz = np.array([fk(q) for q in J])
    gcmd = (acts[:, :, 7] > 0.5).mean(1)
    out[eid] = dict(success=bool(e.get('success')), steps=steps.tolist(), z=xyz[:, 2].round(3).tolist(),
                    x=xyz[:, 0].round(3).tolist(), y=xyz[:, 1].round(3).tolist(), g=np.round(G, 3).tolist(), gcmd=np.round(gcmd, 2).tolist())
    s = 'S' if e.get('success') else 'F'
    print(f"{eid} {s} nrep={len(J)}")
    line = []
    for i in range(0, len(J), 2):
        line.append(f"{steps[i]}:z{xyz[i,2]:.2f} g{G[i]:.2f} c{gcmd[i]:.1f}")
    print('   ' + ' | '.join(line))
json.dump(out, open('/work/notes/base_traj.json', 'w'))
