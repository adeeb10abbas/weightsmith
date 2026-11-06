import numpy as np, json
exec(open('/work/notes/fk_src.py').read()) if False else None
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
    return T[:3, 3], T[:3, :3]
for cand in ['base', 'p1']:
    eps = episodes(cand=cand)
    tot = dict(hold=0, commit=0, rot=0, lift_hold=0, rot_eps=0, maxz_hold=[], first_rot=[])
    for e in eps:
        o = obs(e['eid'])
        J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']
        rot_ep = False
        for i in range(len(J)):
            p, R = fk(J[i])
            yaw = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            if yaw > 60 and not rot_ep:
                rot_ep = True; tot['first_rot'].append(int(o['step'][i]))
            if 0.1 < G[i] < 0.45:
                tot['hold'] += 1
                tot['commit'] += A[i, 0, 7] > 0.5
                tot['maxz_hold'].append(p[2])
                if p[2] > 0.06:
                    tot['lift_hold'] += 1
        tot['rot_eps'] += rot_ep
    mz = np.array(tot['maxz_hold'])
    print(cand, 'n eps', len(eps), 'hold replans', tot['hold'], 'commit', tot['commit'], 'hold with z>0.06', tot['lift_hold'],
          'z>0.08', int((mz > 0.08).sum()), 'rotated eps', tot['rot_eps'], 'median first rot step', np.median(tot['first_rot']) if tot['first_rot'] else None)
