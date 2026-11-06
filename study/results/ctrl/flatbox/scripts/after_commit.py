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
    return T[:3, 3], T[:3, :3]
for cand in ['base', 'p1']:
    nxt = {'stay_hold': 0, 'slip_to_closed': 0, 'opened': 0}
    zs = []
    for e in episodes(cand=cand):
        o = obs(e['eid'])
        J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']
        for i in range(len(J) - 1):
            if 0.1 < G[i] < 0.45 and A[i, 0, 7] > 0.5 and (A[i, :, 7] > 0.5).mean() > 0.9:
                p, R = fk(J[i]); p2, _ = fk(J[i + 1])
                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                g2 = G[i + 1]
                k = 'stay_hold' if 0.1 < g2 < 0.6 else ('slip_to_closed' if g2 >= 0.6 else 'opened')
                nxt[k] += 1
                zs.append((e['eid'][-6:], i, round(float(p[2]), 3), round(float(yaw)), round(float(G[i]), 2), round(float(p2[2]), 3), round(float(g2), 2), round(float(A[i, -1, 7]), 2)))
    print(cand, nxt)
    for z in zs:
        print('   ', z)
