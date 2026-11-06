import numpy as np, json
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

rows = []
for e in episodes(cand='base'):
    eid = e['eid']
    o = obs(eid)
    J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']
    for i in range(len(J)):
        p, R = fk(J[i])
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        g7 = A[i, :, 7]
        prevg = A[i - 1, :, 7] if i > 0 else np.zeros(15)
        rows.append(dict(eid=eid, k=i, step=int(o['step'][i]), z=float(p[2]), yaw=yaw, gobs=float(G[i]),
                         g0=float(g7[0]), gmean=float(g7.mean()), gfrac=float((g7 > 0.5).mean()), glast=float(g7[-1]),
                         prev_last=float(prevg[-1]), prev_frac=float((prevg > 0.5).mean()),
                         first_close=int(np.argmax(g7 > 0.5)) if (g7 > 0.5).any() else -1))
json.dump(rows, open('/work/notes/base_rows.json', 'w'))
rows = np.array([(r['gobs'], r['g0'], r['gfrac'], r['z'], abs(r['yaw'])) for r in rows])
bins = [(-0.01, 0.02), (0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.7), (0.7, 0.95), (0.95, 1.1)]
print('gobs bin: n, mean g0, frac(g0>0.5), mean gfrac')
for lo, hi in bins:
    m = (rows[:, 0] > lo) & (rows[:, 0] <= hi)
    if m.sum():
        print(f"  ({lo:.2f},{hi:.2f}] n={m.sum():3d} g0={rows[m,1].mean():.2f} commit={(rows[m,1]>0.5).mean():.2f} gfrac={rows[m,2].mean():.2f} z={rows[m,3].mean():.3f}")
# rotated & low
m = (rows[:, 4] > 60) & (rows[:, 3] < 0.04)
print('rotated & low n', m.sum(), 'mean gfrac', rows[m, 2].mean(), 'g0>0.5', (rows[m, 1] > 0.5).mean())
m = (rows[:, 4] < 20) & (rows[:, 3] < 0.05) & (rows[:, 0] < 0.05)
print('unrotated & low & open n', m.sum(), 'mean gfrac', rows[m, 2].mean(), 'g0>0.5', (rows[m, 1] > 0.5).mean())
