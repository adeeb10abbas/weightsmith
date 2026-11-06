import json
def dh(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
P = [(0, 0.333, 0), (0, 0, -np.pi/2), (0, 0.316, np.pi/2), (0.0825, 0, np.pi/2),
     (-0.0825, 0.384, -np.pi/2), (0, 0, np.pi/2), (0.088, 0, np.pi/2)]
def fk(q, tip=0.2104):
    T = np.eye(4)
    for (a, d, al), th in zip(P, q[:7]):
        T = T @ dh(a, d, al, th)
    T = T @ dh(0, tip, 0, 0)
    return T[:3, 3]

rows = []
for e in episodes(cand='base'):
    eid = e['eid']; o = obs(eid)
    J = np.array(o['joint']); G = np.array(o['gripper'])[:, 0]; A = np.array(o['actions']); S = np.array(o['step'])
    for i in range(len(S)):
        if 0.2 < G[i] < 0.5:
            p0 = fk(J[i]); p7 = fk(A[i, 7]); p14 = fk(A[i, 14])
            rows.append(dict(eid=eid, k=i, step=int(S[i]), succ=bool(e['success']), g=float(G[i]), cmdg=float(A[i, :, 7].mean()),
                             p0=p0.round(3).tolist(), d14=(p14 - p0).round(3).tolist()))
json.dump(rows, open('/work/notes/carry_rows.json', 'w'))
cur = None
for r in rows:
    if r['eid'] != cur:
        cur = r['eid']; print(cur, 'S' if r['succ'] else 'F')
    p, d = r['p0'], r['d14']
    print(f"   k{r['k']:2d} s{r['step']:3d} p=({p[0]:+.2f},{p[1]:+.2f},{p[2]:.2f}) d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) g={r['g']:.2f} cmdg={r['cmdg']:.2f}")
