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

idx, arr = acts('a1-prompts')
idx2, arr2 = acts('a2-frontback')
A = np.concatenate([np.array(arr['actions']), np.array(arr2['actions'])])
items = idx['items'] + [dict(it, i=it['i'] + 72) for it in idx2['items']]
print(idx2['items'][0]['prompt_token_ids_at_slots'][3:16], idx2['items'][12]['prompt_token_ids_at_slots'][3:16])
cache = {}
res = {}
for it in items:
    e = it['episode']
    if e not in cache:
        cache[e] = obs(e)
    o = cache[e]
    J = np.array(o['joint'])[it['k']]
    p0 = fk(J)
    a = A[it['i']]
    d7 = fk(a[7]) - p0
    d14 = fk(a[14]) - p0
    res.setdefault(it['prompt'], []).append((e[-6:], it['k'], d7, d14, a[:, 7].mean()))
for p, rows in res.items():
    D = np.array([r[3] for r in rows])
    print(f"{p:50s} mean d14=({D[:,0].mean():+.3f},{D[:,1].mean():+.3f},{D[:,2].mean():+.3f})")
print()
for j in range(12):
    line = []
    for p, rows in res.items():
        r = rows[j]
        line.append(f"({r[3][0]:+.2f},{r[3][1]:+.2f})")
    print(res[list(res)[0]][j][0], res[list(res)[0]][j][1], ' '.join(line))
