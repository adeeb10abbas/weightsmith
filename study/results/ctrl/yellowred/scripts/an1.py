import json

def fk(q, tip=0.257):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, tip]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -d[i]*sa],
                      [st*sa, ct*sa, ca, d[i]*ca],
                      [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

idx, arr = acts("a1-prompts")
if isinstance(idx, dict):
    for k, v in idx.items():
        s = repr(v)
        print('IDX', k, s[:400])
for k, v in arr.items():
    print('ARR', k, getattr(v, 'shape', None))

spec = json.load(open('/work/notes/a1spec.json')) if False else None
items = idx.get('items') if isinstance(idx, dict) else None
A = arr['actions']
N = A.shape[0]
prompts = ["orig", "pickup", "lime", "cube", "yel-green", "next-green", "GREEN", "put-on-top"]
decs = [("base.base.s70004", 1), ("base.base.s70013", 1), ("base.base.s70028", 1), ("base.base.s70030", 1),
        ("base.base.s70004", 10), ("base.base.s70013", 3), ("base.base.s70020", 3), ("base.base.s70028", 5)]
cache = {}
res = {}
for i in range(N):
    p = prompts[i // 8]; e, k = decs[i % 8]
    if e not in cache:
        cache[e] = obs(e)
    o = cache[e]
    cur = fk(o['joint'][k])
    mid = fk(A[i, 7, :7]); end = fk(A[i, 14, :7])
    rec = fk(o['actions'][k][14, :7])
    g = A[i, :, 7].mean()
    res.setdefault(p, []).append((e[-2:], k, cur.round(2).tolist(), end.round(2).tolist(), round(float(g), 2)))
for p in prompts:
    print(p)
    for r in res[p]:
        print('   ', r)

# attention mass summary
if 'action.attn_mass' in arr:
    M = arr['action.attn_mass']  # N,S,La,8,5
    print('attn_mass shape', M.shape)
    m = M.mean(axis=(2, 3))  # N,S,5
    for i in range(N):
        if i % 8 == 0:
            print(prompts[i // 8])
        print('   ', decs[i % 8], 'S0', m[i, 0].round(3), 'S-1', m[i, -1].round(3))
