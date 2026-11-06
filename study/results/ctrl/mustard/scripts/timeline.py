import json
eps = episodes(cand="base")
print(len(eps), eps[0].keys() if eps else None)
e0 = eps[0]
o = obs(e0["eid"])
print(type(o), (o.keys() if hasattr(o, "keys") else len(o)))
for k in o:
    v = o[k]
    try:
        print(k, np.asarray(v).shape, np.asarray(v).dtype)
    except Exception as ex:
        print(k, type(v))

def fk(q):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107 + 0.1034]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -sa*d[i]],
                      [st*sa, ct*sa, ca, ca*d[i]],
                      [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

out = {}
for e in eps:
    o = obs(e["eid"])
    steps = np.asarray(o["step"])
    J = np.asarray(o["joint"]); G = np.asarray(o["gripper"]); A = np.asarray(o["actions"])
    rows = []
    for i in range(len(steps)):
        p = fk(J[i])
        rows.append([int(steps[i]), round(float(np.ravel(G[i])[0]), 3), round(float(A[i, :, 7].min()), 2), round(float(A[i, :, 7].max()), 2),
                     round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)])
    out[e["eid"]] = {"success": e["success"], "rows": rows}
json.dump(out, open("/work/notes/timeline.json", "w"))
for eid in ["base.base.s70000", "base.base.s70015", "base.base.s70001", "base.base.s70004"]:
    print(eid, out[eid]["success"])
    for r in out[eid]["rows"]:
        print("  ", r)
