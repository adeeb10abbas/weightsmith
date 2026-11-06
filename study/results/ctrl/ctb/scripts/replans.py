import json
import numpy as np

def fk(q):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
    T = np.eye(4)
    for i in range(7):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(q[i]), np.sin(q[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -d[i]*sa],
                      [st*sa, ct*sa, ca, d[i]*ca],
                      [0, 0, 0, 1]])
        T = T @ A
    p = T @ np.array([0, 0, 0.257, 1.0])
    return p[:3]

eps = episodes(cand="base")
out = {}
for e in eps:
    eid = e["eid"]
    o = obs(eid)
    n = len(o["step"])
    rows = []
    for k in range(n):
        q = np.asarray(o["joint"][k], dtype=np.float64)
        p = fk(q[:7])
        act = np.asarray(o["actions"][k])
        g = act[:, 7]
        rows.append(dict(k=k, step=int(o["step"][k]), gobs=float(np.asarray(o["gripper"][k]).ravel()[0]),
                         amin=float(g.min()), amean=float(g.mean()), amax=float(g.max()),
                         a0=float(g[0]), aend=float(g[-1]),
                         x=float(p[0]), y=float(p[1]), z=float(p[2])))
    out[eid] = dict(success=e["success"], rows=rows)

json.dump(out, open("/work/notes/replans.json", "w"))
print("keys", list(o.keys()), "shapes", {k: np.asarray(v).shape for k, v in o.items() if k in ("joint", "gripper", "actions", "base_img")})
for eid in sorted(out):
    r = out[eid]
    s = "S" if r["success"] else "F"
    line = []
    for row in r["rows"][:12]:
        line.append(f'{row["k"]}:{row["gobs"]:.2f}/{row["amean"]:.2f}/z{row["z"]:.2f}')
    print(eid[-6:], s, " ".join(line))
