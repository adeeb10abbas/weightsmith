import json
def fk(q):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107 + 0.1034]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]], [st*ca, ct*ca, -sa, -sa*d[i]], [st*sa, ct*sa, ca, ca*d[i]], [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

dec = []
for e in episodes(cand="base"):
    o = obs(e["eid"])
    G = o["gripper"][:, 0]; J = o["joint"]; A = o["actions"]
    ks = [k for k in range(len(G)) if G[k] > 0.1]
    if not ks:
        continue
    k = ks[0]
    for kk in [k, k+1]:
        if kk >= len(G) or G[kk] <= 0.1 or G[kk] > 0.95:
            continue
        p0 = fk(J[kk]); p1 = fk(A[kk, -1, :7]); pm = fk(A[kk, 7, :7])
        dec.append(dict(eid=e["eid"], k=int(kk), step=int(o["step"][kk]), g=float(G[kk]), dy=float(p1[1]-p0[1]), dz=float(p1[2]-p0[2]), dymid=float(pm[1]-p0[1]), first=(kk == k), success=e["success"]))
for d in dec:
    print(d["eid"][-6:], d["k"], d["step"], "g%.2f dy%+.3f dymid%+.3f dz%+.3f" % (d["g"], d["dy"], d["dymid"], d["dz"]), "first" if d["first"] else "", "S" if d["success"] else "")
json.dump(dec, open("/work/notes/decisions.json", "w"))
# also approach decisions dy at step 15 for reference
