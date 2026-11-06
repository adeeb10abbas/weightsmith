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
summ = {}
for cand in ["base", "p1"]:
    nl = nr = 0; ymins = []
    for e in sorted(episodes(cand=cand), key=lambda e: e["slot"]):
        o = obs(e["eid"]); G = o["gripper"][:, 0]; J = o["joint"]; st = o["step"]
        ys = np.array([fk(j)[1] for j in J]); zs = np.array([fk(j)[2] for j in J])
        ks = [k for k in range(len(G)) if G[k] > 0.1]
        if not ks:
            continue
        k0 = ks[0]
        seg = range(k0, min(k0 + 5, len(G)))
        ymin = min(ys[k] for k in seg); ymax = max(ys[k] for k in seg)
        went_right = ymin < -0.08
        nr += went_right; nl += (not went_right) and ymax > 0.08
        if cand == "p1":
            print(e["slot"], "S" if e["success"] else "-", " ".join("[%d g%.2f y%+.2f z%.2f]" % (st[k], G[k], ys[k], zs[k]) for k in range(max(0, k0 - 1), min(k0 + 6, len(G)))))
    summ[cand] = (nl, nr)
print("first-5-replans-after-grasp: (reached left y>0.08 w/o going right, went right y<-0.08)", summ)
