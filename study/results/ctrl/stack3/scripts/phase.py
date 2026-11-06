import numpy as np, json
A_ = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
D_ = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
AL = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
def _T(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a],[st*ca, ct*ca, -sa, -d*sa],[st*sa, ct*sa, ca, d*ca],[0,0,0,1]])
def fk(q):
    T = np.eye(4)
    for i in range(7):
        T = T @ _T(A_[i], D_[i], AL[i], q[i])
    T = T @ _T(0, 0.107 + 0.1034, 0, 0.0)
    return T[:3, 3]

eps = sorted(episodes(cand="base"), key=lambda e: e["eid"])
o = obs(eps[0]["eid"])
print("actions[0] first rows", np.round(o["actions"][0][:3], 3))
print("joint[0], joint[1]", np.round(o["joint"][0], 3), np.round(o["joint"][1], 3))
print("gripper first 24", np.round(o["gripper"][:, 0], 2))
out = {}
for e in eps:
    o = obs(e["eid"])
    J = o["joint"]; G = o["gripper"][:, 0]; S = o["step"]; Ac = o["actions"]
    P = np.array([fk(q) for q in J])
    gcmd = (Ac[:, :, 7] > 0.5).mean(1)
    rows = []
    for k in range(len(S)):
        rows.append(f"{S[k]}:{P[k,0]:.2f},{P[k,1]:.2f},{P[k,2]:.2f}|g{G[k]:.2f}|c{gcmd[k]:.1f}")
    out[e["eid"]] = dict(step=S.tolist(), P=P.round(4).tolist(), G=G.round(3).tolist(), gcmd=gcmd.round(2).tolist())
    print(e["eid"][-6:], "OK" if e["success"] else "FAIL")
    print("   " + " ".join(rows))
json.dump(out, open("/work/notes/phase.json", "w"))
