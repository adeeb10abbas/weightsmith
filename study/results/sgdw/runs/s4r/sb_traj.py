# owsp py base: where does the hand go under each wording? Panda FK of the measured joints at each replan.
import json


def panda_fk(q):
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        th = float(q[i])
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


rows = episodes()
out = {}
for lab in sorted({r["cand"] for r in rows}):
    eps = [r for r in rows if r["cand"] == lab]
    P = []
    first_close = []
    for r in eps:
        o = obs(r["eid"])
        p = np.array([panda_fk(q) for q in o["joint"][:12]])
        P.append(np.pad(p, ((0, 12 - len(p)), (0, 0)), mode="edge"))
        closed = np.flatnonzero((o["actions"][:, :, 7] > 0.5).mean(1) >= 0.5)
        first_close.append(int(closed[0]) if len(closed) else -1)
    P = np.array(P)
    out[lab] = {"n": len(eps), "prompt": eps[0]["prompt"], "success": int(sum(r["success"] for r in eps)),
                "ee_mean_cm_by_k": [[round(float(v) * 100, 1) for v in P[:, k].mean(0)] for k in range(12)],
                "first_close_k_median": float(np.median([f for f in first_close if f >= 0])) if any(f >= 0 for f in first_close) else None,
                "never_close": int(sum(f < 0 for f in first_close))}
    print(lab, out[lab]["success"], "/", out[lab]["n"], "first gripper-close replan (median)", out[lab]["first_close_k_median"],
          "never closed", out[lab]["never_close"])
    for k in (0, 2, 4, 6, 8):
        print("   k=%d  hand x,y,z (cm, robot base) %s" % (k, out[lab]["ee_mean_cm_by_k"][k]))
print("TRAJ_JSON " + json.dumps(out))
