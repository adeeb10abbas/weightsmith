# owsp py base: where do the two wordings disagree on the same recorded input?
# Two action-only captures replay the same decisions (items) under RF (the task's wording) and TF. For every item
# this prints the chunk-end end-effector distance between the two replays (Panda forward kinematics on the last
# joint target), the lateral offset (robot y; y is left, so TF - RF < 0 means TF goes further right), and the
# gripper decision under each wording, then ranks the items. Edit CAP_RF / CAP_TF / TOP / PER_EP below.
CAP_RF, CAP_TF = "div-rf", "div-tf"
TOP, PER_EP = 8, 2


def gripper_closed(actions, threshold=0.5):
    a = np.asarray(actions)
    return (a[..., 7] > threshold).mean(-1) >= 0.5


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



idx, x = acts("div2")
items = idx["items"]
A = x["actions"]
rows = []
for i in range(0, len(items), 2):
    r, t = items[i], items[i + 1]
    assert (r["episode"], r["k"]) == (t["episode"], t["k"]) and r["prompt"] != t["prompt"]
    ea, eb = panda_fk(A[i][-1, :7]), panda_fk(A[i + 1][-1, :7])
    rows.append({"episode": r["episode"], "k": r["k"], "grip_state": round(r["gripper_state"], 3),
                 "ee_dist_cm": round(float(np.linalg.norm(ea - eb)) * 100, 1), "dy_tf_minus_rf_cm": round(float(eb[1] - ea[1]) * 100, 1),
                 "dz_tf_minus_rf_cm": round(float(eb[2] - ea[2]) * 100, 1),
                 "closed_rf": bool(gripper_closed(A[i])), "closed_tf": bool(gripper_closed(A[i + 1]))})
for e in sorted({r["episode"] for r in rows}):
    rr = [r for r in rows if r["episode"] == e]
    print(e, " ".join(f"k{r['k']}:g{r['grip_state']:.2f}:{r['ee_dist_cm']:.0f}/{r['dy_tf_minus_rf_cm']:+.0f}/{r['dz_tf_minus_rf_cm']:+.0f}{'R' if r['closed_rf'] else 'r'}{'T' if r['closed_tf'] else 't'}" for r in rr))
print("DIV2_JSON " + json.dumps(rows))

