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


io, xo = acts(CAP_RF)
it, xt = acts(CAP_TF)
assert [(r["episode"], r["k"]) for r in io["items"]] == [(r["episode"], r["k"]) for r in it["items"]], "items differ"
A, B = xo["actions"], xt["actions"]
rows = []
for i, r in enumerate(io["items"]):
    ea, eb = panda_fk(A[i][-1, :7]), panda_fk(B[i][-1, :7])
    e0 = panda_fk(A[i][0, :7])
    rows.append({"i": i, "episode": r["episode"], "k": r["k"], "step": r["sim_step"], "grip_state": round(r["gripper_state"], 3),
                 "ee_dist_cm": round(float(np.linalg.norm(ea - eb)) * 100, 2),
                 "dy_tf_minus_rf_cm": round(float(eb[1] - ea[1]) * 100, 2),
                 "rf_move_cm": [round(float(v) * 100, 1) for v in (ea - e0)],
                 "closed_rf": bool(gripper_closed(A[i])), "closed_tf": bool(gripper_closed(B[i]))})
eps = sorted({r["episode"] for r in rows})
print(f"{len(rows)} items from {len(eps)} episodes; parity rf {io['base_parity_max_abs_action_error']:.4f}")
for e in eps:
    rr = [r for r in rows if r["episode"] == e]
    print(e, " ".join(f"k{r['k']}:{r['ee_dist_cm']:.0f}/{r['dy_tf_minus_rf_cm']:+.0f}{'G' if r['closed_rf'] != r['closed_tf'] else ''}" for r in rr))
rank = sorted(rows, key=lambda r: -r["ee_dist_cm"])
pick, used = [], {}
for r in rank:
    if used.get(r["episode"], 0) >= PER_EP:
        continue
    pick.append(r)
    used[r["episode"]] = used.get(r["episode"], 0) + 1
    if len(pick) >= TOP:
        break
print("top items:", [(r["episode"], r["k"], r["ee_dist_cm"], r["dy_tf_minus_rf_cm"]) for r in pick])
print("DIVERGE_JSON " + json.dumps({"rows": rows, "pick": [{"episode": r["episode"], "k": r["k"]} for r in pick]}))
