# owsp py base: offline check of a distilled edit on recorded decisions (no rollout).
# Captures (action-only or full; every capture stores the replayed action chunk), all on the same items in the same
# order unless noted:
#   BASE_RF  base policy, RF wording (the task's own)          TF_REF  base policy, TF wording (the target)
#   CAND_RF  edited policy, RF wording
#   and for the no-harm group: NH_BASE (base, TF wording) and NH_CAND (edited, TF wording) on TF-episode decisions.
# Groups are index ranges into the first set. Prints, per group: how far the chunk-end end effector is from the TF
# replay before and after the edit, the gripper decisions, and the lateral (robot y) offset; for no-harm, how far
# the edit moves the TF wording's own actions. Edit the names below.
# (capture id, wording) pairs; captures may mix wordings item by item (acts items carry their own prompt)
RF = "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF = "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
BASE_RF, TF_REF, CAND_RF = ("chk1", RF), ("chk2", TF), ("chk-i", RF)
NH_BASE, NH_CAND = ("chk1", TF), ("chk-i", TF)
GROUPS = {"build": [0, 12], "approach_heldout": [12, 42], "transport_heldout": [42, 56]}  # None = one group


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


def load_actions(src):
    aid, prompt = src
    idx, x = acts(aid)
    sel = [i for i, r in enumerate(idx["items"]) if r["prompt"] == prompt]
    return [(idx["items"][i]["episode"], idx["items"][i]["k"]) for i in sel], x["actions"][sel], idx


def ee(A):
    return np.array([panda_fk(a[-1, :7]) for a in A])


res = {}
kb, B, ib = load_actions(BASE_RF)
kt, T, it = load_actions(TF_REF)
kc, C, ic = load_actions(CAND_RF)
assert kb == kt == kc, "items differ between captures"
eb, et, ec = ee(B), ee(T), ee(C)
gb, gt, gc = gripper_closed(B), gripper_closed(T), gripper_closed(C)
groups = GROUPS or {"all": [0, len(kb)]}
for g, (lo, hi) in groups.items():
    s = slice(lo, hi)
    r = {"n": hi - lo,
         "ee_base_to_tf_cm": round(float(np.linalg.norm(eb[s] - et[s], axis=1).mean() * 100), 2),
         "ee_cand_to_tf_cm": round(float(np.linalg.norm(ec[s] - et[s], axis=1).mean() * 100), 2),
         "ee_cand_vs_base_cm": round(float(np.linalg.norm(ec[s] - eb[s], axis=1).mean() * 100), 2),
         "closer_to_tf": int((np.linalg.norm(ec[s] - et[s], axis=1) < np.linalg.norm(eb[s] - et[s], axis=1)).sum()),
         "dy_base_minus_tf_cm": round(float((eb[s, 1] - et[s, 1]).mean() * 100), 2),
         "dy_cand_minus_tf_cm": round(float((ec[s, 1] - et[s, 1]).mean() * 100), 2),
         "grip_closed_base_tf_cand": [int(gb[s].sum()), int(gt[s].sum()), int(gc[s].sum())],
         "grip_agree_tf_base_cand": [int((gb[s] == gt[s]).sum()), int((gc[s] == gt[s]).sum())]}
    res[g] = r
    print(f"{g:24s} n={r['n']:3d}  ee->TF base {r['ee_base_to_tf_cm']:5.1f} cm  edit {r['ee_cand_to_tf_cm']:5.1f} cm  "
          f"(closer on {r['closer_to_tf']}/{r['n']})  y-TF base {r['dy_base_minus_tf_cm']:+5.1f} edit {r['dy_cand_minus_tf_cm']:+5.1f} cm  "
          f"grip closed b/tf/e {r['grip_closed_base_tf_cand']}  agree-with-TF b/e {r['grip_agree_tf_base_cand']}")
try:
    kn, NB, _ = load_actions(NH_BASE)
    kn2, NC, _ = load_actions(NH_CAND)
    assert kn == kn2
    enb, enc = ee(NB), ee(NC)
    r = {"n": len(kn), "ee_shift_cm": round(float(np.linalg.norm(enc - enb, axis=1).mean() * 100), 2),
         "ee_shift_max_cm": round(float(np.linalg.norm(enc - enb, axis=1).max() * 100), 2),
         "grip_flips": int((gripper_closed(NB) != gripper_closed(NC)).sum())}
    res["tf_noharm"] = r
    print(f"{'tf_noharm':24s} n={r['n']:3d}  edit moves TF actions by {r['ee_shift_cm']:.1f} cm (max {r['ee_shift_max_cm']:.1f})  "
          f"gripper flips {r['grip_flips']}")
except Exception as e:  # noqa: BLE001
    print("no-harm captures not available:", repr(e)[:200])
print("CHECK_JSON " + json.dumps(res))
