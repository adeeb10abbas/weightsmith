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
BASE_RF, TF_REF, CAND_RF = ("chk1", RF), ("chk2", TF), ("chk-c", RF)
NH_BASE, NH_CAND = ("chk1", TF), ("chk-c", TF)
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
kb, B, _ = load_actions(BASE_RF)          # chk1, RF wording
kc, C, _ = load_actions(CAND_RF)          # chk-c, RF wording
assert kb == kc
eb, ec = ee(B), ee(C)
gb, gc = gripper_closed(B), gripper_closed(C)
# TF reference for the build items: capture s4r-ox items 12..23 (base, TF wording)
kt, T, _ = load_actions(("s4r-ox", TF))
assert kt == kb[:12]
et, gt = ee(T), gripper_closed(T)
s = slice(0, 12)
r = {"n": 12, "ee_base_to_tf_cm": round(float(np.linalg.norm(eb[s] - et, axis=1).mean() * 100), 2),
     "ee_cand_to_tf_cm": round(float(np.linalg.norm(ec[s] - et, axis=1).mean() * 100), 2),
     "closer_to_tf": int((np.linalg.norm(ec[s] - et, axis=1) < np.linalg.norm(eb[s] - et, axis=1)).sum()),
     "grip_agree_tf_base_cand": [int((gb[s] == gt).sum()), int((gc[s] == gt).sum())]}
print("build (vs TF replay):", r)
res["build_vs_tf"] = r
for g, (lo, hi) in GROUPS.items():
    s = slice(lo, hi)
    d = ec[s] - eb[s]
    r = {"n": hi - lo, "ee_shift_cm": round(float(np.linalg.norm(d, axis=1).mean() * 100), 2),
         "mean_dy_cand_minus_base_cm": round(float(d[:, 1].mean() * 100), 2),
         "mean_dz_cand_minus_base_cm": round(float(d[:, 2].mean() * 100), 2),
         "grip_closed_base_cand": [int(gb[s].sum()), int(gc[s].sum())], "grip_flips": int((gb[s] != gc[s]).sum())}
    res[g] = r
    print(g, r)
kn, NB, _ = load_actions(NH_BASE)
kn2, NC, _ = load_actions(NH_CAND)
assert kn == kn2
d = ee(NC) - ee(NB)
r = {"n": len(kn), "ee_shift_cm": round(float(np.linalg.norm(d, axis=1).mean() * 100), 2),
     "ee_shift_max_cm": round(float(np.linalg.norm(d, axis=1).max() * 100), 2),
     "grip_flips": int((gripper_closed(NB) != gripper_closed(NC)).sum())}
res["tf_noharm"] = r
print("tf_noharm", r)
print("CHECK_JSON " + json.dumps(res))

