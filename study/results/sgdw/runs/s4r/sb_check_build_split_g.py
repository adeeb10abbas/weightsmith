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
BASE_RF, TF_REF, CAND_RF = ("chk1", RF), ("chk2", TF), ("chk-g", RF)
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



kb, B, _ = load_actions(BASE_RF)
kc, C, _ = load_actions(CAND_RF)
kt, T, _ = load_actions(("s4r-ox", TF))
eb, ec, et = ee(B[:12]), ee(C[:12]), ee(T)
for name, s in (("approach_build", slice(0, 6)), ("transport_build", slice(6, 12))):
    print(name, "ee->TF base %.1f edit %.1f cm" % (np.linalg.norm(eb[s] - et[s], axis=1).mean() * 100, np.linalg.norm(ec[s] - et[s], axis=1).mean() * 100),
          "| dy TF-base %+.1f, edit-base %+.1f cm" % ((et[s, 1] - eb[s, 1]).mean() * 100, (ec[s, 1] - eb[s, 1]).mean() * 100),
          "| dz TF-base %+.1f, edit-base %+.1f cm" % ((et[s, 2] - eb[s, 2]).mean() * 100, (ec[s, 2] - eb[s, 2]).mean() * 100))
    for i in range(s.start, s.stop):
        print("   ", kb[i][0][-18:], kb[i][1], "dy TF %+.1f edit %+.1f | dist base %.1f edit %.1f" % ((et[i, 1] - eb[i, 1]) * 100, (ec[i, 1] - eb[i, 1]) * 100, np.linalg.norm(eb[i] - et[i]) * 100, np.linalg.norm(ec[i] - et[i]) * 100))

