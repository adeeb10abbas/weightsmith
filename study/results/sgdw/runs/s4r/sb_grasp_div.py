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



kr, R, _ = load_actions(("s4r-grasp-ox", RF))
kt, T, _ = load_actions(("s4r-grasp-ox", TF))
assert kr == kt
er, et = ee(R), ee(T)
for i in range(len(kr)):
    print(kr[i][0][-6:], kr[i][1], "grip cmd RF %.2f TF %.2f" % (R[i][:, 7].mean(), T[i][:, 7].mean()), "closed RF", bool(gripper_closed(R[i:i+1])[0]), "TF", bool(gripper_closed(T[i:i+1])[0]), "| ee dist %.1f cm, dz TF-RF %+.1f" % (np.linalg.norm(er[i] - et[i]) * 100, (et[i, 2] - er[i, 2]) * 100))

