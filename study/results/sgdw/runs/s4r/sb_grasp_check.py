# owsp py base: gripper decisions on grasp replans (TF episodes' first closing replan): base RF/TF vs edited RF.
import json
RF = "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF = "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
CANDS = [c for c in ("gchk-g3", "gchk-g2") if os.path.exists(f"/work/data/acts/{c}/index.json")]


def closed(a):
    return (np.asarray(a)[..., 7] > 0.5).mean(-1) >= 0.5


def get(aid, prompt):
    idx, x = acts(aid)
    sel = [i for i, r in enumerate(idx["items"]) if r["prompt"] == prompt]
    return [(idx["items"][i]["episode"], idx["items"][i]["k"]) for i in sel], x["actions"][sel]


kb, B = get("gchk-b", RF)          # held-out grasp replans, base RF
kt, T = get("gchk-b", TF)          # held-out grasp replans, base TF
_, BB = get("s4r-grasp-ox", RF)    # build grasp replans, base RF
_, BT = get("s4r-grasp-ox", TF)    # build grasp replans, base TF
res = {"heldout_grasp": {"n": len(kb), "closed_base_rf": int(closed(B).sum()), "closed_base_tf": int(closed(T).sum())},
       "build_grasp": {"n": len(BB), "closed_base_rf": int(closed(BB).sum()), "closed_base_tf": int(closed(BT).sum())}}
for c in CANDS:
    kc, C = get(c, RF)            # first 6 = build grasp replans, last 6 = held-out
    res["build_grasp"][f"closed_{c}_rf"] = int(closed(C[:6]).sum())
    res["heldout_grasp"][f"closed_{c}_rf"] = int(closed(C[6:]).sum())
print(json.dumps(res, indent=1))
print("GRASP_JSON " + json.dumps(res))
