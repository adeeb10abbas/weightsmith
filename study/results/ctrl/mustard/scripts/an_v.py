import json, sys
AID = "v1-p1-check"
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
idx, arr = acts(AID)
print({k: v for k, v in idx.items() if k in ("shapes", "base_parity_max_abs_action_error")})
items = idx["items"]
act = arr["actions"]; rec = arr["recorded_actions"]
dyB, dyP, rows = [], [], []
for i, it in enumerate(items):
    o = obs(it["episode"]); k = it["k"]
    p0 = fk(o["joint"][k])
    pb = fk(rec[i, -1, :7]); pp = fk(act[i, -1, :7])
    dyB.append(pb[1] - p0[1]); dyP.append(pp[1] - p0[1])
    rows.append((it["episode"][-6:], k, float(np.abs(act[i, :, :7] - rec[i, :, :7]).max()), float(rec[i, :, 7].mean()), float(act[i, :, 7].mean()), pb - p0, pp - p0))
dyB = np.array(dyB); dyP = np.array(dyP)
print("TARGET first-post-grasp (32):")
for r in rows[:32]:
    print("  %s k%d maxdq %.3f grip %.2f->%.2f  dxyz base %s  p1 %s" % (r[0], r[1], r[2], r[3], r[4], np.round(r[5], 3), np.round(r[6], 3)))
t = slice(0, 32)
print("base: left %d right %d | p1: left %d right %d" % ((dyB[t] > 0.05).sum(), (dyB[t] < -0.05).sum(), (dyP[t] > 0.05).sum(), (dyP[t] < -0.05).sum()))
print("mean dy base %.3f p1 %.3f" % (dyB[t].mean(), dyP[t].mean()))
print("PRESERVE approach/grasp (16):")
for r in rows[32:]:
    print("  %s k%d maxdq %.3f grip %.2f->%.2f  dxyz base %s  p1 %s" % (r[0], r[1], r[2], r[3], r[4], np.round(r[5], 3), np.round(r[6], 3)))
if "vlm.img_mlp_hidden" in arr:
    ih = arr["vlm.img_mlp_hidden"].astype(np.float32)
    print("img_mlp_hidden", ih.shape)
    li = idx.get("vlm_layers", [0, 12]).index(0)
    for u in [3097, 8023]:
        v = ih[:, li, :, :, u]
        print("unit", u, "image-token pooled act: mean %.3f maxabs %.3f" % (v.mean(), np.abs(v).max()))
    allu = np.abs(ih[:, li]).max(axis=(0, 1, 2))
    print("ref: median over units of max|pooled act| %.3f" % np.median(allu))
