import json
idx, arr = acts("a1-lr-postgrasp")
N = 32
vr = arr["vlm.resid_out"].astype(np.float32)   # 64,18,24,2048
vh = arr["vlm.mlp_hidden"].astype(np.float32)  # 64,18,24,16384
D = vr[:N] - vr[N:]
print("rel |mean D| / |r| per layer, slots 8..23 (slot 8='left')")
for L in range(18):
    nrm = np.linalg.norm(vr[:N, L], axis=-1).mean(0)
    md = np.linalg.norm(D[:, L].mean(0), axis=-1)
    # consistency: |mean D| / mean |D_i|
    cons = md / (np.linalg.norm(D[:, L], axis=-1).mean(0) + 1e-9)
    print("L%02d" % L, " ".join("%.2f" % x for x in (md / nrm)[8:]), "| cons8 %.2f cons_last %.2f |r8| %.0f" % (cons[8], cons[23], nrm[8]))
# unit selectivity at slot 8
for L in [0, 1, 2, 3, 4, 5, 6]:
    hL = vh[:N, L, 8]           # left token hidden, 32 x 16384
    hR = vh[N:, L, 8]
    others = np.concatenate([vh[:, L, s] for s in range(24) if s != 8], 0)  # all other slots, both prompts
    others = np.concatenate([others, hR], 0)
    mL = hL.mean(0)
    omax = np.abs(others).max(0)
    sel = np.where((mL > 0.5) & (omax < 0.1 * mL))[0]
    order = sel[np.argsort(-mL[sel])]
    print("L%d n_selective=%d top:" % (L, len(sel)), [(int(u), round(float(mL[u]), 2), round(float(omax[u]), 3), round(float(hL[:, u].std()), 3)) for u in order[:12]])
    # how much of the left-vs-right MLP output contrast do the selective units explain
