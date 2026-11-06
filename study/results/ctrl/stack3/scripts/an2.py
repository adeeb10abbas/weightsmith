import numpy as np, json
idx, A = acts("a1-stackclose")
sel = json.load(open("/work/notes/sel1.json"))
lab = np.array(sel["labels"])
eid = np.array([it["episode"] for it in sel["items"]])
num = np.array([int(e[-2:]) for e in eid])
M = np.asarray(A["action.mlp_hidden"], dtype=np.float64)[:, :, :, 0, :]  # [N,S,2,4096]
Tm = lab == "T"; Pm = (lab == "P1") | (lab == "P2"); P4 = lab == "P4"
for li, L in enumerate([8, 17]):
    X = M[:, :, li, :].mean(1)  # [N,4096] mean over denoise steps
    res = []
    for split in ["even", "odd"]:
        trn = (num % 2 == 0) if split == "even" else (num % 2 == 1)
        tst = ~trn
        # margin on train: min_T - max_P (sign-agnostic)
        mt = X[Tm & trn].min(0) - X[Pm & trn].max(0)
        mn = X[Pm & trn].min(0) - X[Tm & trn].max(0)
        res.append((mt, mn, trn, tst))
    # units with positive train margin in both splits (T high)
    sc = np.minimum(res[0][0], res[1][0])
    order = np.argsort(-sc)[:8]
    print(f"layer {L}: best T>P units (min train margin across splits)")
    for u in order:
        x = X[:, u]
        print(f"   u{u} margin={sc[u]:.3f} T[{x[Tm].min():.2f},{x[Tm].mean():.2f},{x[Tm].max():.2f}] P1max={x[lab=='P1'].max():.2f} P2max={x[lab=='P2'].max():.2f} Pmean={x[Pm].mean():.2f} P4mean={x[P4].mean():.2f}")
    sc2 = np.minimum(res[0][1], res[1][1])
    order2 = np.argsort(-sc2)[:4]
    print(f"layer {L}: best P>T units")
    for u in order2:
        x = X[:, u]
        print(f"   u{u} margin={sc2[u]:.3f} Tmax={x[Tm].max():.2f} Pmin={x[Pm].min():.2f}")
    # how many units fully separate on all data
    full = X[Tm].min(0) - X[Pm].max(0)
    print(f"   n units with T>P full separation: {(full > 0).sum()}")
