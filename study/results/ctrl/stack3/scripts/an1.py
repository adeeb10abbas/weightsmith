import numpy as np, json
idx, A = acts("a1-stackclose")
print("keys", list(A.keys()))
for k, v in A.items():
    print(k, np.asarray(v).shape)
sel = json.load(open("/work/notes/sel1.json"))
lab = np.array(sel["labels"])
eid = np.array([it["episode"] for it in sel["items"]])
H = np.asarray(A["action.head_input"], dtype=np.float64)  # [N,S,15,1024]
act = np.asarray(A["actions"], dtype=np.float64)          # [N,15,8]
W = base("action.action_out_proj.kernel").astype(np.float64); b = base("action.action_out_proj.bias").astype(np.float64)
v7 = H @ W[:, 7] + b[7]   # [N,S,15]
print("gripper action mean per group (captured):")
for g in ["T", "P1", "P2", "P4"]:
    m = lab == g
    print(" ", g, m.sum(), "a7 mean", act[m, :, 7].mean().round(3), "frac>0.5", (act[m, :, 7] > 0.5).mean().round(2), "v7 mean", v7[m].mean().round(3))
hb = H.mean(axis=(1, 2))  # [N,1024]
Tm = lab == "T"; Pm = (lab == "P1") | (lab == "P2")
def direction(mask_t, mask_p):
    d = hb[mask_t].mean(0) - hb[mask_p].mean(0)
    return d / np.linalg.norm(d)
# split-half validation by episode number parity
num = np.array([int(e[-2:]) for e in eid])
for name, trn in [("all", np.ones(len(lab), bool)), ("even", num % 2 == 0), ("odd", num % 2 == 1)]:
    d = direction(Tm & trn, Pm & trn)
    p = hb @ d
    print(f"dir from {name}:")
    for g in ["T", "P1", "P2", "P4"]:
        for part, pm in [("trn", trn), ("tst", ~trn)]:
            m = (lab == g) & pm
            if m.sum() == 0: continue
            print(f"   {g} {part} n={m.sum()} proj mean={p[m].mean():.2f} min={p[m].min():.2f} max={p[m].max():.2f}")
d = direction(Tm, Pm)
np.save("/work/notes/d1.npy", d)
# per-token per-step projection distributions
pt = H @ d
print("per token proj: T min/mean", pt[Tm].min().round(2), pt[Tm].mean().round(2), " P max/mean", pt[Pm].max().round(2), pt[Pm].mean().round(2), " P4 mean", pt[lab=="P4"].mean().round(2))
print("hb norm mean", np.linalg.norm(hb, axis=1).mean().round(2), "W7 norm", np.linalg.norm(W[:, 7]).round(3), "cos(d,W7)", (d @ W[:, 7] / np.linalg.norm(W[:, 7])).round(3))
