import json, sys
import numpy as np
# p1 (revised): key restricted to the complement of within-chunk token-deviation PCs (removes noise/x_t-driven features)
C7 = -1.8
KPC = int(open("/work/notes/kpc.txt").read()) if False else 96
SAVE = True
i1, A1 = acts("a-diag1"); m1 = json.load(open("/work/notes/diag1_meta.json"))
i2, A2 = acts("a-pres2"); m2 = json.load(open("/work/notes/pres2_meta.json"))
H1 = A1["action.head_input"].astype(np.float64); H2 = A2["action.head_input"].astype(np.float64)
K = base("action.action_out_proj.kernel").astype(np.float64); bb = base("action.action_out_proj.bias").astype(np.float64)
Hall = np.concatenate([H1, H2])  # [N,S,15,1024]
dev = (Hall - Hall.mean(2, keepdims=True)).reshape(-1, 1024)
U, sv, Vt = np.linalg.svd(dev, full_matrices=False)
ev = sv ** 2 / (sv ** 2).sum()
print("token-deviation variance captured by top k PCs:", {k: round(float(ev[:k].sum()), 3) for k in (8, 16, 32, 64, 96, 128)})
P = np.eye(1024) - Vt[:KPC].T @ Vt[:KPC]
def ix(meta, tag, p="orig"):
    return [i for i, m in enumerate(meta) if m["tag"] == tag and m.get("prompt", "orig") == p]
G = {"rel": H1[ix(m1, "rel")], "bowlrel": H2[ix(m2, "bowlrel")], "stage2": H2[ix(m2, "stage2")], "approach": H2[ix(m2, "approach")]}
rows, ys = [], []
for s in range(3):
    for t, y in (("rel", 1.0), ("bowlrel", 0.0), ("stage2", 0.0), ("approach", 0.0)):
        rows.append(P @ G[t][:, s].mean((0, 1))); ys.append(y)
M = np.stack(rows); y = np.array(ys)
a = M.T @ np.linalg.solve(M @ M.T, y)
a = P @ a
print("|a| =", round(float(np.linalg.norm(a)), 3), "|C7*a| =", round(float(np.linalg.norm(C7 * a)), 3), "|K7| =", round(float(np.linalg.norm(K[:, 7])), 3))
for A, Mm in ((A1, m1), (A2, m2)):
    H = A["action.head_input"].astype(np.float64)
    key = (H @ a)  # [N,S,15]
    a7 = A["actions"][:, :, 7].astype(np.float64)
    ks = key.mean(1)  # [N,15]
    new = a7 - C7 * ks / 2.0
    for i, m in enumerate(Mm):
        if m.get("prompt", "orig") != "orig":
            continue
        b_close = (a7[i] > 0.5).mean(); n_close = (new[i] > 0.5).mean()
        flag = "" if (b_close > 0.5) == (n_close > 0.5) else "  <-- FLIP"
        print(f'{m["tag"]:8s} {m["slot"]} k{m["k"]:2d} key {ks[i].mean():+.2f} tokstd {ks[i].std():.2f} base_close {b_close:.2f} new_close {n_close:.2f}{flag}')
if SAVE:
    Kn = K.copy(); Kn[:, 7] = K[:, 7] + C7 * a
    save("action.action_out_proj.kernel", bf16(Kn.astype(np.float32)))
    np.save("/work/notes/p1_key.npy", a)
    print("saved C7", C7, "KPC", KPC)
