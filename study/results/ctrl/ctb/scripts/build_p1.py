import json, sys
import numpy as np
# Rank-1 key->value edit on the gripper column of action.action_out_proj.kernel:
#   v7' = v7 + c7 * (head_input . a)
# a = least-norm solution of class-mean constraints (per captured denoise step):
#   mean(rel) . a = 1   (premature release at cube, orig prompt)
#   mean(bowlrel) . a = 0, mean(stage2) . a = 0, mean(approach) . a = 0
C7 = -2.0
SAVE = True
i1, A1 = acts("a-diag1"); m1 = json.load(open("/work/notes/diag1_meta.json"))
i2, A2 = acts("a-pres2"); m2 = json.load(open("/work/notes/pres2_meta.json"))
H1 = A1["action.head_input"].astype(np.float64)  # [N,S,15,1024]
H2 = A2["action.head_input"].astype(np.float64)
K = base("action.action_out_proj.kernel").astype(np.float64); bb = base("action.action_out_proj.bias").astype(np.float64)
def ix(meta, tag, p="orig"):
    return [i for i, m in enumerate(meta) if m["tag"] == tag and m.get("prompt", "orig") == p]
G = {"rel": H1[ix(m1, "rel")], "hold": H1[ix(m1, "hold")], "miss": H1[ix(m1, "miss")], "close": H1[ix(m1, "close")],
     "relc": H1[ix(m1, "rel", "Put the cube in the bowl")]}
for t in ("bowlrel", "carry", "stage2", "approach"):
    G[t] = H2[ix(m2, t)]
rows, ys = [], []
for s in range(3):
    for t, y in (("rel", 1.0), ("bowlrel", 0.0), ("stage2", 0.0), ("approach", 0.0)):
        rows.append(G[t][:, s].mean((0, 1))); ys.append(y)
M = np.stack(rows); y = np.array(ys)
a = M.T @ np.linalg.solve(M @ M.T, y)
print("|a| =", round(float(np.linalg.norm(a)), 3), " constraints:", len(y))
print("per-item key (token-mean) and v7 base -> edited, per step; and token min/max of key")
for t in ("rel", "hold", "relc", "miss", "close", "bowlrel", "carry", "stage2", "approach"):
    X = G[t]  # [n,S,15,1024]
    key = X @ a  # [n,S,15]
    v7 = X @ K[:, 7] + bb[7]
    out = []
    for s in range(3):
        out.append(f"s{s}: key {np.round(key[:, s].mean(1), 2).tolist()} v7 {np.round(v7[:, s].mean(1), 2).tolist()} -> {np.round((v7[:, s] + C7 * key[:, s]).mean(1), 2).tolist()}")
    print("==", t, "n", len(X)); print("\n".join(out))
# also print recorded base action gripper means for pres2 items to see what they did
act2 = A2["actions"]
print("pres2 replay gripper chunk means:")
for i, m in enumerate(m2):
    print(i, m["tag"], m["slot"], m["k"], round(float(act2[i, :, 7].mean()), 2), "first/last", round(float(act2[i, 0, 7]), 2), round(float(act2[i, -1, 7]), 2))
if SAVE:
    Kn = K.copy(); Kn[:, 7] = K[:, 7] + C7 * a
    save("action.action_out_proj.kernel", bf16(Kn.astype(np.float32)))
    np.save("/work/notes/p1_key.npy", a)
    print("saved; max |dK| =", float(np.abs(Kn - K).max()), " col7 |K|", float(np.linalg.norm(K[:, 7])), "|dK7|", float(np.linalg.norm(C7 * a)))
