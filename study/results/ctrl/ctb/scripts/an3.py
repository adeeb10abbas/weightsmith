import json
import numpy as np
idx, A = acts("a-diag1")
meta = json.load(open("/work/notes/diag1_meta.json"))
print("denoise_t", idx["denoise_t"], "steps", idx["denoise_steps"])
K = base("action.action_out_proj.kernel"); bb = base("action.action_out_proj.bias")
HI = A["action.head_input"].astype(np.float64)  # [N,S,15,1024]
V = HI @ K + bb  # [N,S,15,32]
def ii(tag, p="orig"):
    return [i for i, m in enumerate(meta) if m["tag"] == tag and m["prompt"] == p]
rel, hold, miss, close = ii("rel"), ii("hold"), ii("miss"), ii("close")
relc = ii("rel", "Put the cube in the bowl"); missc = ii("miss", "Put the cube in the bowl")
for s in range(3):
    print("step", s, "v7 token-mean: rel", np.round(V[rel, s, :, 7].mean(), 3), "relc", np.round(V[relc, s, :, 7].mean(), 3),
          "hold", np.round(V[hold, s, :, 7].mean(), 3), "miss", np.round(V[miss, s, :, 7].mean(), 3), "close", np.round(V[close, s, :, 7].mean(), 3))
    dv = (V[relc, s] - V[rel, s])  # [16,15,32]
    print("   dv (relc-rel) dims0-7 token-mean:", np.round(dv[:, :, :8].mean((0, 1)), 3), " token std of dv7:", np.round(dv[:, :, 7].std(1).mean(), 3))
# action-level difference: joints vs gripper
act = A["actions"]
dj = np.abs(act[relc, :, :7] - act[rel, :, :7]).mean()
print("actions |relc-rel| joints mean abs", round(float(dj), 4), " gripper", round(float(np.abs(act[relc, :, 7] - act[rel, :, 7]).mean()), 3))
print("actions joints rel vs hold typical |diff to own mean|", round(float(np.abs(act[rel, :, :7] - act[rel, :, :7].mean(0)).mean()), 4))
# key selectivity in head_input space (token-mean), per step
Hm = HI.mean(2)  # [N,S,1024]
for s in range(3):
    X = Hm[:, s]
    mu = {g: X[ix].mean(0) for g, ix in (("rel", rel), ("hold", hold), ("relc", relc), ("close", close))}
    M = np.stack([mu["rel"], mu["hold"], mu["relc"], mu["close"]])
    e = np.array([1.0, 0, 0, 0])
    a = M.T @ np.linalg.solve(M @ M.T, e)
    q = X @ a
    print("step", s, "least-norm key |a|", round(float(np.linalg.norm(a)), 2), "proj rel", np.round(q[rel], 2))
    print("     hold", np.round(q[hold], 2), "relc", np.round(q[relc], 2))
    print("     miss", np.round(q[miss], 2), "close", np.round(q[close], 2), "missc", np.round(q[missc], 2))
    # simple contrast key
    d = mu["rel"] - 0.5 * (mu["hold"] + mu["relc"])
    q2 = (X - 0.5 * (mu["hold"] + mu["relc"])) @ d / (d @ d)
    print("     contrast key: rel", np.round(q2[rel], 2), "hold", np.round(q2[hold], 2), "relc", np.round(q2[relc], 2), "miss", np.round(q2[miss], 2), "close", np.round(q2[close], 2))
np.save("/work/notes/diag1_V.npy", V)
