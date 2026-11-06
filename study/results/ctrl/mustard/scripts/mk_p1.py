import json
BETA = 2.0
UNITS = [3097, 8023]
idx, arr = acts("a1-lr-postgrasp")
N = 32
vr = arr["vlm.resid_out"].astype(np.float32)
vh = arr["vlm.mlp_hidden"].astype(np.float32)
dr = (vr[:N, 0, 8] - vr[N:, 0, 8]).mean(0)          # left-minus-right contrast at the 'left' token after VLM block 0
h = vh[:N, 0, 8][:, UNITS].mean(0)
print("|dr|", np.linalg.norm(dr), "|r8|", np.linalg.norm(vr[:N, 0, 8], axis=-1).mean(), "h units", h)
gamma = BETA / h.sum()
name = "language.blocks.00.mlp.linear"
W = load(name).copy()
print("row norms before", np.linalg.norm(W[UNITS], axis=1), "typical row norm", np.linalg.norm(W, axis=1).mean())
for u in UNITS:
    W[u] = W[u] + gamma * dr
W = bf16(W)
save(name, W)
# check realized addition on the left token
Wb = base(name)
add = (vh[:N, 0, 8] @ (W - Wb))   # 32 x 2048
print("realized add / dr (cos, ratio):", float(add.mean(0) @ dr / (np.linalg.norm(add.mean(0)) * np.linalg.norm(dr))), float(np.linalg.norm(add.mean(0)) / np.linalg.norm(dr)))
# leakage on other text tokens (both prompts, all other slots)
oth = np.concatenate([vh[:, 0, s] for s in range(24) if s != 8] + [vh[N:, 0, 8]], 0)
leak = np.linalg.norm(oth @ (W - Wb), axis=-1)
print("leak on other text tokens: max |add| %.2f mean %.3f (vs |dr|*beta=%.1f)" % (leak.max(), leak.mean(), BETA * np.linalg.norm(dr)))
np.save("/work/notes/p1_dr0.npy", dr)
print(saved())
