import numpy as np, json, sys
# Build the gate unit edit for action.blocks.17.mlp_1 unit r. Run with `owsp py p1 scripts/build.py` (SAVE=True below).
SAVE = True
K_NOISE = 256; MARGIN = 0.25; KAPPA_MED = 10.0; TARGET_DA = -0.9; UNIT = 2437
f64 = lambda a: np.asarray(a, dtype=np.float64)
idx, A = acts("a2-pertoken")
sel = json.load(open("/work/notes/sel2.json"))
lab = np.array(sel["labels"])
TC = np.load("/work/notes/tcond.npz")
sF, shF, s17, sh17, g17 = TC["sF"], TC["shF"], TC["s17"], TC["sh17"], TC["g17"]
R = f64(A["action.resid_out"]); MH = f64(A["action.mlp_hidden"]); H = f64(A["action.head_input"]); act = f64(A["actions"])
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def rms_norm(x): return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
Lw = f64(base("action.blocks.17.mlp_1.linear")); Gw = f64(base("action.blocks.17.mlp_1.gating_einsum"))
W = f64(base("action.action_out_proj.kernel"))
r17 = R[:, :, 1]
rmid = r17 - g17[None, :, None, :] * (MH[:, :, 1] @ Lw)
X = rms_norm(rmid) * (1 + s17[None, :, None, :]) + sh17[None, :, None, :]
N, S, Tk, Dm = X.shape
Tm = (lab == "T") | (lab == "T2"); Pm = (lab == "P1") | (lab == "P2") | (lab == "P2s")
# --- direction with within-item noise subspace removed
dev = (X - X.mean((1, 2), keepdims=True)).reshape(-1, Dm)
C = dev.T @ dev / len(dev); ev, V = np.linalg.eigh(C); V = V[:, ::-1]
d = X[Tm].mean((0, 1, 2)) - X[Pm].mean((0, 1, 2)); Vk = V[:, :K_NOISE]; d = d - Vk @ (Vk.T @ d); d /= np.linalg.norm(d)
xa = X.reshape(-1, Dm); mu = xa.mean(0); Ca = np.cov(xa.T); lam = 1e-3 * np.trace(Ca) / Dm
v = np.linalg.solve(Ca + lam * np.eye(Dm), mu); v /= mu @ v
zr = (X @ d) / (X @ v)
pmax = zr[Pm].max(); tmed = np.median(zr[Tm]); theta = pmax + MARGIN * (tmed - pmax)
kappa = KAPPA_MED / (tmed - theta)
print(f"theta {theta:.4f} pmax {pmax:.4f} tmed {tmed:.4f} kappa {kappa:.2f}")
g0 = kappa * (d - theta * v)
g1 = v.copy()            # up channel ~ 1
# --- least-norm output row: zero first-order change on joint dims 0..6 at every step, unit change on dim 7
rows, rhs = [], []
rbarT = r17[Tm].mean((0, 2)); rbarA = r17.mean((0, 2))     # [S,1024] mean pre-norm residual per step
for s in range(S):
    a_s = (1 + sF[s]) * g17[s]
    for j in range(8):
        rows.append(a_s * W[:, j]); rhs.append(1.0 if j == 7 else 0.0)
    rows.append(g17[s] * rbarT[s]); rhs.append(0.0)        # no first-order rms change on target tokens
    rows.append(g17[s] * rbarA[s]); rhs.append(0.0)        # nor on average tokens
M = np.array(rows); b = np.array(rhs)
Dw = (g17 ** 2).sum(0) + 1e-8                               # weighted least norm: minimise sum_s |g17_s * o|^2 = |delta r|^2
o = (M / Dw).T @ np.linalg.solve((M / Dw) @ M.T, b)
print("o norm", np.linalg.norm(o).round(2), "max|o|", np.abs(o).max().round(2), "constraint resid", np.abs(M @ o - b).max())
def simulate(g0, g1, o, scale):
    pre = X @ g0; up = X @ g1
    h = gelu(pre) * up                                    # [N,S,15]
    hold = MH[:, :, 1, :, UNIT]
    dr = g17[None, :, None, :] * (h[..., None] * (scale * o)[None, None, None, :] - hold[..., None] * Lw[UNIT][None, None, None, :])
    Hn = rms_norm(r17 + dr) * (1 + sF[None, :, None, :]) + shF[None, :, None, :]
    dv = (Hn - H) @ W[:, :8]                              # [N,S,15,8]
    da = -0.1 * dv.sum(1)                                 # [N,15,8]
    ratio = np.linalg.norm(dr, axis=-1) / np.linalg.norm(r17, axis=-1)
    print(f"   |dr|/|r| on T tokens: median {np.median(ratio[Tm]):.3f} max {ratio[Tm].max():.3f}")
    return h, da
h, da1 = simulate(g0, g1, o, 1.0)
# choose scale so that median T token gets TARGET_DA on gripper
med = np.median(da1[Tm][:, :, 7]); scale = TARGET_DA / med
print("median da7 per unit scale on T", med.round(4), "-> scale", scale.round(2))
g0b, g1b, ob = bf16(g0), bf16(g1), bf16(scale * o)
h, da = simulate(g0b, g1b, ob, 1.0)
a7 = act[:, :, 7]; a7n = a7 + da[:, :, 7]
for g in ["T", "T2", "P1", "P2", "P2s", "P4"]:
    m = lab == g
    print(f"{g:4s} n={m.sum():2d} close-frac {np.mean(a7[m] > 0.5):.2f} -> {np.mean(a7n[m] > 0.5):.2f} | da7 min/med/max {da[m][:,:,7].min():.3f}/{np.median(da[m][:,:,7]):.3f}/{da[m][:,:,7].max():.3f} | max|dJoint| {np.abs(da[m][:,:,:7]).max():.4f} | h max {h[m].max():.2f} min {h[m].min():.3f}")
flipP = [(sel['items'][i]['episode'][-6:], sel['items'][i]['k'], lab[i], int(((a7[i] > 0.5) & (a7n[i] <= 0.5)).sum())) for i in np.where(Pm)[0] if ((a7[i] > 0.5) & (a7n[i] <= 0.5)).any()]
print("P items with any token flipped to open:", flipP)
print("T items with <8 tokens closed after edit:", int(sum((a7n[i] > 0.5).sum() < 8 for i in np.where(Tm)[0])), "of", int(Tm.sum()))
# bf16 robustness: real model feeds bf16 x into the gate
Xb = bf16(X.astype(np.float32)).astype(np.float64)
pre_f = X @ g0b; pre_b = Xb @ g0b; up_b = Xb @ g1b
print("norm v", np.linalg.norm(v).round(2), "sum|x_i v_i| median", np.median(np.abs(X[:4] * v).sum(-1)).round(2))
print("pre-act bf16-x deviation: median %.3f  p99 %.3f max %.3f | up channel bf16 range %.3f..%.3f" % (
    np.median(np.abs(pre_b - pre_f)), np.percentile(np.abs(pre_b - pre_f), 99), np.abs(pre_b - pre_f).max(), up_b.min(), up_b.max()))
print("P tokens max pre-act (bf16 x): %.2f   T tokens frac pre-act>0 (bf16 x): %.2f" % (pre_b[Pm].max(), (pre_b[Tm] > 0).mean()))
np.savez("/work/notes/gate_p1.npz", g0=g0b, g1=g1b, o=ob, d=d, v=v, theta=theta, kappa=kappa)
if SAVE:
    G = load("action.blocks.17.mlp_1.gating_einsum").copy()
    G[0][:, UNIT] = g0b; G[1][:, UNIT] = g1b
    save("action.blocks.17.mlp_1.gating_einsum", bf16(G))
    L = load("action.blocks.17.mlp_1.linear").copy()
    L[UNIT] = ob
    save("action.blocks.17.mlp_1.linear", bf16(L))
    print("saved")
