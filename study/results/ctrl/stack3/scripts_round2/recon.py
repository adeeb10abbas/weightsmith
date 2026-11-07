import numpy as np, json
f64 = lambda a: np.asarray(a, dtype=np.float64)
idx, A = acts("a2-pertoken")
for k, v in A.items():
    print(k, np.asarray(v).shape)
R = f64(A["action.resid_out"])      # [N,S,2,15,1024]  layers 16,17
MH = f64(A["action.mlp_hidden"])    # [N,S,2,15,4096]
H = f64(A["action.head_input"])     # [N,S,15,1024]
ts = np.array([1.0 - 0.1 * s for s in range(10)])

def swish(x): return x / (1 + np.exp(-x))
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def posemb(t, dim=1024, minp=4e-3, maxp=4.0):
    frac = np.linspace(0, 1, dim // 2); period = minp * (maxp / minp) ** frac
    inp = t[:, None] * (1.0 / period * 2 * np.pi)[None]
    return np.concatenate([np.sin(inp), np.cos(inp)], -1)
te = posemb(ts)
te = swish(te @ f64(base("action.time_mlp_in.kernel")) + f64(base("action.time_mlp_in.bias")))
te = swish(te @ f64(base("action.time_mlp_out.kernel")) + f64(base("action.time_mlp_out.bias")))
def mod(name):
    m = te @ f64(base(name + ".Dense_0.kernel")) + f64(base(name + ".Dense_0.bias"))  # [S,3072]
    return m[:, :1024], m[:, 1024:2048], m[:, 2048:]
def rms_norm(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)
sF, shF, _ = mod("action.final_norm")
s17, sh17, g17 = mod("action.blocks.17.pre_ffw_norm_1")
np.savez("/work/notes/tcond.npz", te=te, sF=sF, shF=shF, s17=s17, sh17=sh17, g17=g17)
r17 = R[:, :, 1]   # [N,S,15,1024]
Hrec = rms_norm(r17) * (1 + sF[None, :, None, :]) + shF[None, :, None, :]
err = np.abs(Hrec - H).max(); rel = np.linalg.norm(Hrec - H) / np.linalg.norm(H)
print(f"final-norm reconstruction: max abs err {err:.4f}, rel {rel:.4f}")
# try alternative: maybe ts reversed / no swish on out
L17 = f64(base("action.blocks.17.mlp_1.linear")); G = f64(base("action.blocks.17.mlp_1.gating_einsum"))
mlp_out = MH[:, :, 1] @ L17                                    # [N,S,15,1024]
rmid = r17 - g17[None, :, None, :] * mlp_out
x = rms_norm(rmid) * (1 + s17[None, :, None, :]) + sh17[None, :, None, :]
hrec = gelu(x @ G[0]) * (x @ G[1])
rel2 = np.linalg.norm(hrec - MH[:, :, 1]) / np.linalg.norm(MH[:, :, 1])
print(f"MLP17 hidden reconstruction rel err {rel2:.4f}  (corr {np.corrcoef(hrec.ravel()[:2000000], MH[:, :, 1].ravel()[:2000000])[0,1]:.4f})")
print("rms(r17) per item stats", np.percentile(np.sqrt((r17 ** 2).mean(-1)), [1, 50, 99]).round(2))
print("g17 stats", np.percentile(g17, [1, 50, 99]).round(3), " sF stats", np.percentile(sF, [1, 50, 99]).round(3))
