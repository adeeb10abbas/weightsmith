import json
import numpy as np
idx, A = acts("a-diag1")
meta = json.load(open("/work/notes/diag1_meta.json"))
R = A["action.resid_out"][:, :, :, 0, :]   # [N,S,L,1024]
H = A["action.mlp_hidden"][:, :, :, 0, :]  # [N,S,L,4096]
def ii(tag, p="orig"):
    return [i for i, m in enumerate(meta) if m["tag"] == tag and m["prompt"] == p]
rel, hold, miss, close = ii("rel"), ii("hold"), ii("miss"), ii("close")
relc = ii("rel", "Put the cube in the bowl")
D = R[relc] - R[rel]  # [16,S,L,1024]
print("layer | |D|/|h| per step | consistency cos(item, mean) | cos(meanD, hold-rel)")
for L in range(18):
    rel_n = [float(np.linalg.norm(D[:, s, L], axis=-1).mean() / np.linalg.norm(R[rel][:, s, L], axis=-1).mean()) for s in range(3)]
    md = D[:, :, L].mean(0)  # [S,1024]
    cons = []
    for s in range(3):
        c = [float(D[j, s, L] @ md[s] / (np.linalg.norm(D[j, s, L]) * np.linalg.norm(md[s]) + 1e-9)) for j in range(16)]
        cons.append(np.mean(c))
    hr = R[hold][:, :, L].mean(0) - R[rel][:, :, L].mean(0)
    cosr = [float(md[s] @ hr[s] / (np.linalg.norm(md[s]) * np.linalg.norm(hr[s]))) for s in range(3)]
    print(L, np.round(rel_n, 3), np.round(cons, 2), np.round(cosr, 2))
# hidden-space separability per layer: direction a = mean(H rel) - mean(H [hold + relc]) averaged over steps
print("\nlayer | proj (normalized so rel=1, hold+relc=0): rel, hold, relc, miss, close, (rel min, hold/relc max)")
for L in range(18):
    Hm = H[:, :, L].mean(1)  # mean over steps [N,4096]
    mu_r = Hm[rel].mean(0); mu_n = np.concatenate([Hm[hold], Hm[relc]]).mean(0)
    a = mu_r - mu_n
    p = Hm @ a
    lo, hi = p[hold + relc].mean(), p[rel].mean()
    q = (p - lo) / (hi - lo)
    print(L, [round(float(q[g].mean()), 2) for g in (rel, hold, relc, miss, close)], round(float(q[rel].min()), 2), round(float(q[hold + relc].max()), 2))
