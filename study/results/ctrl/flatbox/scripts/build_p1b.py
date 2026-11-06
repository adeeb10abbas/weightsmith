import numpy as np, json
idx, arr = acts('a2-hold')
lab = np.array(['H'] * 16 + ['U'] * 8 + ['P'] * 12 + ['A'] * 8 + ['C'] * 4)
cls = {c: lab == c for c in 'HUPAC'}
HI = arr['action.head_input'].astype(np.float64)  # N,S,15,1024
N, S, T, D = HI.shape
A = arr['actions']
hm = HI.mean(2)
d0 = (hm[cls['H']].mean((0, 1)) - hm[cls['A'] | cls['P']].mean((0, 1)))
# within-item per-token/per-step variation (nuisance): deviations from each item's own mean
W = (HI - HI.mean((1, 2), keepdims=True)).reshape(-1, D)
# also between-item variation within class
B = []
for c in 'HUPAC':
    m = hm[cls[c]].mean(1)  # items x D
    B.append(m - m.mean(0))
B = np.concatenate(B)
U, s, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
Ub, sb, Vbt = np.linalg.svd(B, full_matrices=False)
print('nuisance sv top', np.round(s[:10] / np.sqrt(len(W)), 3))
res = {}
for k in [0, 5, 10, 20, 40, 80]:
    d = d0.copy()
    if k:
        V = Vt[:k]
        d = d - V.T @ (V @ d)
    d /= np.linalg.norm(d)
    proj = HI @ d  # N,S,T
    r = proj[cls['P']].mean()
    tot = (proj - r).sum(1)  # N,T  sum over steps
    # per class: fraction of tokens with tot > 0 and token-0 value stats
    line = f"k={k:3d} "
    for c in 'HUPAC':
        t0 = tot[cls[c], 0]
        line += f"{c}: mean {tot[cls[c]].mean():+.2f} sd {tot[cls[c]].std():.2f} t0[{t0.min():+.2f},{t0.max():+.2f}] | "
    sep = (tot[cls['H']].mean() - tot[cls['A']].mean()) / tot.std()
    print(line, 'sepHA', round(sep, 2), 'sepHP', round((tot[cls['H']].mean() - tot[cls['P']].mean()) / tot.std(), 2))
    res[k] = d
np.save('/work/notes/p1_dirs.npy', np.array([res[k] for k in [0, 5, 10, 20, 40, 80]]))
