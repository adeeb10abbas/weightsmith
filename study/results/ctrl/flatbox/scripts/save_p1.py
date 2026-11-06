import numpy as np, json
# p1: rank-1 readout edit on action_out_proj gripper column.
# d = (mean head_input | holding box) - (mean | approach or open pre-grasp), with the top-40 principal directions of
# within-item (per-token / per-denoise-step) variation projected out. dv7 = c * (h.d - r), r = mean P projection.
K_PC = 40
C = -1.25
idx, arr = acts('a2-hold')
lab = np.array(['H'] * 16 + ['U'] * 8 + ['P'] * 12 + ['A'] * 8 + ['C'] * 4)
cls = {c: lab == c for c in 'HUPAC'}
HI = arr['action.head_input'].astype(np.float64)
N, S, T, D = HI.shape
hm = HI.mean(2)
d0 = hm[cls['H']].mean((0, 1)) - hm[cls['A'] | cls['P']].mean((0, 1))
W = (HI - HI.mean((1, 2), keepdims=True)).reshape(-1, D)
_, _, Vt = np.linalg.svd(W - W.mean(0), full_matrices=False)
V = Vt[:K_PC]
d = d0 - V.T @ (V @ d0)
d /= np.linalg.norm(d)
proj = HI @ d
r = float(proj[cls['P']].mean())
K = base('action.action_out_proj.kernel').astype(np.float64)
b = base('action.action_out_proj.bias').astype(np.float64)
K2 = K.copy(); b2 = b.copy()
K2[:, 7] = K[:, 7] + C * d
b2[7] = b[7] - C * r
K2 = bf16(K2.astype(np.float32)); b2 = bf16(b2.astype(np.float32))
# effective realized edit after bf16 rounding
dK = K2[:, 7] - K[:, 7]
print('r', r, '|C d|', np.linalg.norm(C * d), 'realized |dK7|', np.linalg.norm(dK), 'cos', dK @ (C * d) / np.linalg.norm(dK) / np.linalg.norm(C * d))
print('bias7', b[7], '->', b2[7])
print('other cols changed:', np.abs(K2[:, :7] - K[:, :7]).max(), np.abs(K2[:, 8:] - K[:, 8:]).max())
# predicted no-feedback gripper shift
tot = (HI @ dK + (b2[7] - b[7])).sum(1)  # N,T  sum over steps of dv7
dx = -0.1 * tot
for c in 'HUPAC':
    print(c, 'pred dx mean', round(float(dx[cls[c]].mean()), 3), 'tok0 range', np.round([dx[cls[c], 0].min(), dx[cls[c], 0].max()], 2))
save('action.action_out_proj.kernel', K2)
save('action.action_out_proj.bias', b2)
np.save('/work/notes/p1_d.npy', d)
json.dump({'r': r, 'C': C, 'K_PC': K_PC}, open('/work/notes/p1_meta.json', 'w'))
print(saved())
