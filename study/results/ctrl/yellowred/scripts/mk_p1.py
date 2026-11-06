import json, sys
# Closed-form, approach-gated steering edit in the action expert.
# Delta = mean resid_out(L) over decisions that head to the yellow block minus decisions that head to the rubik/red cluster.
# Gate key k (in MLP hidden space of layer L): least-norm solution of k.h_app = 1, k.h_carry = 0 (class means).
# Edit: W_down[L] += alpha * outer(k, Delta)  (rank 1).
CFG = dict(L=12, alpha=1.0, dry=False)
L = CFG['L']; alpha = CFG['alpha']; dry = CFG.get('dry', False)
idx, arr = acts("a2-phases")
items = idx['items']; N = len(items)
Yi = list(range(0, 21)); Ri = list(range(21, 33)); Ci = list(range(33, N))
La = idx['action_layers']; li = La.index(L)
R = arr['action.resid_out'][:, :, li, 0, :].mean(1)   # N,1024 (mean over denoise steps)
H = arr['action.mlp_hidden'][:, :, li, 0, :].mean(1)  # N,4096
delta = R[Yi].mean(0) - R[Ri].mean(0)
happ = H[Yi + Ri].mean(0); hcar = H[Ci].mean(0)
M = np.stack([happ, hcar])  # 2,4096
lam = CFG.get('ridge', 0.0)
G = M @ M.T + lam * np.eye(2)
k = M.T @ np.linalg.solve(G, np.array([1.0, 0.0]))
resp = H @ k
print('L', L, 'alpha', alpha, '|delta|', np.linalg.norm(delta).round(3), '|k|', np.linalg.norm(k).round(4))
print('resp Y', resp[Yi].round(2))
print('resp R', resp[Ri].round(2))
print('resp C', resp[Ci].round(2))
# per denoise step response
Hs = arr['action.mlp_hidden'][:, :, li, 0, :]
for s in range(Hs.shape[1]):
    rs = Hs[:, s] @ k
    print(' s', s, 'Y', rs[Yi].mean().round(2), 'R', rs[Ri].mean().round(2), 'C', rs[Ci].mean().round(2), 'Cmax', np.abs(rs[Ci]).max().round(2))
name = f"action.blocks.{L:02d}.mlp_1.linear"
W = base(name)
dW = alpha * np.outer(k, delta)
print('|dW|F', np.linalg.norm(dW).round(4), '|W|F', np.linalg.norm(W).round(2), 'max|dW|', np.abs(dW).max())
Wn = bf16(W + dW)
print('changed entries', int((Wn != W).sum()), 'of', W.size)
if not dry:
    save(name, Wn)
    json.dump(dict(L=L, alpha=alpha, k=k.tolist(), delta=delta.tolist()), open('/work/scratch/p1_edit.json', 'w'))
    print('saved', name)
