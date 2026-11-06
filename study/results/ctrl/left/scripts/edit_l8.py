import json
# Steering at layer LAY, slot 10 ("left" token), written through a least-norm combination of layer-LAY MLP
# units selective for that slot. delta = A*(left-right) + B*(front-back) measured at resid_out[LAY, slot 10].
LAY, LI = 8, 3          # layer and its index in vlm_layers [0,1,3,8,17]
A, B = 0.5, 1.0
K = 64                  # candidate units
RIDGE = 1e-2
i1, a1 = acts('a1-prompts')
i2, a2 = acts('a2-frontback')
H = np.array(a1['vlm.mlp_hidden'], dtype=np.float32)[:, LI]      # [72,28,16384]
R1 = np.array(a1['vlm.resid_out'], dtype=np.float32)[:, LI, 10]
R2 = np.array(a2['vlm.resid_out'], dtype=np.float32)[:, LI, 10]
Hi = np.array(a2['vlm.img_mlp_hidden'], dtype=np.float32)[:, LI].reshape(-1, 16384)   # image grid cells
L = list(range(12)); Rr = list(range(12, 24))
delta = A * (R1[L].mean(0) - R1[Rr].mean(0)) + B * (R2[:12].mean(0) - R2[12:].mean(0))
h10 = H[L, 10]                                        # [12,16384] target rows
oth = H[L][:, [s for s in range(28) if s != 10]].reshape(-1, 16384)
othp = H[12:][:, :, :].reshape(-1, 16384)             # all slots of other prompts (incl. their slot 10) - not needed at deploy
score = h10.min(0) - np.maximum(np.abs(oth).max(0), np.abs(Hi).max(0))
U = np.argsort(-score)[:K]
# least squares: minimize sum (h10 c - 1)^2 * w + sum (oth c)^2 + sum (img c)^2 + ridge |c|^2
X = np.concatenate([h10[:, U] * 3.0, oth[:, U], Hi[:, U]])
y = np.concatenate([np.full(12, 3.0), np.zeros(len(oth)), np.zeros(len(Hi))])
c = np.linalg.solve(X.T @ X + RIDGE * np.eye(K), X.T @ y)
name = f'language.blocks.{LAY:02d}.mlp.linear'
W = base(name).copy()
W[U] = W[U] + np.outer(c, delta)
W = bf16(W)
save(name, W)
dW = W[U] - base(name)[U]
g10 = h10[:, U] @ c
print('layer', LAY, 'A', A, 'B', B, '|delta|', np.linalg.norm(delta).round(1), '|r10|', np.linalg.norm(R1[L], axis=1).mean().round(0))
print('gain at slot10 per item', g10.round(3))
go = oth[:, U] @ c
print('leak gain other prompt slots: max', np.abs(go).max().round(3), 'mean', np.abs(go).mean().round(3))
gi = Hi[:, U] @ c
print('leak gain image grid cells: max', np.abs(gi).max().round(3), 'mean', np.abs(gi).mean().round(3))
print('|c|', np.linalg.norm(c).round(3), 'max|dW|', np.abs(dW).max())
print('saved', saved())
