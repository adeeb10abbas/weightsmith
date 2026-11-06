import json
# Steering edit: add delta at layer-0 residual of the "left" token (slot 10) via the two layer-0 MLP
# units that fire (almost) only on the token "left" (8023, 3097). delta = A*(left-right) + B*(front-back),
# measured as layer-0 resid_out differences at slot 10 (exact, item-independent).
A, B = 0.5, 0.8
UNITS = [8023, 3097]
i1, a1 = acts('a1-prompts')
i2, a2 = acts('a2-frontback')
R1 = np.array(a1['vlm.resid_out'], dtype=np.float32)[:, 0, 10]
R2 = np.array(a2['vlm.resid_out'], dtype=np.float32)[:, 0, 10]
H1 = np.array(a1['vlm.mlp_hidden'], dtype=np.float32)[:, 0]
def sel(idx, R, p):
    ii = [it['i'] for it in idx['items'] if it['prompt'] == p]
    return R[ii].mean(0), ii
rL, iL = sel(i1, R1, "Put the rubiks cube to the left of the bowl")
rR, _ = sel(i1, R1, "Put the rubiks cube to the right of the bowl")
rF, _ = sel(i2, R2, "Put the rubiks cube to the front of the bowl")
rB, _ = sel(i2, R2, "Put the rubiks cube to the back of the bowl")
delta = A * (rL - rR) + B * (rF - rB)
print('A', A, 'B', B, '|rL|', np.linalg.norm(rL).round(1), '|L-R|', np.linalg.norm(rL - rR).round(1),
      '|F-B|', np.linalg.norm(rF - rB).round(1), 'cos(L-R,F-B)', float((rL - rR) @ (rF - rB) / np.linalg.norm(rL - rR) / np.linalg.norm(rF - rB)),
      '|delta|', np.linalg.norm(delta).round(1))
hL = H1[iL, 10][:, UNITS].mean(0)
c = hL / (hL @ hL)          # least-norm coefficients so that sum_u h_u c_u = 1
print('hL units', hL, 'c', c)
name = 'language.blocks.00.mlp.linear'
W = base(name).copy()
for u, cu in zip(UNITS, c):
    W[u] = W[u] + cu * delta
W = bf16(W)
save(name, W)
# realized injection at slot 10 and leakage at other prompt slots
dW = W[UNITS] - base(name)[UNITS]
inj = H1[iL, 10][:, UNITS] @ dW
print('realized |inj| slot10', np.linalg.norm(inj, axis=1).mean().round(2), 'cos to delta',
      float(inj.mean(0) @ delta / np.linalg.norm(inj.mean(0)) / np.linalg.norm(delta)))
for s in range(28):
    if s == 10: continue
    lk = np.linalg.norm(H1[iL, s][:, UNITS] @ dW, axis=1).max()
    if lk > 0.5: print('  leak slot', s, round(float(lk), 2))
# image tokens (grid means) leakage at layer 0
if 'vlm.img_mlp_hidden' in a2:
    Hi = np.array(a2['vlm.img_mlp_hidden'], dtype=np.float32)[:, 0][..., UNITS]  # [N,cam,17,2]
    lk = np.linalg.norm(Hi @ dW, axis=-1)
    print('image-token grid-mean leak max', lk.max().round(3), 'mean', lk.mean().round(3),
          'unit acts on image max', np.abs(Hi).max(axis=(0, 1, 2)))
print('saved', saved())
