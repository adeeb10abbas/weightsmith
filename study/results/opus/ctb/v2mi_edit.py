# v2-multi + image cells: as v2m_edit.py (bos, 'Task:', 37 state slots) plus 32 image keys (2 cams x 16 cells): image
# cell key = mean orig img MLP hidden in that cell, value = ALPHA * per-layer increment of mean(V2 - orig) in that cell.
import json
CFG = json.load(open('/work/notes/steer_cfg.json')); LAYERS = CFG['layers']; ALPHA = CFG['alpha']; LAM = CFG.get('lam', 1e-3)
_, X = acts('c2-multi')
VR = X['vlm.resid_out'].astype(np.float64); VH = X['vlm.mlp_hidden'].astype(np.float64)
IR = X['vlm.img_resid_out'].astype(np.float64); IH = X['vlm.img_mlp_hidden'].astype(np.float64)
so, sv, n = 13, 17, 37
oslots = [0, 1, 2] + list(range(so, so + n)); vslots = [0, 1, 2] + list(range(sv, sv + n))
def dl(L):
    t = [(VR[12:18, L, b] - VR[0:6, L, a]).mean(0) for a, b in zip(oslots, vslots)]
    im = [(IR[12:18, L, c, e] - IR[0:6, L, c, e]).mean(0) for c in range(2) for e in range(1, 17)]
    return np.stack(t + im)
D = {L: dl(L) for L in range(17)}
for L in LAYERS:
    V = ALPHA * (D[L] - (D[L - 1] if L > 0 else 0))
    K = np.stack([VH[0:6, L, a].mean(0) for a in oslots] + [IH[0:6, L, c, e].mean(0) for c in range(2) for e in range(1, 17)])
    G = K @ K.T; lam = LAM * np.trace(G) / len(G)
    dW = K.T @ np.linalg.solve(G + lam * np.eye(len(G)), V)
    name = 'language.blocks.%02d.mlp.linear' % L
    W = load(name).copy(); save(name, bf16(W + dW))
print('saved', len(saved()), 'tensors')
