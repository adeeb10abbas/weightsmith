# "v2-multi" edit. HYPOTHESIS (before eval): the release decision reads per-token structure of the prefix tail
# (bos/'Task:' and the state..'Action:' slots), not just its mean. Reproduce V2's per-layer increment at each of 40
# aligned slots with a closed-form least-norm multi-key update per VLM MLP (MEMIT-style, no optimisation):
#   for each slot s: key K_s = mean orig MLP hidden at s (6 PREM items), value V_s = ALPHA*(d_L[s] - d_{L-1}[s]),
#   d_L[s] = mean_items(resid_V2[s'] - resid_orig[s]) at layer L output (s' = aligned V2 slot)
#   dW = K^T (K K^T + lam I)^-1 V    (exact on the 40 mean keys; small ridge for conditioning)
# PREDICTION: PREM releases drop well below 21/21; CARRY 0/9 and phase-2 unchanged.
import json
CFG = json.load(open('/work/notes/steer_cfg.json')); LAYERS = CFG['layers']; ALPHA = CFG['alpha']; LAM = CFG.get('lam', 1e-3)
_, X = acts('c2-multi')
VR = X['vlm.resid_out'].astype(np.float64); VH = X['vlm.mlp_hidden'].astype(np.float64)
so, sv, n = 13, 17, 37
oslots = [0, 1, 2] + list(range(so, so + n)); vslots = [0, 1, 2] + list(range(sv, sv + n))
d = {L: np.stack([(VR[12:18, L, b] - VR[0:6, L, a]).mean(0) for a, b in zip(oslots, vslots)]) for L in range(18)}
for L in LAYERS:
    V = ALPHA * (d[L] - (d[L - 1] if L > 0 else 0))                     # [40, 2048]
    K = np.stack([VH[0:6, L, a].mean(0) for a in oslots])               # [40, 16384]
    G = K @ K.T; lam = LAM * np.trace(G) / len(G)
    dW = K.T @ np.linalg.solve(G + lam * np.eye(len(G)), V)
    name = 'language.blocks.%02d.mlp.linear' % L
    W = load(name).copy(); W = W + dW
    save(name, bf16(W))
    err = np.linalg.norm(K @ dW - V) / np.linalg.norm(V)
    print('L%2d |V|=%.2f fit-rel-err %.3f |dW|/|W| %.4f' % (L, np.linalg.norm(V), err, np.linalg.norm(dW) / np.linalg.norm(W)))
print('saved', len(saved()), 'tensors')
