# v2p: image-cell binding edit WITH preservation keys (MEMIT-style).
# HYPOTHESIS (before eval): v2mi-f's side effect on other tasks (butter-on-raisin 14/16 -> 10/16) comes from its keys
# also firing on other tasks' image tokens. Requiring the update to map other tasks' mean image-cell activations to
# zero keeps the cube-then-banana fix (image-only keys alone already give 0/21 releases) while leaving other tasks alone.
#   K = [K_target (32 cells, c2-multi orig); K_preserve (32 cells x tasks)],  V = [V_target; 0]
#   dW = K^T (K K^T + lam I)^-1 V
import json
CFG = json.load(open('/work/notes/steer_cfg.json')); LAYERS = CFG['layers']; ALPHA = CFG['alpha']; LAM = CFG.get('lam', 1e-3)
PRES = CFG.get('preserve', [])
_, X = acts('c2-multi')
IR = X['vlm.img_resid_out'].astype(np.float64); IH = X['vlm.img_mlp_hidden'].astype(np.float64)
D = {L: np.stack([(IR[12:18, L, c, e] - IR[0:6, L, c, e]).mean(0) for c in range(2) for e in range(1, 17)]) for L in range(17)}
PH = {}
for aid in PRES:
    _, Y = acts(aid)
    PH[aid] = Y['vlm.img_mlp_hidden']          # [N, 17 layers (0..16), cam, 17, F]
for L in LAYERS:
    Vt = ALPHA * (D[L] - (D[L - 1] if L > 0 else 0))
    Kt = np.stack([IH[0:6, L, c, e].mean(0) for c in range(2) for e in range(1, 17)])
    Kp = [np.stack([PH[a][:, L, c, e].astype(np.float64).mean(0) for c in range(2) for e in range(1, 17)]) for a in PRES]
    K = np.concatenate([Kt] + Kp); V = np.concatenate([Vt] + [np.zeros_like(Vt) for _ in Kp])
    G = K @ K.T; lam = LAM * np.trace(G) / len(G)
    dW = K.T @ np.linalg.solve(G + lam * np.eye(len(G)), V)
    name = 'language.blocks.%02d.mlp.linear' % L
    W = load(name).copy(); save(name, bf16(W + dW))
    et = np.linalg.norm(Kt @ dW - Vt) / np.linalg.norm(Vt)
    ep = max([np.linalg.norm(k @ dW) for k in Kp] or [0]) / np.linalg.norm(Vt)
    print('L%2d keys %d  target rel-err %.3f  preserve leak %.3f  |dW|/|W| %.4f' % (L, len(K), et, ep, np.linalg.norm(dW) / np.linalg.norm(W)))
print('saved', len(saved()), 'tensors')
