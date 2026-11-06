# "v2-state" edit. HYPOTHESIS (before eval): the explicit instruction changes the state/prompt-tail tokens' residual
# trajectory (which the action expert attends to, 33-58% of its attention mass in L8-16), and that is what keeps the
# held cube. Reproduce V2's per-layer increment at the aligned state tokens with one rank-1 update per VLM MLP:
#   v_L = d_L - d_{L-1},  d_L = mean over items & aligned state slots of resid_V2 - resid_orig at layer L output
#   key k_L = mean orig state-slot MLP hidden, orthogonalised against the mean image-token hidden (so image tokens ~0),
#   W_down_L += k_L / (m_state . k_L) (x) (ALPHA * v_L)
# PREDICTION: PREM releases 21/21 -> well under half; CARRY 0/9 and phase-2 unchanged.
import json
CFG = json.load(open('/work/notes/steer_cfg.json')); LAYERS = CFG['layers']; ALPHA = CFG['alpha']
idx, X = acts('c2-multi')
VR = X['vlm.resid_out'].astype(np.float64); VH = X['vlm.mlp_hidden'].astype(np.float64); IH = X['vlm.img_mlp_hidden'].astype(np.float64)
O, V = slice(0, 6), slice(12, 18)
so, sv, n = 13, 17, 37           # 'State' token index in orig / V2, and number of aligned slots through 'Action', ':'
# alignment check at layer 0
cs = [float(np.mean([VR[i, 0, so + j] @ VR[12 + i, 0, sv + j] / np.linalg.norm(VR[i, 0, so + j]) / np.linalg.norm(VR[12 + i, 0, sv + j]) for i in range(6)])) for j in range(n)]
print('alignment cos at L0 (aligned slots): min %.3f mean %.3f' % (min(cs), np.mean(cs)))
cs_bad = float(np.mean([VR[i, 0, so + 5] @ VR[12 + i, 0, so + 5] / np.linalg.norm(VR[i, 0, so + 5]) / np.linalg.norm(VR[12 + i, 0, so + 5]) for i in range(6)]))
print('  (misaligned control cos %.3f)' % cs_bad)
d = {L: (VR[V, L, sv:sv + n] - VR[O, L, so:so + n]).mean((0, 1)) for L in range(18)}
for L in LAYERS:
    v = ALPHA * (d[L] - (d[L - 1] if L > 0 else 0))
    ms = VH[O, L, so:so + n].mean((0, 1)); mi = IH[O, L, :, 1:].mean((0, 1, 2))
    k = ms - (ms @ mi) / (mi @ mi) * mi
    name = 'language.blocks.%02d.mlp.linear' % L
    W = load(name).copy()
    W = W + np.outer(k / (ms @ k), v)
    save(name, bf16(W))
    print('L%2d |v|=%.2f |d_L|=%.2f  key: state %.2f image %.2f' % (L, np.linalg.norm(v), np.linalg.norm(d[L]), (ms @ k) / (ms @ k), (mi @ k) / (ms @ k)))
print('saved', len(saved()), 'tensors')
