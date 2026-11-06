# Rank-1 "binding steer" edit. HYPOTHESIS (before eval): the explicit instruction raises the image-token target-tag
# (cube-vs-banana axis u_L) by ~0.4 in VLM L10-12, and that shift is what keeps the held cube. Adding the same shift,
# keyed on the mean phase-1 image-token MLP activation, should make the original instruction behave like the explicit
# one: PREM releases 21/21 -> few, CARRY 0/9 unchanged, phase-2 approach/grasp/carry unchanged.
# Edit: W_down_L += k_hat (x) v_L, with k_hat = m1 / |m1|^2 (m1 = mean phase-1 image hidden at layer L) and
# v_L = ALPHA * (ref_cube - ref_banana) * u_L, so a token with hidden == m1 gets exactly +ALPHA target-tag units at L.
import json, os
CFG = json.load(open('/work/notes/steer_cfg.json'))
LAYERS, ALPHA = CFG['layers'], CFG['alpha']
_, C2 = acts('c2-multi'); _, A = acts('ctb-ph')
U = np.load('/work/notes/u_img.npy')
c2 = C2['vlm.img_resid_out'].astype(np.float64).reshape(8, 6, 18, 2, 17, -1)[:, :, :, :, 1:]
lay = [10, 11, 12, 13, 14, 15, 16]
H = A['vlm.img_mlp_hidden'].astype(np.float64)[:, :, :, 1:]
for L in LAYERS:
    j = lay.index(L)
    m1 = H[:30, j].mean((0, 1, 2))
    u = U[L]; scale = float((c2[6, :, L] @ u).mean() - (c2[7, :, L] @ u).mean())
    v = ALPHA * scale * u
    name = 'language.blocks.%02d.mlp.linear' % L
    W = base(name).copy()                       # [16384, 2048]
    W = W + np.outer(m1 / (m1 @ m1), v)
    save(name, bf16(W))
    print('L%d: |v|=%.3f scale=%.3f, rows touched %d' % (L, np.linalg.norm(v), scale, int((np.abs(m1) > 0).sum())))
print('saved', saved())
