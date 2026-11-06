# Target-tag score (0 = banana-only prompt, 1 = cube-only prompt; refs from c2-multi) for each decision group under the
# ORIGINAL prompt, per VLM layer; plus phase-1 vs phase-2 separability of image-token MLP hidden (cosine of means).
_, C2 = acts('c2-multi'); _, A = acts('ctb-ph')
U = np.load('/work/notes/u_img.npy')
c2 = C2['vlm.img_resid_out'].astype(np.float64).reshape(8, 6, 18, 2, 17, -1)[:, :, :, :, 1:]
lay = [10, 11, 12, 13, 14, 15, 16]
R = A['vlm.img_resid_out'].astype(np.float64)[:, :, :, 1:]       # [49, 7, cam, 16, F]
H = A['vlm.img_mlp_hidden'].astype(np.float32)[:, :, :, 1:]      # [49, 7, cam, 16, 16384]
groups = {'PREM': range(0, 21), 'CARRY': range(21, 30), 'P2app': range(30, 37), 'P2grasp': range(37, 42), 'P2carry': range(42, 49)}
act = A['actions']; close = (act[:, :, 7] > 0.5).mean(1)
print('close frac (release if <0.5):', {g: '%d/%d release' % ((close[list(ix)] < 0.5).sum(), len(ix)) for g, ix in groups.items()})
for j, L in enumerate(lay):
    u = U[L]
    rc = (c2[6, :, L] @ u).mean(); rb = (c2[7, :, L] @ u).mean()
    s = (R[:, j] @ u).mean((1, 2))
    s = (s - rb) / (rc - rb)
    print('L%2d score  ' % L + '  '.join('%s %.2f±%.2f' % (g, s[list(ix)].mean(), s[list(ix)].std()) for g, ix in groups.items()))
print()
for j, L in enumerate(lay):
    h = H[:, j].mean((1, 2))                                          # [49, 16384] mean over cams & cells
    m1 = h[:30].mean(0); m2 = h[30:].mean(0); k = m1 - m2
    p = h @ k / (k @ k)                                               # projection on phase key
    print('L%2d phase key proj  ph1 %.2f±%.2f  ph2 %.2f±%.2f   |m1|=%.1f |k|=%.1f' % (L, p[:30].mean(), p[:30].std(), p[30:].mean(), p[30:].std(), np.linalg.norm(m1), np.linalg.norm(k)))
