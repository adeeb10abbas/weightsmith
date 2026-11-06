# Target-identity direction u_L per VLM layer (image-token residual: cube-only prompt minus banana-only prompt, c2-multi),
# per-layer target-tag score for each prompt, and each attention head's write capacity along u_L (||W_O[h] u_L|| / ||W_O[h]||_F).
import json
idx, arr = acts('c2-multi')
P = ['orig', 'mirror', 'V2', 'cibtb', 'and', 'nodest', 'cube', 'banana']
IR = arr['vlm.img_resid_out'].astype(np.float64)        # [48, 18, cam, 17, F]
G = IR.reshape(8, 6, *IR.shape[1:])                     # [g, item, L, cam, cell, F]
cells = G[:, :, :, :, 1:]                               # drop camera-mean cell
U, out = {}, {}
for L in range(18):
    d = (cells[6, :, L] - cells[7, :, L]).reshape(-1, cells.shape[-1]).mean(0)
    u = d / np.linalg.norm(d)
    U[L] = u
    ref_c = (cells[6, :, L] @ u).mean(); ref_b = (cells[7, :, L] @ u).mean()
    sc = {P[g]: float(((cells[g, :, L] @ u).mean() - ref_b) / (ref_c - ref_b)) for g in range(6)}
    out[L] = sc
    print('L%2d  ' % L + '  '.join('%s=%.2f' % (k, v) for k, v in sc.items()))
np.save('/work/notes/u_img.npy', np.stack([U[L] for L in range(18)]))
print('\nhead write capacity along u_L (fraction of W_O energy), top heads per layer:')
cap = {}
for L in range(18):
    W = load('language.blocks.%02d.attn.attn_vec_einsum.w' % L)      # [8, 256, 2048]
    c = [float(np.linalg.norm(W[h] @ U[L]) / np.linalg.norm(W[h])) for h in range(8)]
    cap[L] = c
    print('L%2d ' % L + ' '.join('h%d=%.3f' % (h, x) for h, x in enumerate(c)))
json.dump({'score': out, 'cap': cap}, open('/work/notes/ctb_dirs.json', 'w'))
