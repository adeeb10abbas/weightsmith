import json
idx, arr = acts("a1-prompts")
AM = arr['action.attn_map']  # N,S,La,cam,16,16
M = arr['action.attn_mass']  # N,S,La,8,5
print(AM.shape)
decs = [("base.base.s70004", 1), ("base.base.s70013", 1), ("base.base.s70028", 1), ("base.base.s70030", 1),
        ("base.base.s70004", 10), ("base.base.s70013", 3), ("base.base.s70020", 3), ("base.base.s70028", 5)]
La = idx['action_layers']

def yellow_patches(im):
    im = im.astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    score = np.minimum(R, G) - B - 0.5 * np.abs(R - G)
    P = np.zeros((16, 16))
    for r in range(16):
        for c in range(16):
            P[r, c] = np.sort(score[r*14:(r+1)*14, c*14:(c+1)*14].ravel())[-30:].mean()
    return P

for i in range(8):
    e, k = decs[i]
    o = obs(e)
    for cam, key in [(0, 'base_img'), (1, 'wrist_img')]:
        Y = yellow_patches(o[key][k])
        ymask = Y > 60
        print(e[-2:], k, 'cam', cam, 'yellow patches', [(r, c, int(Y[r, c])) for r in range(16) for c in range(16) if ymask[r, c]])
        for li, L in enumerate(La):
            a = AM[i, 0, li, cam]  # S=0 (t=1.0)
            tot = a.sum()
            top = np.argsort(a.ravel())[::-1][:4]
            ys = a[ymask].sum()
            print(f"   L{L:2d} tot={tot:.3f} yellowmass={ys:.4f} ({ys/max(tot,1e-9):.3f}) top=" +
                  ' '.join(f"({t//16},{t%16}):{a.ravel()[t]:.4f}" for t in top))
