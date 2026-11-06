import json
o = obs("base.base.s70004")
for k in [0]:
    im = o['base_img'][k].astype(np.float32)
    print('img stats', im.mean((0, 1)), im.min(), im.max())
    # row/col mean brightness to detect padding
    print('row means', im.mean((1, 2))[::8].round(0))
    print('col means', im.mean((0, 2))[::8].round(0))
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    score = np.minimum(R, G) - B - 0.5 * np.abs(R - G)
    P = np.zeros((16, 16))
    for r in range(16):
        for c in range(16):
            P[r, c] = np.sort(score[r*14:(r+1)*14, c*14:(c+1)*14].ravel())[-20:].mean()
    for r in range(16):
        print('   ', ' '.join(f"{v:4.0f}" for v in P[r]))
    order = np.argsort(P.ravel())[::-1][:8]
    for f in order:
        r, c = divmod(f, 16)
        blk = im[r*14:(r+1)*14, c*14:(c+1)*14].reshape(-1, 3)
        s = score[r*14:(r+1)*14, c*14:(c+1)*14].ravel()
        print('patch', r, c, 'score', P[r, c].round(0), 'top col', blk[np.argsort(s)[-20:]].mean(0).round(0))
    redsc = R - np.maximum(G, B)
    P2 = np.zeros((16, 16))
    for r in range(16):
        for c in range(16):
            P2[r, c] = np.sort(redsc[r*14:(r+1)*14, c*14:(c+1)*14].ravel())[-20:].mean()
    order = np.argsort(P2.ravel())[::-1][:6]
    for f in order:
        r, c = divmod(f, 16)
        print('red patch', r, c, P2[r, c].round(0))
