i1, a1 = acts('a1-prompts')
i2, a2 = acts('a2-frontback')
H = np.array(a1['vlm.mlp_hidden'], dtype=np.float32)
R = np.array(a1['vlm.resid_out'], dtype=np.float32)
R2 = np.array(a2['vlm.resid_out'], dtype=np.float32)
Hi = np.array(a2['vlm.img_mlp_hidden'], dtype=np.float32)   # [24,5,2,17,16384]
L = i1['vlm_layers']
left = list(range(12)); right = list(range(12, 24)); front = list(range(12)); back = list(range(12, 24))
for li, l in enumerate(L):
    h = H[left, li, 10]
    oth = H[left, li][:, [s for s in range(28) if s != 10]].reshape(-1, 16384)
    img = Hi[:, li].reshape(-1, 16384)
    mh = h.mean(0); mn = h.min(0)
    score = mn - np.maximum(np.abs(oth).max(0), np.abs(img).max(0))
    top = np.argsort(-score)[:5]
    s = ' '.join(f'{u}:{mh[u]:.2f}/min{mn[u]:.2f}/oth{np.abs(oth[:,u]).max():.2f}/img{np.abs(img[:,u]).max():.2f}' for u in top)
    dLR = (R[left, li, 10] - R[right, li, 10]); dFB = (R2[front, li, 10] - R2[back, li, 10])
    cLR = np.mean([np.dot(x, dLR.mean(0)) / np.linalg.norm(x) / np.linalg.norm(dLR.mean(0)) for x in dLR])
    cFB = np.mean([np.dot(x, dFB.mean(0)) / np.linalg.norm(x) / np.linalg.norm(dFB.mean(0)) for x in dFB])
    print(f'L{l}: |r10| {np.linalg.norm(R[left, li, 10], axis=1).mean():.0f} |L-R| {np.linalg.norm(dLR.mean(0)):.0f} (cons {cLR:.3f}) |F-B| {np.linalg.norm(dFB.mean(0)):.0f} (cons {cFB:.3f})')
    print('   top selective units:', s)
