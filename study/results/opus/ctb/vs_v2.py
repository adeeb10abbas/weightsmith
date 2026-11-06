exec(open('/work/notes/fk.py').read())
_, B = acts('a1-orig'); _, V = acts('a1-v2'); _, C = acts('chk-v2mi-f')
bowl = np.array([0.455, 0.11])
for nm, X in [('orig', B), ('V2 prompt', V), ('v2mi-f edit', C)]:
    A = X['actions'][:30]
    e = np.array([fk(A[i, -1, :7]) for i in range(30)])
    cl = (A[:, :, 7] > 0.5).mean(1)
    dist = np.linalg.norm(e[:, :2] - bowl, axis=1) * 100
    print('%-12s PREM: release %2d/21  end-dist-to-bowl %.1f cm  end-z %.3f | CARRY: release %d/9 dist %.1f cm' % (nm, (cl[:21] < .5).sum(), dist[:21].mean(), e[:21, 2].mean(), (cl[21:] < .5).sum(), dist[21:].mean()))
ev = np.array([fk(V['actions'][i, -1, :7]) for i in range(30)]); ec = np.array([fk(C['actions'][i, -1, :7]) for i in range(30)]); eo = np.array([fk(B['actions'][i, -1, :7]) for i in range(30)])
print('chunk-end distance: edit vs V2 %.1f cm | orig vs V2 %.1f cm' % (np.linalg.norm(ec - ev, axis=1).mean() * 100, np.linalg.norm(eo - ev, axis=1).mean() * 100))
