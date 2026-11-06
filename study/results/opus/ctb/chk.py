# Release counts + phase-2 motion for a candidate on the 49 decisions (actions only).
import json
aid = json.load(open('/work/notes/steer_cfg.json'))['check_aid']
exec(open('/work/notes/fk.py').read())
_, B = acts('ctb-ph'); _, C = acts(aid)
groups = {'PREM': range(0, 21), 'CARRY': range(21, 30), 'P2app': range(30, 37), 'P2grasp': range(37, 42), 'P2carry': range(42, 49)}
for nm, X in [('base', B), (aid, C)]:
    cl = (X['actions'][:, :, 7] > 0.5).mean(1)
    print('%-16s releases: ' % nm + '  '.join('%s %d/%d' % (g, (cl[list(ix)] < 0.5).sum(), len(ix)) for g, ix in groups.items()))
eb = np.array([fk(B['actions'][i, -1, :7]) for i in range(49)]); ec = np.array([fk(C['actions'][i, -1, :7]) for i in range(49)])
d = np.linalg.norm(eb - ec, axis=1) * 100
print('chunk-end EE shift vs base (cm): ' + '  '.join('%s %.1f' % (g, d[list(ix)].mean()) for g, ix in groups.items()))
