import json
eps = episodes(cand='base')
for e in sorted(eps, key=lambda x: x['eid']):
    o = obs(e['eid'])
    G = np.array(o['gripper']).reshape(-1)
    A = np.array(o['actions'])
    st = np.array(o['step'])
    s = []
    for i in range(min(14, len(st))):
        s.append('%d:%.2f/%.2f>%.2f' % (st[i], G[i], A[i][0, 7], A[i][-1, 7]))
    print(e['eid'][-6:], 'OK' if e['success'] else 'F ', ' '.join(s))
