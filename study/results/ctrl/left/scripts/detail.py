import json
D = json.load(open('/work/notes/base_traj.json'))
want = {'base.base.s70007': (60, 200), 'base.base.s70028': (40, 260), 'base.base.s70001': (90, 400),
        'base.base.s70000': (40, 160), 'base.base.s70016': (40, 130), 'base.base.s70002': (150, 230),
        'base.base.s70005': (140, 330), 'base.base.s70013': (40, 200)}
for eid, (a, b) in want.items():
    d = D[eid]
    o = obs(eid)
    A = np.array(o['actions'])
    print(eid, 'success', d['success'], 'actions shape', A.shape)
    for i, s in enumerate(d['step']):
        if a <= s <= b:
            x = d['xyz'][i]
            print(f"  {s:4d} xyz=({x[0]:+.3f},{x[1]:+.3f},{x[2]:.3f}) g={d['grip'][i]:.2f} cmdg={A[i,:,7].mean():.2f} [{A[i,0,7]:.2f}..{A[i,-1,7]:.2f}]")
