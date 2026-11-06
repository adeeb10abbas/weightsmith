import json, sys
D = json.load(open('/work/notes/base_traj.json'))
want = {'base.base.s70001': (150, 400), 'base.base.s70002': (150, 230), 'base.base.s70028': (150, 260),
        'base.base.s70000': (60, 160), 'base.base.s70016': (40, 130), 'base.base.s70026': (90, 440), 'base.base.s70011': (40, 200)}
for eid, (a, b) in want.items():
    d = D[eid]
    print(eid, d['success'])
    for i, s in enumerate(d['step']):
        if a <= s <= b:
            x = d['xyz'][i]
            print(f"  {s:4d} ({x[0]:+.3f},{x[1]:+.3f},{x[2]:.3f}) g={d['grip'][i]:.2f}")
