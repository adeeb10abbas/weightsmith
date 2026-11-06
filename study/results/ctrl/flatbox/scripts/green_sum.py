import numpy as np, json
G = json.load(open('/work/notes/green_geom.json'))
for e in episodes(cand='base'):
    eid = e['eid']
    o = obs(eid)
    q6 = o['joint'][:, 6]
    rows = G[eid]
    r0 = rows[0]
    angs = [r[8] for r in rows if len(r) > 2]
    el = [r[9] for r in rows if len(r) > 2]
    gi = o['gripper'][:, 0]
    mid = [(int(o['step'][i]), round(float(gi[i]), 2)) for i in range(len(gi)) if 0.1 < gi[i] < 0.9]
    print(eid[-6:], 'ang0', r0[8] if len(r0) > 2 else None, 'el0', r0[9] if len(r0) > 2 else None,
          'bbox0 w', (r0[5] - r0[4]) if len(r0) > 2 else None, 'h', (r0[7] - r0[6]) if len(r0) > 2 else None,
          '| q6 min %.2f max %.2f end %.2f' % (q6.min(), q6.max(), q6[-1]), '| mid-g', mid[:6])
