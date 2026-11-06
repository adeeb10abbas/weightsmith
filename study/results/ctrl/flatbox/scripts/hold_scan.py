import json, numpy as np
rows = json.load(open('/work/notes/base_rows.json'))
print([n for n in NAMES if not n.startswith('vision') and not n.startswith('language.blocks') and not n.startswith('action.blocks')])
for n in NAMES:
    if 'action_out' in n or 'state' in n:
        print(n, stats(n))
by = {}
for r in rows:
    by.setdefault(r['eid'], []).append(r)
nhold = 0
for eid, rs in by.items():
    hold = [r for r in rs if 0.1 < r['gobs'] < 0.6]
    if not hold:
        continue
    nhold += 1
    print(eid, 'n_hold_replans', len(hold))
    for r in rs:
        if r['k'] < min(h['k'] for h in hold) - 2:
            continue
        if r['k'] > max(h['k'] for h in hold) + 2:
            continue
        print(f"   k{r['k']:2d} st{r['step']:3d} z{r['z']:.3f} yaw{r['yaw']:+5.0f} gobs{r['gobs']:.2f} g0{r['g0']:.2f} gfrac{r['gfrac']:.2f} glast{r['glast']:.2f}")
print('episodes with hold', nhold)
