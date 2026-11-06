import json
rows = json.load(open('/work/notes/carry_rows.json'))
cur = None
for r in rows:
    if r['eid'] != cur:
        cur = r['eid']; print(cur[-6:], 'S' if r['succ'] else 'F')
    p, d = r['p0'], r['d14']
    print(f"   k{r['k']:2d} s{r['step']:3d} p=({p[0]:+.2f},{p[1]:+.2f},{p[2]:.2f}) d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) g={r['g']:.2f} cmdg={r['cmdg']:.2f}")
