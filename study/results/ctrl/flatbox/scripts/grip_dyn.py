import numpy as np
from collections import defaultdict
d = defaultdict(list)
for e in episodes(cand='base'):
    o = obs(e['eid'])
    G = o['gripper'][:, 0]; A = o['actions']
    for i in range(1, len(G)):
        if G[i - 1] < 0.02:
            c = A[i - 1, :, 7] > 0.5
            # count trailing close steps
            n = 0
            for v in c[::-1]:
                if v: n += 1
                else: break
            tot = int(c.sum())
            d[(n, tot)].append(round(float(G[i]), 2))
for k in sorted(d):
    print(k, d[k][:12])
