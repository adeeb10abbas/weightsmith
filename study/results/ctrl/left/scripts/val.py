import json, sys
AID = 'a4-p1l8val'
def dh(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
P = [(0, 0.333, 0), (0, 0, -np.pi/2), (0, 0.316, np.pi/2), (0.0825, 0, np.pi/2),
     (-0.0825, 0.384, -np.pi/2), (0, 0, np.pi/2), (0.088, 0, np.pi/2)]
def fk(q, tip=0.2104):
    T = np.eye(4)
    for (a, d, al), th in zip(P, q[:7]):
        T = T @ dh(a, d, al, th)
    T = T @ dh(0, tip, 0, 0)
    return T[:3, 3]
idx, arr = acts(AID)
A = np.array(arr['actions']); RA = np.array(arr['recorded_actions'])
cache = {}
groups = {'target': range(0, 12), 'approach': range(12, 24), 'succ-carry': range(24, len(idx['items']))}
out = []
for g, rng in groups.items():
    print('==', g)
    dsum = []
    for i in rng:
        it = idx['items'][i]; e = it['episode']
        if e not in cache: cache[e] = obs(e)
        J = np.array(cache[e]['joint'])[it['k']]
        p0 = fk(J)
        db = fk(RA[i, 14]) - p0; dc = fk(A[i, 14]) - p0
        gb = RA[i, :, 7].mean(); gc = A[i, :, 7].mean()
        dsum.append((db, dc))
        print(f"  {e[-6:]} k{it['k']:2d} base d=({db[0]:+.3f},{db[1]:+.3f},{db[2]:+.3f}) g{gb:.2f} | cand d=({dc[0]:+.3f},{dc[1]:+.3f},{dc[2]:+.3f}) g{gc:.2f}  |diff|={np.linalg.norm(dc-db):.3f}")
    D = np.array(dsum)
    print(f"  mean base ({D[:,0,0].mean():+.3f},{D[:,0,1].mean():+.3f}) cand ({D[:,1,0].mean():+.3f},{D[:,1,1].mean():+.3f}) mean|diff| {np.linalg.norm(D[:,1]-D[:,0],axis=1).mean():.3f}")
print('parity', idx.get('base_parity_max_abs_action_error'))
