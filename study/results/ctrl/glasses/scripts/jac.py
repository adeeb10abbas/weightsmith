import json
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi/2), (0.0, 0.316, np.pi/2), (0.0825, 0.0, np.pi/2),
      (-0.0825, 0.384, -np.pi/2), (0.0, 0.0, np.pi/2), (0.088, 0.0, np.pi/2)]
def tf(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])
def fk(q):
    T = np.eye(4)
    for i, (a, d, al) in enumerate(DH):
        T = T @ tf(a, d, al, q[i])
    T = T @ tf(0, 0.107, 0, 0)
    T = T @ tf(0, 0.1034, 0, -np.pi/4)
    return T
def jac(q, eps=1e-5):
    T0 = fk(q); J = np.zeros((6, 7))
    for i in range(7):
        dq = np.zeros(7); dq[i] = eps
        T1 = fk(q + dq)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
    return J
it = json.load(open('/work/notes/items_a1.json'))
dqs = []
for item, lab in zip(it['items'], it['labels']):
    if lab != 'HOLD':
        continue
    o = obs(item['episode'])
    k = int(np.where(o['step'] <= item['step'])[0][-1])
    q = o['joint'][k]
    J = jac(q)
    dx = np.array([0, 0, 0.10, 0, 0, 0])
    dq = np.linalg.pinv(J) @ dx
    dqs.append(dq)
dqs = np.array(dqs)
print('dq for +0.10m z (rad), mean', np.round(dqs.mean(0), 3))
print('std', np.round(dqs.std(0), 3))
# check accuracy of mean dq at each config
errs = []
for item, lab in zip(it['items'], it['labels']):
    if lab != 'HOLD':
        continue
    o = obs(item['episode'])
    k = int(np.where(o['step'] <= item['step'])[0][-1])
    q = o['joint'][k]
    p0 = fk(q)[:3, 3]; p1 = fk(q + dqs.mean(0))[:3, 3]
    errs.append(p1 - p0)
print('effect of mean dq on xyz: mean', np.round(np.mean(errs, 0), 3), 'std', np.round(np.std(errs, 0), 3))
# are actions absolute targets? compare actions[k] with joint[k], joint[k+1]
o = obs('base.base.s70011')
for k in [3, 4, 8, 9]:
    print('k', k, 'joint', np.round(o['joint'][k], 3))
    print('   act0', np.round(o['actions'][k, 0, :7], 3))
    print('   act14', np.round(o['actions'][k, 14, :7], 3))
    print('   next joint', np.round(o['joint'][k + 1], 3))
# global ranges of actions per joint over all base episodes
A = np.concatenate([obs(e['eid'])['actions'].reshape(-1, 8) for e in episodes(cand='base')])
print('action q01', np.round(np.quantile(A, 0.01, axis=0), 3))
print('action q99', np.round(np.quantile(A, 0.99, axis=0), 3))
json.dump({'dq_10cm': dqs.mean(0).tolist()}, open('/work/notes/dq.json', 'w'))
