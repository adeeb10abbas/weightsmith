import json

def fk(q, tip=0.257):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, tip]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]],
                      [st*ca, ct*ca, -sa, -d[i]*sa],
                      [st*sa, ct*sa, ca, d[i]*ca],
                      [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

eps = episodes(cand="base")
o = obs(eps[0]['eid'])
print('steps', o['step'][:10])
print('joint0', o['joint'][0], 'grip', o['gripper'][:5,0])
print('act0 first', o['actions'][0][0], 'last', o['actions'][0][-1])
print('joint1', o['joint'][1])
out = {}
for e in eps:
    eid = e['eid']
    o = obs(eid)
    P = np.array([fk(q) for q in o['joint']])
    out[eid] = dict(step=o['step'].tolist(), xyz=P.round(3).tolist(), grip=o['gripper'][:,0].round(3).tolist(),
                    gcmd=o['actions'][:,:,7].mean(1).round(2).tolist(), success=bool(e['success']))
    line = []
    for k in range(min(len(P), 60)):
        line.append(f"{o['step'][k]}:({P[k,0]:.2f},{P[k,1]:.2f},{P[k,2]:.2f}|{o['gripper'][k,0]:.2f})")
    print(eid, 'OK' if e['success'] else 'FAIL', len(P))
    print('   ', ' '.join(line[:24]))
json.dump(out, open('/work/notes/traj_base.json', 'w'))
