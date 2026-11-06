import numpy as np
_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
_AL = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
def _tf(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
def fk(q, tip=0.257):
    T = np.eye(4)
    for i in range(7):
        T = T @ _tf(_A[i], _D[i], _AL[i], q[i])
    T = T @ _tf(0, tip, 0, 0)
    return T[:3, 3]

idx, arr = acts('a1-prompts')
print('index keys', list(idx.keys())[:40])
for k, v in arr.items():
    print(k, getattr(v, 'shape', None))
for k in idx:
    if 'piece' in k:
        print(k, idx[k])
A = arr['actions']
N = A.shape[0]
npt = 20
names = ['base', 'veg box', 'box of frozen']
items = idx.get('items')
print('items sample', items[:2] if items is not None else None)
for j in range(npt):
    it = items[j]
    o = obs(it['episode'])
    k = it['k']
    q = o['joint'][k]
    rec = o['actions'][k]
    line = f"{it['episode'][-6:]} k{k:2d} gobs={o['gripper'][k,0]:.2f} rec: dq6={rec[-1,6]-q[6]:+.2f} gfr={np.mean(rec[:,7]>0.5):.2f} | "
    for p in range(3):
        a = A[p * npt + j]
        z = fk(a[-1, :7])[2]
        line += f"{names[p]}: dq6={a[-1,6]-q[6]:+.2f} g=[{a[0,7]:.2f} {a[7,7]:.2f} {a[14,7]:.2f}] z14={z:.3f} | "
    print(line)
