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
    return T[:3, 3], T[:3, :3]

for eid in ['base.base.s70012', 'base.base.s70007', 'base.base.s70006', 'base.base.s70002']:
    o = obs(eid)
    J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']; st = o['step']
    print(eid)
    print('  action joint range', A[:, :, :7].min(), A[:, :, :7].max(), 'grip act range', A[:, :, 7].min(), A[:, :, 7].max())
    for i in range(len(J)):
        p, R = fk(J[i])
        pa, _ = fk(A[i, -1, :7])
        pm, _ = fk(A[i, 7, :7])
        g7 = A[i, :, 7]
        print(f"  k{i:2d} s{st[i]:3d} xyz=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) gobs={G[i]:.2f} zA7={pm[2]:.3f} zA14={pa[2]:.3f} dxy14=({pa[0]-p[0]:+.3f},{pa[1]-p[1]:+.3f}) grip=[{g7[0]:.2f} {g7[4]:.2f} {g7[9]:.2f} {g7[14]:.2f}] zax={R[2,2]:+.2f}")
