import numpy as np, sys
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

for eid in ['base.base.s70005', 'base.base.s70018', 'base.base.s70024', 'base.base.s70025']:
    o = obs(eid)
    J = o['joint']; G = o['gripper'][:, 0]; A = o['actions']; st = o['step']
    print(eid)
    for i in range(len(J)):
        p, R = fk(J[i])
        pa, _ = fk(A[i, -1, :7])
        g7 = A[i, :, 7]
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"  k{i:2d} s{st[i]:3d} xyz=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) yaw={yaw:+5.0f} q6={J[i,6]:+.2f}->{A[i,-1,6]:+.2f} gobs={G[i]:.2f} zA14={pa[2]:.3f} grip=[{g7[0]:.2f} {g7[4]:.2f} {g7[9]:.2f} {g7[14]:.2f}]")
