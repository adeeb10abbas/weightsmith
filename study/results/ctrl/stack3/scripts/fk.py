import numpy as np, json
# Franka Panda modified DH (Craig)
A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
D = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
AL = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
TIP = 0.1034

def _T(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a],
                     [st*ca, ct*ca, -sa, -d*sa],
                     [st*sa, ct*sa, ca, d*ca],
                     [0, 0, 0, 1]])

def fk(q):
    T = np.eye(4)
    for i in range(7):
        T = T @ _T(A[i], D[i], AL[i], q[i])
    T = T @ _T(A[7], D[7], AL[7], 0.0)
    T = T @ _T(0, TIP, 0, 0.0)
    return T[:3, 3]
