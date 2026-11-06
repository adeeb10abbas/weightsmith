# shared FK helper; saved to /work/notes/fk_lib.py so other scripts can exec it
src = r'''
import numpy as np
def _tf(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st*ca, ct*ca, -sa, -d*sa],
                     [st*sa, ct*sa, ca, d*ca],
                     [0, 0, 0, 1.0]])
_DH = [(0,0.333,0),(0,0,-np.pi/2),(0,0.316,np.pi/2),(0.0825,0,np.pi/2),(-0.0825,0.384,-np.pi/2),(0,0,np.pi/2),(0.088,0,np.pi/2)]
def fk(q, tip=0.257):
    T = np.eye(4)
    for (a,d,al),th in zip(_DH, q[:7]):
        T = T @ _tf(a,d,al,th)
    T = T @ _tf(0, tip, 0, 0)
    return T[:3,3], T[:3,:3]
'''
open('/work/notes/fk_lib.py','w').write(src)
exec(src)
print(fk(np.zeros(7)))
