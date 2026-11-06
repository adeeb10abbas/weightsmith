import json, os
import numpy as np

def dh(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1]])

P = [(0, 0.333, 0), (0, 0, -np.pi/2), (0, 0.316, np.pi/2), (0.0825, 0, np.pi/2),
     (-0.0825, 0.384, -np.pi/2), (0, 0, np.pi/2), (0.088, 0, np.pi/2)]

def fk(q, tip=0.107 + 0.1034):
    T = np.eye(4)
    for (a, d, al), th in zip(P, q[:7]):
        T = T @ dh(a, d, al, th)
    T = T @ dh(0, tip, 0, 0)
    return T[:3, 3]
