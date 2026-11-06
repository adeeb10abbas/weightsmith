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
AID = 'v1-p1'
idx, arr = acts(AID)
A = np.asarray(arr['actions'], dtype=np.float64)
RA = np.asarray(arr['recorded_actions'], dtype=np.float64)
Mh = np.asarray(arr['action.mlp_hidden'], dtype=np.float64)
unit = json.load(open('/work/notes/p1_meta.json'))['unit']
u = Mh[:, 0, 0, 0, unit]
N = A.shape[0]
labs = ['HOLDC'] * 16 + ['EMPTY'] * 4 + ['CLOSING'] * 30 + ['OPENLOW'] * 6 + ['OPENHIGH'] * 6
items = idx['items']
res = {}
for n in range(N):
    zb = fk(RA[n, 14, :7])[2, 3]; zp = fk(A[n, 14, :7])[2, 3]
    pb = fk(RA[n, 14, :7])[:3, 3]; pp = fk(A[n, 14, :7])[:3, 3]
    gb = (RA[n, :, 7] > 0.5).sum(); gp = (A[n, :, 7] > 0.5).sum()
    dq = np.abs(A[n, :, :7] - RA[n, :, :7]).max()
    it = items[n]
    ep = it.get('episode', '?') if isinstance(it, dict) else str(it)
    st = it.get('step', '?') if isinstance(it, dict) else ''
    print('%-8s %-18s %4s unit=%6.2f dz=%+.3f dxy=%.3f max|dq|=%.3f grip_close_tokens base=%2d p1=%2d' % (
        labs[n], str(ep)[10:], st, u[n], zp - zb, np.linalg.norm((pp - pb)[:2]), dq, gb, gp))
    res.setdefault(labs[n], []).append((zp - zb, dq, gb, gp))
for k, v in res.items():
    v = np.array(v)
    print('%-8s n=%d dz mean %+.3f min %+.3f max %+.3f | max|dq| mean %.3f | grip flips %d' % (
        k, len(v), v[:, 0].mean(), v[:, 0].min(), v[:, 0].max(), v[:, 1].mean(), int((np.abs(v[:, 2] - v[:, 3]) >= 4).sum())))
