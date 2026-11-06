import json
idx, arr = acts('a1-states')
print('index keys', list(idx.keys())[:30])
print({k: np.asarray(v).shape for k, v in arr.items()})
it = json.load(open('/work/notes/items_a1.json'))
lab = np.array(it['labels'])
H = np.asarray(arr['action.head_input'], dtype=np.float64)  # N,S,15,1024
A = np.asarray(arr['actions'], dtype=np.float64)  # N,15,8
W = base('action.action_out_proj.kernel').astype(np.float64)
b = base('action.action_out_proj.bias').astype(np.float64)
N, S, T, D = H.shape
V = H @ W + b  # N,S,15,32
Y = -0.1 * V.sum(1)  # = x0 - x1 (normalized), N,15,32
# calibrate normalization per dim: Y = (A - o)/s - x1  -> regress Y on A
cal = []
for j in range(8):
    a = A[:, :, j].ravel(); y = Y[:, :, j].ravel()
    M = np.stack([a, np.ones_like(a)], 1)
    coef, res, _, _ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ coef
    s = 1.0 / coef[0]; o = -coef[1] * s
    cal.append((s, o))
    print('dim', j, 'scale(rad per norm unit) %.4f offset %.4f resid std %.3f' % (s, o, resid.std()))
# compare with quantile guess
q01 = np.array([-0.337, -0.556, -0.295, -2.728, -1.136, 1.906, -0.743, 0.])
q99 = np.array([0.533, 0.631, 0.341, -1.626, 0.242, 2.8, 1.041, 0.988])
print('quantile-guess scale', np.round((q99 - q01) / 2, 3))
json.dump({'cal': cal}, open('/work/notes/cal.json', 'w'))
# grip readout in rad space: check gripper threshold
# holding feature
Hm = H.mean(axis=(1, 2))  # N,1024
cats = ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']
mu = {c: Hm[lab == c].mean(0) for c in cats}
openm = Hm[(lab == 'OPENLOW') | (lab == 'OPENHIGH')].mean(0)
d = mu['HOLD'] - openm
a_hat = d / np.linalg.norm(d)
print('|HOLD-OPEN| =', np.linalg.norm(d), ' |H| typical', np.linalg.norm(Hm, axis=1).mean())
# per-item projection averaged over steps and tokens (this is what a linear readout edit sees)
P = np.einsum('nstd,d->nst', H, a_hat)  # N,S,T
Pm = P.mean(axis=(1, 2))
for c in cats:
    x = Pm[lab == c]
    print(c, 'P mean %.3f std %.3f min %.3f max %.3f' % (x.mean(), x.std(), x.min(), x.max()))
print('per-denoise-step mean P by cat')
for c in cats:
    print(c, np.round(P[lab == c].mean(axis=(0, 2)), 2))
print('per-token std of P within item (mean)', P.std(axis=2).mean())
# leave-one-episode-out robustness: direction from half the episodes
eps_ = np.array([x['episode'] for x in it['items']])
ue = sorted(set(eps_))
half = set(ue[::2])
m1 = np.array([e in half for e in eps_])
d1 = Hm[m1 & (lab == 'HOLD')].mean(0) - Hm[m1 & ((lab == 'OPENLOW') | (lab == 'OPENHIGH'))].mean(0)
d1 /= np.linalg.norm(d1)
P1 = np.einsum('nd,d->n', Hm, d1)
print('half-split direction, held-out items:')
for c in cats:
    x = P1[(~m1) & (lab == c)]
    print('  ', c, 'mean %.3f std %.3f min %.3f max %.3f' % (x.mean(), x.std(), x.min(), x.max()))
print('cos(full, half)', float(a_hat @ d1))
# print per-item with g
g = np.array([float(v) for v in idx.get('gripper', [np.nan] * N)]) if 'gripper' in idx else None
order = np.argsort(Pm)
for n in order:
    print('%-20s %4d %-8s P=%.3f' % (it['items'][n]['episode'][10:], it['items'][n]['step'], lab[n], Pm[n]),
          '' if g is None else 'g=%.2f' % g[n])
np.save('/work/notes/a_hat.npy', a_hat)
np.save('/work/notes/Pm_a1.npy', Pm)
