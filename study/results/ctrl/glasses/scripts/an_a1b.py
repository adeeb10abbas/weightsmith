import json
idx, arr = acts('a1-states')
it = json.load(open('/work/notes/items_a1.json'))
lab = np.array(it['labels'])
H = np.asarray(arr['action.head_input'], dtype=np.float64)  # N,S,15,1024
A = np.asarray(arr['actions'], dtype=np.float64)
RA = np.asarray(arr['recorded_actions'], dtype=np.float64)
M = np.asarray(arr['action.mlp_hidden'], dtype=np.float64)[:, :, 0, 0, :]  # N,S,4096
R = np.asarray(arr['action.resid_out'], dtype=np.float64)[:, :, 0, 0, :]  # N,S,1024
W = base('action.action_out_proj.kernel').astype(np.float64)
b = base('action.action_out_proj.bias').astype(np.float64)
N = H.shape[0]
err = np.abs(A - RA).max(axis=(1, 2))
print('parity per item: median %.4f max %.3f; n>0.05: %d' % (np.median(err), err.max(), (err > 0.05).sum()))
print('  worst dims', np.abs(A - RA).max(axis=(0, 1)).round(3))
# joint state per item
Q = []; G = []; Z = []
for x in it['items']:
    o = obs(x['episode']); k = int(np.where(o['step'] <= x['step'])[0][-1])
    Q.append(o['joint'][k]); G.append(o['gripper'][k, 0])
Q = np.array(Q); G = np.array(G)
V = H @ W + b
Y = -0.1 * V.sum(1)  # x0 - x1
cal = []
for j in range(7):
    a = (A[:, :, j] - Q[:, None, j]).ravel(); y = Y[:, :, j].ravel()
    Mx = np.stack([a, np.ones_like(a)], 1)
    coef = np.linalg.lstsq(Mx, y, rcond=None)[0]
    s = 1.0 / coef[0]; o_ = -coef[1] * s
    r2 = 1 - ((y - Mx @ coef) ** 2).mean() / y.var()
    cal.append((s, o_))
    print('delta dim', j, 'scale %.4f rad/unit offset %.4f  r2 %.3f  delta std %.3f' % (s, o_, r2, a.std()))
json.dump({'cal_delta': cal}, open('/work/notes/cal.json', 'w'))
# ---- unit selectivity in block 17 MLP
hold = lab == 'HOLD'
keep = RA[:, :, 7].mean(1) > 0.5
HC = hold & keep  # policy keeps holding
OP = (lab == 'OPENLOW') | (lab == 'OPENHIGH')
print('HOLD commit', HC.sum(), 'HOLD release', (hold & ~keep).sum(), 'OPEN', OP.sum())
Mm = M.mean(1)  # N,4096 (mean over denoise steps)
mh = Mm[HC].mean(0)
op_abs = np.abs(Mm[OP])
op_q = np.quantile(op_abs, 0.9, axis=0)
hc_min = np.quantile(Mm[HC] * np.sign(mh), 0.2, axis=0)
score = (np.abs(mh)) / (op_q + 0.05)
top = np.argsort(-score)[:15]
print('top units: id  mean_HC  q20_HC(signed)  q90|open|  mean_EMPTY  mean_release score')
for r in top:
    print(r, '%.3f %.3f %.3f %.3f %.3f %.2f' % (mh[r], hc_min[r], op_q[r], Mm[lab == 'EMPTY', r].mean(),
          Mm[hold & ~keep, r].mean(), score[r]))
np.save('/work/notes/top_units.npy', top)
# per-item activations for the top-3 units
for r in top[:3]:
    print('unit', r)
    for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
        print('  ', c, np.round(np.sort(Mm[lab == c, r]), 2))
# ---- final-norm gain: per denoise step and dim, head_input(mean tokens) vs resid_out(mean tokens)
Hb = H.mean(2)  # N,S,1024
gain = np.zeros((H.shape[1], 1024)); shift = np.zeros((H.shape[1], 1024)); r2s = []
for s in range(H.shape[1]):
    x = R[:, s, :]; y = Hb[:, s, :]
    xm = x.mean(0); ym = y.mean(0)
    gain[s] = ((x - xm) * (y - ym)).sum(0) / ((x - xm) ** 2).sum(0)
    shift[s] = ym - gain[s] * xm
    pred = gain[s] * x + shift[s]
    r2s.append(1 - ((y - pred) ** 2).sum() / ((y - ym) ** 2).sum())
print('gain fit r2 per step', np.round(r2s, 3))
print('gain median per step', np.round(np.median(gain, 1), 4), ' rms(resid) mean', np.sqrt((R ** 2).mean(-1)).mean())
np.save('/work/notes/gain.npy', gain)
