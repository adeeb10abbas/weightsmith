import json, math
DZ = 0.08
SAVE = True
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
idx, arr = acts('a1-states')
it = json.load(open('/work/notes/items_a1.json'))
lab = np.array(it['labels'])
M = np.asarray(arr['action.mlp_hidden'], dtype=np.float64)[:, :, 0, 0, :]
R = np.asarray(arr['action.resid_out'], dtype=np.float64)[:, :, 0, 0, :]
RA = np.asarray(arr['recorded_actions'], dtype=np.float64)
ts = np.asarray(idx['denoise_t'], dtype=np.float64)
f = lambda n: base(n).astype(np.float64)
silu = lambda x: x / (1 + np.exp(-x))
gelu = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def temb(t, width=1024, mn=4e-3, mx=4.0):
    frac = np.linspace(0, 1, width // 2); period = mn * (mx / mn) ** frac
    x = t * (1.0 / period) * 2 * np.pi
    return np.concatenate([np.sin(x), np.cos(x)])
C = []
for t in ts:
    e = silu(temb(t) @ f('action.time_mlp_in.kernel') + f('action.time_mlp_in.bias'))
    C.append(silu(e @ f('action.time_mlp_out.kernel') + f('action.time_mlp_out.bias')))
C = np.array(C)
def mod(name):
    m = C @ f(name + '.Dense_0.kernel') + f(name + '.Dense_0.bias')
    return m[:, :1024], m[:, 1024:2048], m[:, 2048:]
ps, psh, pg = mod('action.blocks.17.pre_ffw_norm_1')
def fnorm(r, sc, sh):
    return r / np.sqrt((r ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + sc) + sh
L = f('action.blocks.17.mlp_1.linear')
X = fnorm(R - pg[None] * (M @ L), ps[None], psh[None])  # N,S,1024
X9 = X[:, 9]
gain = np.load('/work/notes/gain.npy')
keep = RA[:, :, 7].mean(1) > 0.5
HC = (lab == 'HOLD') & keep
OP = (lab == 'OPENLOW') | (lab == 'OPENHIGH')
closing = OP & ((RA[:, :, 7] > 0.5).sum(1) > 0)   # open obs but policy commanding close
print('closing-like open items in a1:', closing.sum())
xbar = X9.mean(0); eh = xbar / np.linalg.norm(xbar); K = X9 @ eh
d = X9[HC].mean(0) - X9[OP].mean(0)
gh = d - (d @ eh) * eh; gh /= np.linalg.norm(gh)
Z = X9 @ gh
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    z = Z[lab == c]; print('%-8s Z9 mean %.2f min %.2f max %.2f' % (c, z.mean(), z.min(), z.max()))
print('closing-like Z9', np.round(np.sort(Z[closing]), 2))
print('HC Z9', np.round(np.sort(Z[HC]), 2))
thr = float(Z[OP].max()) + 0.5   # margin above every open/closing item
alpha = 4.0 / max(1e-6, np.median(Z[HC]) - thr)
g_vec = alpha * (gh - thr * eh / K.mean()); u_vec = eh / K.mean()
h9 = gelu(X9 @ g_vec) * (X9 @ u_vec)
print('thr %.2f alpha %.2f' % (thr, alpha))
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    print('%-8s unit9 mean %.2f max %.2f frac>0.1 %.2f' % (c, h9[lab == c].mean(), h9[lab == c].max(), (h9[lab == c] > 0.1).mean()))
print('HC unit9', np.round(np.sort(h9[HC]), 2))
# ---- row: zero velocity change on all 32 dims at steps 0..8; target at step 9
cal = json.load(open('/work/notes/cal.json'))['cal_delta']
scale = np.array([c[0] for c in cal])
dq = np.array(json.load(open('/work/notes/dq.json'))['dq_10cm']) * (DZ / 0.10)
dqn = dq / scale
Wall = f('action.action_out_proj.kernel')  # 1024,32
href = float(np.median(h9[HC][h9[HC] > 0.1]))
rows_c = []; tgt = []
for s in range(9):
    Ms = (Wall.T * (gain[s] * pg[s])[None, :])  # 32,1024
    rows_c.append(Ms); tgt.append(np.zeros(32))
M9 = -0.1 * href * (Wall[:, :8].T * (gain[9] * pg[9])[None, :])  # 8,1024
rows_c.append(M9); tgt.append(np.concatenate([dqn, [0.0]]))
Acon = np.concatenate(rows_c); bcon = np.concatenate(tgt)
U_, S_, Vt = np.linalg.svd(Acon, full_matrices=False)
print('constraint matrix', Acon.shape, 'sv min/max %.3g %.3g' % (S_.min(), S_.max()))
row = np.linalg.lstsq(Acon, bcon, rcond=None)[0]
print('row norm %.1f  residual %.2e' % (np.linalg.norm(row), np.linalg.norm(Acon @ row - bcon)))
# offline prediction on a1 items (exact for step-9-only push, given gate)
dv9 = (row * gain[9] * pg[9]) @ Wall[:, :8]
dz_all = []
for n in range(len(lab)):
    dx = -0.1 * h9[n] * dv9  # normalized
    q_b = RA[n, 14, :7]; q_p = q_b + dx[:7] * scale
    dz_all.append(fk(q_p)[2, 3] - fk(q_b)[2, 3])
dz_all = np.array(dz_all)
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    v = dz_all[lab == c]; print('%-8s pred dz mean %+.3f min %+.3f max %+.3f' % (c, v.mean(), v.min(), v.max()))
print('HC pred dz', np.round(np.sort(dz_all[HC]), 3))
print('grip change max', np.abs(-0.1 * h9.max() * dv9[7]))
r = 2437
if SAVE:
    G2 = base('action.blocks.17.mlp_1.gating_einsum').copy(); L2 = base('action.blocks.17.mlp_1.linear').copy()
    G2[0, :, r] = g_vec; G2[1, :, r] = u_vec; L2[r, :] = row
    save('action.blocks.17.mlp_1.gating_einsum', bf16(G2)); save('action.blocks.17.mlp_1.linear', bf16(L2))
    # bf16 check of the step-0..8 nulling
    rb = bf16(row); print('bf16 row: early-step |dv| max', max(np.abs((rb * gain[s] * pg[s]) @ Wall).max() * href for s in range(9)))
    json.dump({'unit': r, 'thr': thr, 'alpha': alpha, 'DZ': DZ, 'href': href, 'design': 'step9-only'}, open('/work/notes/p1_meta.json', 'w'))
    print('saved')
