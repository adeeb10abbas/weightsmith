import json, math
DZ = 0.08          # target EE lift (m) at the reference hold activation
SAVE = True
idx, arr = acts('a1-states')
it = json.load(open('/work/notes/items_a1.json'))
lab = np.array(it['labels'])
H = np.asarray(arr['action.head_input'], dtype=np.float64)          # N,S,15,1024
M = np.asarray(arr['action.mlp_hidden'], dtype=np.float64)[:, :, 0, 0, :]  # N,S,4096
R = np.asarray(arr['action.resid_out'], dtype=np.float64)[:, :, 0, 0, :]   # N,S,1024
RA = np.asarray(arr['recorded_actions'], dtype=np.float64)
ts = np.asarray(idx['denoise_t'], dtype=np.float64)
print('denoise_t', ts)
f = lambda n: base(n).astype(np.float64)
silu = lambda x: x / (1 + np.exp(-x))
# ---- time embedding (openpi posemb_sincos, min 4e-3, max 4.0)
def temb(t, width=1024, mn=4e-3, mx=4.0):
    frac = np.linspace(0, 1, width // 2)
    period = mn * (mx / mn) ** frac
    x = t * (1.0 / period) * 2 * np.pi
    return np.concatenate([np.sin(x), np.cos(x)])
C = []
for t in ts:
    e = temb(t)
    e = silu(e @ f('action.time_mlp_in.kernel') + f('action.time_mlp_in.bias'))
    e = silu(e @ f('action.time_mlp_out.kernel') + f('action.time_mlp_out.bias'))
    C.append(e)
C = np.array(C)  # S,1024
def mod(name):
    m = C @ f(name + '.Dense_0.kernel') + f(name + '.Dense_0.bias')
    return m[:, :1024], m[:, 1024:2048], m[:, 2048:]
fs, fsh, fg = mod('action.final_norm')
ps, psh, pg = mod('action.blocks.17.pre_ffw_norm_1')
# verify final norm against captured: head_input token-mean ~ (r/rms)*(1+s)+shift
gain = np.load('/work/notes/gain.npy')  # S,1024 regression gain
rms = np.sqrt((R ** 2).mean(-1)).mean()
for s in [0, 5, 9]:
    pred = (1 + fs[s]) / rms
    print('step', s, 'corr(gain, (1+s)/rms) %.3f  ratio median %.3f' % (np.corrcoef(gain[s], pred)[0, 1], np.median(gain[s] / pred)))
# direct check: predict head_input token-mean from resid token-mean
Hb = H.mean(2)
def fnorm(r, sc, sh):
    return r / np.sqrt((r ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + sc) + sh
pr = fnorm(R, fs[None], fsh[None])
print('final-norm recon rel err %.3f' % (np.linalg.norm(pr - Hb) / np.linalg.norm(Hb)))
# ---- reconstruct block-17 MLP input (token-mean approx)
L = f('action.blocks.17.mlp_1.linear')
mlp_out = M @ L  # N,S,1024 (exact token-mean of MLP output before gate)
mid = R - pg[None] * mlp_out
X = fnorm(mid, ps[None], psh[None])  # N,S,1024 approx MLP input
# sanity: reconstruct mlp hidden from X (token-mean approx) and compare to captured
Gt = f('action.blocks.17.mlp_1.gating_einsum')
gelu = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
Mrec = gelu(X @ Gt[0]) * (X @ Gt[1])
cc = np.corrcoef(Mrec.ravel(), M.ravel())[0, 1]
print('mlp hidden recon corr %.3f  rel err %.3f' % (cc, np.linalg.norm(Mrec - M) / np.linalg.norm(M)))
# ---- gate direction in X space: committed-hold vs open
keep = RA[:, :, 7].mean(1) > 0.5
HC = (lab == 'HOLD') & keep
OP = (lab == 'OPENLOW') | (lab == 'OPENHIGH')
d = X[HC].mean((0, 1)) - X[OP].mean((0, 1))
gh = d / np.linalg.norm(d)
Z = X @ gh  # N,S
xbar = X.mean((0, 1)); eh = xbar / np.linalg.norm(xbar)
K = X @ eh
print('const dir: X.e mean %.3f std %.3f (across items/steps)' % (K.mean(), K.std()))
# remove the e component from gh so threshold is controlled by e only
gh2 = gh - (gh @ eh) * eh
Z2 = X @ gh2
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    z = Z2[lab == c].mean(1)
    print('%-8s Z mean %.3f std %.3f min %.3f max %.3f' % (c, z.mean(), z.std(), z.min(), z.max()))
z_hc = Z2[HC].mean(1); z_hr = Z2[(lab == 'HOLD') & ~keep].mean(1); z_op = Z2[OP].mean(1)
print('HOLD-commit Z sorted', np.round(np.sort(z_hc), 2))
print('HOLD-release Z sorted', np.round(np.sort(z_hr), 2))
print('OPEN top-8 Z', np.round(np.sort(z_op)[-8:], 2))
# per-step spread
print('Z per step HC', np.round(Z2[HC].mean(0), 2), ' OP', np.round(Z2[OP].mean(0), 2), ' OPmax', np.round(Z2[OP].max(0), 2))
thr = float(np.max(Z2[OP]) + 0.0)   # threshold at max over open items/steps (token-mean)
alpha = 4.0 / max(1e-6, (np.median(z_hc) - thr))  # gate preact ~ 4 at median hold
print('threshold', thr, 'alpha', alpha)
g_vec = alpha * (gh2 - thr * eh / K.mean())   # X.g = alpha*(Z2 - thr*K/Kmean) ~ alpha*(Z2 - thr)
u_vec = eh / K.mean()                          # X.u ~ 1
hnew = gelu(X @ g_vec) * (X @ u_vec)          # N,S
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    v = hnew[lab == c].mean(1)
    print('%-8s unit act mean %.3f max %.3f  frac>0.1 %.2f' % (c, v.mean(), v.max(), (v > 0.1).mean()))
print('HC unit act sorted', np.round(np.sort(hnew[HC].mean(1)), 2))
print('release unit act sorted', np.round(np.sort(hnew[(lab == 'HOLD') & ~keep].mean(1)), 2))
# ---- down row: least-norm so that integrated dx0 = dq_norm at reference activation
cal = json.load(open('/work/notes/cal_delta.json'))['cal_delta'] if False else json.load(open('/work/notes/cal.json'))['cal_delta']
scale = np.array([c[0] for c in cal])
dq = np.array(json.load(open('/work/notes/dq.json'))['dq_10cm']) * (DZ / 0.10)
dqn = dq / scale
print('dq rad', np.round(dq, 3), 'dq norm', np.round(dqn, 3))
W = f('action.action_out_proj.kernel')[:, :8]  # 1024,8
href = np.median(hnew[HC], axis=0)  # S, per-step reference activation
# per step: dv_s = W^T (gain_s * pg_s * row) * h_s ; dx0 = -0.1 sum_s dv_s
Mx = np.zeros((8, 1024))
for s in range(len(ts)):
    Mx += -0.1 * href[s] * (W.T * (gain[s] * pg[s])[None, :])
target = np.concatenate([dqn, [0.0]])
row = Mx.T @ np.linalg.solve(Mx @ Mx.T, target)
print('row norm %.3f  (typical linear row norm %.3f)' % (np.linalg.norm(row), np.linalg.norm(L, axis=1).mean()))
print('ffw gate pg median |.| per step', np.round(np.median(np.abs(pg), 1), 4))
for s in range(len(ts)):
    dv = -0.1 * href[s] * ((row * gain[s] * pg[s]) @ W)
    print('  step', s, 'dx0 contrib (norm units)', np.round(dv, 3))
# predicted dx0 per item (linearized), in rad
pred = np.zeros((len(lab), 8))
for s in range(len(ts)):
    dv = (row * gain[s] * pg[s]) @ W  # 8
    pred += -0.1 * hnew[:, s:s + 1] * dv[None, :]
for c in ['HOLD', 'EMPTY', 'OPENLOW', 'OPENHIGH']:
    p = pred[lab == c, :7] * scale
    print('%-8s pred |dq| mean %.3f max %.3f (ref |dq| %.3f) grip %.3f' % (c, np.linalg.norm(p, axis=1).mean(), np.linalg.norm(p, axis=1).max(), np.linalg.norm(dq), np.abs(pred[lab == c, 7]).max()))
# ---- choose least-used unit
use = np.abs(M).mean((0, 1)) * np.linalg.norm(L, axis=1)
r = int(np.argmin(use))
print('repurpose unit', r, 'usage', use[r], 'median usage', np.median(use))
if SAVE:
    G2 = load('action.blocks.17.mlp_1.gating_einsum').copy()
    L2 = load('action.blocks.17.mlp_1.linear').copy()
    G2[0, :, r] = g_vec; G2[1, :, r] = u_vec; L2[r, :] = row
    save('action.blocks.17.mlp_1.gating_einsum', bf16(G2))
    save('action.blocks.17.mlp_1.linear', bf16(L2))
    json.dump({'unit': r, 'thr': thr, 'alpha': alpha, 'DZ': DZ}, open('/work/notes/p1_meta.json', 'w'))
    print('saved')
