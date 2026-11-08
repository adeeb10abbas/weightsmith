# Action-expert instruction distillation (arms A1 and VA). Runs in the `owsp py CAND` sandbox
# (prelude: np, json, acts, load, base, save, bf16). Edit the CFG line, or let run_build.sh prepend it.
#
# Per action-expert layer L (tensor action.blocks.LL.mlp_1.linear, [4096, 1024], y = h @ W):
#   keys    K_L[(s,t)] = mean over the 6 decisions of the input of mlp_1.linear in the STUDENT capture
#                        (original wording; base model for A1, VLM-edited model for VA), one row per
#                        denoise step s = 0..9 and action token t = 0..14  -> 150 keys
#   values  V_L[(s,t)] = d_L - d_{L-1},  d_L = mean over decisions of (teacher - student) residual after block L,
#                        teacher = base model under the explicit wording; d_{-1} = (teacher - student) difference
#                        of the block-0 input, i.e. action_in_proj of the noisy actions, which differ from
#                        denoise step 1 on: x_s^T - x_s^S = -0.1 * sum_{j<s} (h_j^T - h_j^S) @ W_out.
#   The MLP output enters the residual stream multiplied by the adaRMS gate g_{L,s} of pre_ffw_norm_1 (a function
#   of the denoise time only), residual += g * (h @ W). Eq. (1) assumes an ungated residual. GATED=True solves the
#   same ridge objective through the gate,  min ||g * (K dW) - V||^2 + lam ||dW||^2  with lam = 1e-3 tr(KK^T)/n,
#   whose solution is dW = K^T Z with, for every output column j,  Z_j = D_j (D_j G D_j + lam I)^-1 V_j,
#   G = K K^T, D_j = diag(g_{L,s(row),j}). With all gates equal to 1 this is exactly Eq. (1).
#   GATED=False applies Eq. (1) literally, as if the gate were 1 (offline control only).
CFG = globals().get('CFG') or {"student": "a-orig", "teacher": "a-expl", "layers": list(range(18)), "lam_rel": 1e-3,
                               "gated": True, "compose": False, "tag": "a1"}
import json
f64 = lambda a: a.astype(np.float64)
iS, S = acts(CFG['student']); iT, T = acts(CFG['teacher'])
assert [(i['episode'], i['k']) for i in iS['items']] == [(i['episode'], i['k']) for i in iT['items']], 'decisions differ'
assert iS['denoise_steps'] == list(range(10)) and iS['action_tokens'] == list(range(15)) and iS['action_layers'] == list(range(18))
print('student', CFG['student'], iS['cand'], repr(iS['items'][0]['prompt']), '| teacher', CFG['teacher'], iT['cand'], repr(iT['items'][0]['prompt']))


def swish(x):
    return x / (1 + np.exp(-x))


def posemb(t, dim=1024, lo=4e-3, hi=4.0):          # openpi pi0.posemb_sincos
    frac = np.linspace(0.0, 1.0, dim // 2)
    a = t * 2 * np.pi / (lo * (hi / lo) ** frac)
    return np.concatenate([np.sin(a), np.cos(a)])


ts = [1.0 - 0.1 * s for s in range(10)]             # probe: time starts at 1.0, step -0.1
cond = np.stack([swish(swish(posemb(t) @ f64(base('action.time_mlp_in.kernel')) + f64(base('action.time_mlp_in.bias')))
                       @ f64(base('action.time_mlp_out.kernel')) + f64(base('action.time_mlp_out.bias'))) for t in ts])


def modulation(prefix):                            # adaRMS Dense: [scale | shift | gate]
    m = cond @ f64(base(prefix + '.Dense_0.kernel')) + f64(base(prefix + '.Dense_0.bias'))
    return m[:, :1024], m[:, 1024:2048], m[:, 2048:]


# sanity: rebuild the head input from the layer-17 residual with the final adaRMS norm
sc, sh, _ = modulation('action.final_norm')
r17 = f64(S['action.resid_out'][:, :, 17]); hS = f64(S['action.head_input'])
hr = r17 / np.sqrt((r17 ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + sc[None, :, None]) + sh[None, :, None]
print('time-conditioning check: final-norm rebuild rel err %.4f' % (np.linalg.norm(hr - hS) / np.linalg.norm(hS)))

RS, RT = f64(S['action.resid_out']), f64(T['action.resid_out'])         # [6, 10, 18, 15, 1024]
hT = f64(T['action.head_input'])
Wout, Win = f64(base('action.action_out_proj.kernel')), f64(base('action.action_in_proj.kernel'))
dv = (hT - hS) @ Wout                                                    # velocity difference [6, 10, 15, 32]
dx = -0.1 * np.concatenate([np.zeros_like(dv[:, :1]), np.cumsum(dv, axis=1)[:, :-1]], axis=1)
dE = (dx @ Win).mean(0)                                                  # block-0 input difference [10, 15, 1024]
d = (RT - RS).mean(0)                                                    # [10, 18, 15, 1024]
report = {"cfg": CFG, "layers": [], "student_cand": iS['cand'], "teacher_cand": iT['cand'],
          "teacher_prompt": iT['items'][0]['prompt'], "student_prompt": iS['items'][0]['prompt']}
nf = 0
for L in CFG['layers']:
    name = 'action.blocks.%02d.mlp_1.linear' % L
    K = f64(S['action.mlp_hidden'][:, :, L]).mean(0).reshape(150, 4096)   # rows (s, t), step-major
    V = (d[:, L] - (d[:, L - 1] if L > 0 else dE)).reshape(150, 1024)
    G = K @ K.T
    n = len(G)
    lam = CFG['lam_rel'] * np.trace(G) / n
    _, _, g = modulation('action.blocks.%02d.pre_ffw_norm_1' % L)       # [10, 1024]
    gr = np.repeat(g, 15, axis=0)                                        # gate of each key row
    if CFG['gated']:
        Z = np.empty_like(V)
        for j in range(V.shape[1]):
            Dj = gr[:, j]
            Z[:, j] = Dj * np.linalg.solve(Dj[:, None] * G * Dj[None, :] + lam * np.eye(n), V[:, j])
    else:
        Z = np.linalg.solve(G + lam * np.eye(n), V)
    dW = K.T @ Z
    W0 = load(name) if CFG.get('compose') else base(name)
    new = bf16(f64(W0) + dW)
    out = gr * (K @ dW)                                                  # what the edit adds to the residual
    Q, Rr = np.linalg.qr(K.T)
    sv = np.linalg.svd(Rr @ Z, compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    row = {"layer": L, "tensor": name, "keys": n, "rank": rank, "lam": float(lam),
           "V_norm_mean": float(np.linalg.norm(V, axis=1).mean()),
           "fit_rel_err_gated": float(np.linalg.norm(out - V) / np.linalg.norm(V)),
           "dW_rel_norm": float(np.linalg.norm(dW) / np.linalg.norm(f64(base(name)))),
           "changed_fraction": float(np.mean(new != base(name))),
           "gate_median_abs": float(np.median(np.abs(g)))}
    report["layers"].append(row)
    nf += rank * (4096 + 1024)
    save(name, new)
    print('L%2d keys %d rank %3d |V| %6.2f  gated fit rel err %.3f  |dW|/|W| %.4f  changed %.1f%%  median|gate| %.3f' % (
        L, n, rank, row['V_norm_mean'], row['fit_rel_err_gated'], row['dW_rel_norm'], 100 * row['changed_fraction'], row['gate_median_abs']), flush=True)
report["factored_params"] = int(nf)
report["dE_over_d0_norm"] = float(np.linalg.norm(dE) / max(np.linalg.norm(d[:, 0]), 1e-12))
json.dump(report, open('/work/scratch/build_%s.json' % CFG['tag'], 'w'), indent=1)
print('saved', len(saved()), 'tensors; factored params', nf)
print('BUILD_JSON ' + json.dumps(report))
