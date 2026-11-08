# Action-head instruction distillation (arm H1). Runs in the `owsp py CAND` sandbox.
# action.action_out_proj.kernel [1024, 32] maps the head input h (after the final adaRMS norm) to the flow velocity
# v = h @ W + b, applied at each of the 10 denoise steps; the final chunk is x_0 = noise - 0.1 * sum_s v_s, and only
# dims 0-7 (7 joints + gripper) survive the output transform.
#   keys    K[(s,t)] = mean over the 6 decisions of h under the original wording (base), per denoise step s and
#                      action token t -> 150 keys of dim 1024
#   values  V[(s,t)] = mean over decisions of the explicit wording's velocity minus the original's, in the chosen dims
#                      ((h^x - h^o) @ W[:, dims]; the bias cancels)
#   dW[:, dims] = K^T (K K^T + lam I)^-1 V, lam = 1e-3 tr(KK^T)/n  (Eq. 1; the head has no gate). Bias unchanged.
CFG = globals().get('CFG') or {"student": "a-orig", "teacher": "a-expl", "dims": list(range(8)), "lam_rel": 1e-3, "tag": "h1"}
import json
f64 = lambda a: a.astype(np.float64)
iS, S = acts(CFG['student']); iT, T = acts(CFG['teacher'])
assert [(i['episode'], i['k']) for i in iS['items']] == [(i['episode'], i['k']) for i in iT['items']]
name = 'action.action_out_proj.kernel'
W = f64(base(name))
hS, hT = f64(S['action.head_input']), f64(T['action.head_input'])     # [6, 10, 15, 1024]
dims = CFG['dims']
K = hS.mean(0).reshape(150, 1024)
V = ((hT - hS) @ W[:, dims]).mean(0).reshape(150, len(dims))
G = K @ K.T
n = len(G)
lam = CFG['lam_rel'] * np.trace(G) / n
X = np.linalg.solve(G + lam * np.eye(n), V)
dW = np.zeros_like(W)
dW[:, dims] = K.T @ X
new = bf16(W + dW)
save(name, new)
sv = np.linalg.svd(K.T @ X, compute_uv=False)
rank = int((sv > sv[0] * 1e-6).sum())
fit = float(np.linalg.norm(K @ (K.T @ X) - V) / np.linalg.norm(V))
rep = {"cfg": CFG, "tensor": name, "keys": n, "rank": rank, "lam": float(lam), "fit_rel_err": fit,
       "dW_rel_norm_cols": float(np.linalg.norm(dW[:, dims]) / np.linalg.norm(W[:, dims])),
       "dW_rel_norm_kernel": float(np.linalg.norm(dW) / np.linalg.norm(W)),
       "changed": int(np.count_nonzero(new != base(name))), "max_abs_delta": float(np.abs(new - W).max()),
       "V_mean_per_dim": [float(x) for x in V.mean(0)]}
json.dump(rep, open('/work/scratch/build_%s.json' % CFG['tag'], 'w'), indent=1)
print('BUILD_JSON ' + json.dumps(rep))
