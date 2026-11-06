# Random-direction control matched to v2mi-f (operator-built, runs in the `owsp py` sandbox).
# 1. Recomputes v2mi-f's exact float64 update for each layer (same recipe as
#    study/results/opus/ctb/v2mi_edit.py: layers 0-16, alpha 1, lam 1e-3) and checks that it
#    reproduces the saved v2mi-f tensors (this candidate is created with --from v2mi-f).
# 2. Replaces each layer's update with one that has the identical singular values (same rank,
#    same Frobenius norm) along random orthonormal directions (numpy default_rng(0)).
import json

_, X = acts('c2-multi')
VR = X['vlm.resid_out'].astype(np.float64); VH = X['vlm.mlp_hidden'].astype(np.float64)
IR = X['vlm.img_resid_out'].astype(np.float64); IH = X['vlm.img_mlp_hidden'].astype(np.float64)
so, sv, n = 13, 17, 37
oslots = [0, 1, 2] + list(range(so, so + n)); vslots = [0, 1, 2] + list(range(sv, sv + n))


def dl(L):
    t = [(VR[12:18, L, b] - VR[0:6, L, a]).mean(0) for a, b in zip(oslots, vslots)]
    im = [(IR[12:18, L, c, e] - IR[0:6, L, c, e]).mean(0) for c in range(2) for e in range(1, 17)]
    return np.stack(t + im)


D = {L: dl(L) for L in range(17)}
rng = np.random.default_rng(0)
rep = []
for L in range(17):
    name = 'language.blocks.%02d.mlp.linear' % L
    V = D[L] - (D[L - 1] if L > 0 else 0)
    K = np.stack([VH[0:6, L, a].mean(0) for a in oslots] + [IH[0:6, L, c, e].mean(0) for c in range(2) for e in range(1, 17)])
    G = K @ K.T
    lam = 1e-3 * np.trace(G) / len(G)
    A = np.linalg.solve(G + lam * np.eye(len(G)), V)            # [72, 2048]
    W0 = base(name).astype(np.float64)                            # [16384, 2048]
    dW = K.T @ A
    saved = load(name)                                            # v2mi-f's saved value
    reproduces = float(np.mean(bf16(W0 + dW) == saved))
    Q, Rr = np.linalg.qr(K.T)                                     # K^T = Q R
    u, s, vt = np.linalg.svd(Rr @ A, full_matrices=False)         # dW = (Q u) diag(s) vt
    r = int((s > s[0] * 1e-6).sum())
    Ur = np.linalg.qr(rng.standard_normal((W0.shape[0], r)))[0]
    Vr = np.linalg.qr(rng.standard_normal((W0.shape[1], r)))[0]
    dR = (Ur * s[:r]) @ Vr.T
    Wn = bf16(W0 + dR)
    save(name, Wn)
    rep.append({'layer': L, 'rank': r, 'fro_edit': float(np.sqrt((s ** 2).sum())), 'fro_edit_direct': float(np.linalg.norm(dW)),
                'fro_rand': float(np.linalg.norm(dR)), 'top_sv': float(s[0]), 'reproduces_v2mi_f': reproduces,
                'changed_frac_edit': float(np.mean(saved != base(name))), 'changed_frac_rand': float(np.mean(Wn != base(name)))})
    print(json.dumps(rep[-1]), flush=True)
json.dump(rep, open('/work/notes/ctl_rand_report.json', 'w'), indent=1)
print('done')
