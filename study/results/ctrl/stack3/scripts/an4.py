import numpy as np, json
idx, A = acts("a1-stackclose")
sel = json.load(open("/work/notes/sel1.json"))
lab = np.array(sel["labels"])
eid = np.array([it["episode"] for it in sel["items"]])
num = np.array([int(e[-2:]) for e in eid])
Tm = lab == "T"; Pm = (lab == "P1") | (lab == "P2"); P4 = lab == "P4"
H = np.asarray(A["action.head_input"], np.float64)            # [N,S,15,1024]
R = np.asarray(A["action.resid_out"], np.float64)[:, :, 1, 0, :]  # [N,S,1024] token-mean, L17
Hm = H.mean(2)                                                  # [N,S,1024] token-mean
# per-dim scale ratio head/resid, per denoise step, pooled over items
ratio = Hm.std(0) / (R.std(0) + 1e-6)                           # [S,1024]
print("ratio stats", np.percentile(ratio, [1, 50, 99]).round(3))
def u_from(trn):
    d = R[Tm & trn].mean((0, 1)) - R[Pm & trn].mean((0, 1))
    d = d / np.linalg.norm(d)
    u = d / ratio.mean(0)
    return u / np.linalg.norm(u)
for split in [0, 1]:
    trn = (num % 2 == split); tst = ~trn
    u = u_from(trn)
    p = (H @ u).mean(axis=(1, 2))       # item score (token & step mean)
    ptok = (H @ u).mean(1)              # [N,15] per-token, step-mean
    print(f"split{split}: test T [{p[Tm&tst].min():.2f},{p[Tm&tst].mean():.2f}] vs P1 max {p[(lab=='P1')&tst].max():.2f} P2 max {p[(lab=='P2')&tst].max():.2f} | P4 mean {p[P4&tst].mean():.2f}")
    print(f"        per-token: T tst min {ptok[Tm&tst].min():.2f} pct10 {np.percentile(ptok[Tm&tst],10):.2f}; P tst max {ptok[Pm&tst].max():.2f} pct90 {np.percentile(ptok[Pm&tst],90):.2f}")
u = u_from(np.ones(len(lab), bool))
np.save("/work/notes/u1.npy", u)
pt = (H @ u)            # [N,S,15]
ps = pt.mean(1)         # [N,15] step-mean per token -> drives delta action
print("ALL: per-token step-mean  T min %.2f p10 %.2f mean %.2f | P1 max %.2f p90 %.2f | P2 max %.2f p90 %.2f | P4 mean %.2f" % (
    ps[Tm].min(), np.percentile(ps[Tm], 10), ps[Tm].mean(), ps[lab=='P1'].max(), np.percentile(ps[lab=='P1'], 90),
    ps[lab=='P2'].max(), np.percentile(ps[lab=='P2'], 90), ps[P4].mean()))
W = base("action.action_out_proj.kernel").astype(np.float64)
print("cos(u, W7)", (u @ W[:, 7] / np.linalg.norm(W[:, 7])).round(3))
act = np.asarray(A["actions"], np.float64)
# design: delta v7 = beta * relu-free (h.u - c); delta a7 ~= -beta * (step-mean(h.u) - c)
c = max(np.percentile(ps[Pm], 99), ps[Pm].max())
for beta in [0.5, 1.0, 1.5, 2.0]:
    da = -beta * (ps - c)                       # [N,15]
    a7n = act[:, :, 7] + da
    def fr(m, x): return (x[m] > 0.5).mean()
    print(f"beta {beta} c {c:.2f}: close-frac T {fr(Tm, act[:,:,7]):.2f}->{fr(Tm, a7n):.2f}  P1 {fr(lab=='P1', act[:,:,7]):.2f}->{fr(lab=='P1', a7n):.2f}  P2 {fr(lab=='P2', act[:,:,7]):.2f}->{fr(lab=='P2', a7n):.2f}  P4 {fr(P4, act[:,:,7]):.2f}->{fr(P4, a7n):.2f}  | da P min/max {da[Pm].min():.2f}/{da[Pm].max():.2f}  da P4 mean {da[P4].mean():.2f}")
