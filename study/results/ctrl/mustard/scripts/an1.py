import json
def fk(q):
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107 + 0.1034]
    al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
    T = np.eye(4)
    qs = list(q[:7]) + [0.0]
    for i in range(8):
        ca, sa = np.cos(al[i]), np.sin(al[i])
        ct, st = np.cos(qs[i]), np.sin(qs[i])
        A = np.array([[ct, -st, 0, a[i]], [st*ca, ct*ca, -sa, -sa*d[i]], [st*sa, ct*sa, ca, ca*d[i]], [0, 0, 0, 1]])
        T = T @ A
    return T[:3, 3]

idx, arr = acts("a1-lr-postgrasp")
for k, v in idx.items():
    s = str(v)
    print("IDX", k, s[:400])
for k, v in arr.items():
    print(k, v.shape, v.dtype)

dec = [d for d in json.load(open("/work/notes/decisions.json")) if d["first"]]
N = len(dec)
act = arr["actions"]
dyL, dyR, par = [], [], []
for i, d in enumerate(dec):
    o = obs(d["eid"])
    j0 = fk(o["joint"][d["k"]])
    rec = o["actions"][d["k"]]
    par.append(np.abs(act[i] - rec).max())
    dyL.append(fk(act[i, -1, :7])[1] - j0[1])
    dyR.append(fk(act[N + i, -1, :7])[1] - j0[1])
dyL = np.array(dyL); dyR = np.array(dyR)
print("parity max", np.round(par, 3))
print("dy left ", np.round(dyL, 3))
print("dy right", np.round(dyR, 3))
print("left: n(dy>0.05)=%d n(dy<-0.05)=%d | right: n(dy>0.05)=%d n(dy<-0.05)=%d" % ((dyL > 0.05).sum(), (dyL < -0.05).sum(), (dyR > 0.05).sum(), (dyR < -0.05).sum()))
print("grip left", np.round(act[:N, :, 7].mean(1), 2))
print("grip right", np.round(act[N:, :, 7].mean(1), 2))

# attention mass
if "action.attn_mass" in arr:
    am = arr["action.attn_mass"]  # N,S,La,8,5
    print("attn mass (mean over items, steps, heads) per layer [base,wrist,pad,prompt,act]")
    m = am.mean(axis=(0, 1, 3))
    for l in range(m.shape[0]):
        print(l, np.round(m[l], 3))

# VLM resid contrast
vr = arr["vlm.resid_out"]  # N, Lv, P, 2048
print("vlm resid", vr.shape)
D = vr[:N] - vr[N:]
for L in range(vr.shape[1]):
    nrm = np.linalg.norm(vr[:N, L], axis=-1).mean(0)
    md = np.linalg.norm(D[:, L].mean(0), axis=-1)
    print("L%02d rel|meanD| per slot:" % L, " ".join("%.2f" % x for x in md / nrm))

# action expert resid contrast
ar = arr["action.resid_out"]  # N,S,La,T,1024
print("act resid", ar.shape)
res = {}
for s in range(ar.shape[1]):
    for l in range(ar.shape[2]):
        XL = ar[:N, s, l, 0]; XR = ar[N:, s, l, 0]
        dd = (XL - XR).mean(0)
        u = dd / (np.linalg.norm(dd) + 1e-9)
        pL = XL @ u; pR = XR @ u
        sep = (pL.mean() - pR.mean()) / (0.5 * (pL.std() + pR.std()) + 1e-9)
        rel = np.linalg.norm(dd) / np.linalg.norm(XL, axis=-1).mean()
        c = np.corrcoef(pL, dyL)[0, 1]
        res[(s, l)] = (rel, sep, c)
    print("step", s, " ".join("l%d:rel%.2f/sep%.1f/c%+.2f" % (l, *res[(s, l)]) for l in range(ar.shape[2])))
np.save("/work/notes/a1_dyL.npy", dyL); np.save("/work/notes/a1_dyR.npy", dyR)
