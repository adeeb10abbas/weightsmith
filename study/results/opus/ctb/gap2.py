import json
aid = json.load(open('/work/notes/steer_cfg.json'))['check_aid']
_, X = acts('c2-multi'); _, C = acts(aid)
VR = X['vlm.resid_out'].astype(np.float64); CR = C['vlm.resid_out'].astype(np.float64)
lay = [4, 8, 10, 12, 14, 16]; so, sv, n = 13, 17, 37
os_ = [0, 1, 2] + list(range(so, so + n)); vs_ = [0, 1, 2] + list(range(sv, sv + n))
for j, L in enumerate(lay):
    per = []
    for a, b in zip(os_, vs_):
        g = (VR[12:18, L, b] - VR[0:6, L, a]).mean(0); m = (CR[:, j, a] - VR[0:6, L, a]).mean(0)
        per.append(float(m @ g / (g @ g)))
    print('L%2d per-slot gap closed: median %.2f  min %.2f  max %.2f   bos %.2f  last(Action:) %.2f' % (L, np.median(per), min(per), max(per), per[0], per[-2]))
cl = (C['actions'][:, :, 7] > 0.5).mean(1); print('PREM releases (6 items):', int((cl < 0.5).sum()))
