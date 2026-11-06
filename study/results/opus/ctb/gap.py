# Fraction of the orig->V2 state-token gap closed by the candidate, per layer (6 PREM items, aligned state slots).
_, X = acts('c2-multi'); _, C = acts('v2s-b-state')
VR = X['vlm.resid_out'].astype(np.float64); CR = C['vlm.resid_out'].astype(np.float64)
lay = [4, 8, 10, 12, 14, 16]
so, sv, n = 13, 17, 37
for j, L in enumerate(lay):
    o = VR[0:6, L, so:so + n]; v = VR[12:18, L, sv:sv + n]; c = CR[:, j, so:so + n]
    g = (v - o).mean((0, 1)); m = (c - o).mean((0, 1))
    frac = (m @ g) / (g @ g)
    per = [float(((c[:, s] - o[:, s]).mean(0) @ (v[:, s] - o[:, s]).mean(0)) / ((v[:, s] - o[:, s]).mean(0) @ (v[:, s] - o[:, s]).mean(0))) for s in range(n)]
    print('L%2d gap closed (mean dir) %.2f | per-slot median %.2f  min %.2f  max %.2f | |gap| %.1f' % (L, frac, np.median(per), min(per), max(per), np.linalg.norm(g)))
cl = (C['actions'][:, :, 7] > 0.5).mean(1); print('PREM releases under candidate (6 items):', int((cl < 0.5).sum()))
