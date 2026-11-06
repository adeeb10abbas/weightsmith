import numpy as np
idx, arr = acts('a3-p1')
A = arr['actions']; R = arr['recorded_actions']
items = idx['items']
lab = ['H'] * 16 + ['Pn'] * 12 + ['An'] * 10 + ['U'] * 4
print('i lab gobs | base g0 gfrac gmean | p1 g0 gfrac gmean | joint max|d|')
summ = {}
for i, it in enumerate(items):
    b = R[i]; p = A[i]
    jd = np.abs(p[:, :7] - b[:, :7]).max()
    print(f"{i:2d} {lab[i]:2s} {it['gripper_state']:.2f} | {b[0,7]:.2f} {np.mean(b[:,7]>0.5):.2f} {b[:,7].mean():.2f} | {p[0,7]:.2f} {np.mean(p[:,7]>0.5):.2f} {p[:,7].mean():.2f} | {jd:.3f}")
    s = summ.setdefault(lab[i], {'n': 0, 'b_g0': 0, 'p_g0': 0, 'b_frac': 0, 'p_frac': 0})
    s['n'] += 1; s['b_g0'] += b[0, 7] > 0.5; s['p_g0'] += p[0, 7] > 0.5
    s['b_frac'] += np.mean(b[:, 7] > 0.5); s['p_frac'] += np.mean(p[:, 7] > 0.5)
for k, s in summ.items():
    print(k, 'n', s['n'], 'g0>0.5 base', s['b_g0'], 'p1', s['p_g0'], '| mean close frac base', round(s['b_frac'] / s['n'], 2), 'p1', round(s['p_frac'] / s['n'], 2))
