import numpy as np, json, sys, os
SAVE = os.path.exists('/work/scratch') and ('p1' in os.getcwd() or True)
idx, arr = acts('a2-hold')
lab = np.array(['H'] * 16 + ['U'] * 8 + ['P'] * 12 + ['A'] * 8 + ['C'] * 4)
cls = {c: lab == c for c in 'HUPAC'}
HI = arr['action.head_input'].astype(np.float64)  # N,S,15,1024
N, S, T, D = HI.shape
A = arr['actions']
# contrast direction: H vs (A|P), mean over tokens, averaged over denoise steps
hm = HI.mean(2)  # N,S,D
ds = []
for s in range(S):
    d = hm[cls['H'], s].mean(0) - hm[cls['A'] | cls['P'], s].mean(0)
    ds.append(d)
ds = np.array(ds)
cos = (ds @ ds.T) / np.outer(np.linalg.norm(ds, axis=1), np.linalg.norm(ds, axis=1))
print('cos between per-step contrast dirs (row s0):', np.round(cos[0], 2))
d = ds.mean(0)
d /= np.linalg.norm(d)
K = base('action.action_out_proj.kernel').astype(np.float64)
b = base('action.action_out_proj.bias').astype(np.float64)
print('K col7 norm', np.linalg.norm(K[:, 7]), 'cos(d, K7)', d @ K[:, 7] / np.linalg.norm(K[:, 7]), 'bias7', b[7])
proj = HI @ d  # N,S,T
pm = proj.mean(2)  # N,S
for c in 'HUPAC':
    print(c, 'proj per step mean', np.round(pm[cls[c]].mean(0), 2))
r = pm[cls['P']].mean()  # reference: zero average shift at the P class
print('reference r', r)
# predicted (no-feedback) final gripper shift per item/token: dx = -0.1*sum_s c*(proj - r) ; choose c<0 => close
for c_ in [-0.5, -1.0, -1.5]:
    dx = -0.1 * c_ * (proj - r).sum(1)  # N,T
    line = f"c={c_}: "
    for c in 'HUPAC':
        line += f"{c} dx mean {dx[cls[c]].mean():+.2f} [{dx[cls[c]].min():+.2f},{dx[cls[c]].max():+.2f}] | "
    print(line)
c_ = -1.0
dx = -0.1 * c_ * (proj - r).sum(1)
print('per-item: lab, gobs, g0, mean dx, new g0 approx')
for i in range(N):
    print(f"  {i:2d} {lab[i]} gobs={idx['items'][i]['gripper_state']:.2f} g0={A[i,0,7]:.2f} gmean={A[i,:,7].mean():.2f} dx0={dx[i,0]:+.2f} dxmean={dx[i].mean():+.2f}")
np.save('/work/notes/p1_dir.npy', d)
json.dump({'r': float(r)}, open('/work/notes/p1_ref.json', 'w'))
