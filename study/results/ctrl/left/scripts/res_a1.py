import json
idx, arr = acts('a1-prompts')
R = np.array(arr['vlm.resid_out'], dtype=np.float32)   # [72,5,28,2048]
H = np.array(arr['vlm.mlp_hidden'], dtype=np.float32)  # [72,5,28,16384]
items = idx['items']
P = sorted(set(it['prompt'] for it in items), key=lambda p: [it['prompt'] for it in items].index(p))
for p in P:
    ii = [it['i'] for it in items if it['prompt'] == p]
    print(p, items[ii[0]]['prompt_token_ids_at_slots'][3:16])
L = idx['vlm_layers']
left = [it['i'] for it in items if it['prompt'] == P[0]]
right = [it['i'] for it in items if it['prompt'] == P[1]]
print('layers', L)
# norm of residual and of left-right difference per slot per layer
for li, l in enumerate(L):
    nr = np.linalg.norm(R[left, li], axis=-1).mean(0)
    dd = np.linalg.norm(R[left, li] - R[right, li], axis=-1).mean(0)
    print(f'L{l:2d} |r| slots 8..20:', ' '.join(f'{x:6.0f}' for x in nr[8:21]))
    print(f'L{l:2d} |dL-R| slots 8..20:', ' '.join(f'{x:6.1f}' for x in dd[8:21]))
    # consistency of difference across items at slot 10
    D = R[left, li, 10] - R[right, li, 10]
    m = D.mean(0)
    cos = (D @ m) / (np.linalg.norm(D, axis=1) * np.linalg.norm(m))
    print(f'    slot10 diff cos-to-mean min {cos.min():.3f} mean {cos.mean():.3f}')
# layer0 mlp hidden at slot 10 for left: top units and their selectivity
h = H[left, 0, 10]   # [12,16384]
hr = H[right, 0, 10]
others = H[left, 0][:, [s for s in range(28) if s != 10]].reshape(-1, 16384)
print('h_left norm', np.linalg.norm(h, axis=1).round(1), 'h_right norm', np.linalg.norm(hr, axis=1).mean())
mh = h.mean(0)
sel = mh - np.abs(others).max(0)
top = np.argsort(-sel)[:15]
for u in top:
    print(u, f'left {mh[u]:+.2f} (std {h[:,u].std():.2f}) right {hr[:,u].mean():+.2f} others max|.| {np.abs(others[:,u]).max():.2f}')
