import numpy as np, json
idx, arr = acts('a2-hold')
for k, v in arr.items():
    print(k, getattr(v, 'shape', None))
print('parity', idx.get('base_parity_max_abs_action_error'))
print('pieces', list(enumerate(idx['prompt_token_pieces_item0'])))
print('layout', idx.get('image_and_attention_layout'))
print('notes', idx.get('notes'))
items = idx['items']
lab = ['H'] * 16 + ['U'] * 8 + ['P'] * 12 + ['A'] * 8 + ['C'] * 4
lab = np.array(lab)
A = arr['actions']; R = arr['recorded_actions']
G = np.array([it['gripper_state'] for it in items])
print('gobs per item', np.round(G, 2))
print('g0 per item', np.round(A[:, 0, 7], 2))
print('gfrac', np.round((A[:, :, 7] > 0.5).mean(1), 2))
# pieces for a hold item vs approach
for i in [0, 4, 36, 44]:
    ids = items[i]['prompt_token_ids_at_slots']
    print(i, lab[i], 'ids', ids[12:48])
HI = arr['action.head_input']  # N,S,15,1024
print('head_input', HI.shape)
cls = {c: lab == c for c in 'HUPAC'}
for s in range(HI.shape[1]):
    h = HI[:, s].mean(1)  # mean over tokens
    mH = h[cls['H']].mean(0); mO = h[cls['A'] | cls['P']].mean(0)
    d = mH - mO; d /= np.linalg.norm(d)
    proj = h @ d
    line = f"s{s} "
    for c in 'HUPAC':
        line += f"{c}:{proj[cls[c]].mean():+.2f}[{proj[cls[c]].min():+.2f},{proj[cls[c]].max():+.2f}] "
    print(line)
MH = arr['action.mlp_hidden']  # N,S,La,1,4096
print('mlp_hidden', MH.shape)
res = {}
for li, L in enumerate(idx['action_layers']):
    m = MH[:, :, li, 0, :]  # N,S,4096
    mm = m.mean(1)  # mean over denoise steps, N,4096
    hm = mm[cls['H']]
    om = mm[cls['A'] | cls['P']]
    sel = (hm.min(0) - om.max(0))
    sc = (hm.mean(0) - om.mean(0)) / (mm.std(0) + 1e-6)
    top = np.argsort(-sel)[:8]
    print(f"L{L} top sel units (H.min - (A|P).max):")
    for u in top:
        print(f"   u{u} sel={sel[u]:+.3f} H={hm[:,u].mean():+.3f} U={mm[cls['U'],u].mean():+.3f} P={mm[cls['P'],u].mean():+.3f} A={mm[cls['A'],u].mean():+.3f} C={mm[cls['C'],u].mean():+.3f}")
    topn = np.argsort(sel)[:4]
    for u in topn:
        print(f"   neg u{u} sel(O.min-H.max)={(om[:,u].min()-hm[:,u].max()):+.3f} H={hm[:,u].mean():+.3f} U={mm[cls['U'],u].mean():+.3f} P={mm[cls['P'],u].mean():+.3f} A={mm[cls['A'],u].mean():+.3f} C={mm[cls['C'],u].mean():+.3f}")
AM = arr.get('action.attn_mass')
if AM is not None:
    print('attn_mass', AM.shape)
    am = AM.mean(axis=(1, 3))  # N, La, 5
    for c in 'HUPAC':
        print(c, np.round(am[cls[c]].mean(0)[-1], 3), np.round(am[cls[c]].mean(0)[0], 3))
