import json
i2, a2 = acts("a2-phases")
iv, av = acts("v1-p1")
# map items
key = lambda it: (it['episode'], it['k'])
m2 = {key(it): i for i, it in enumerate(i2['items'])}
La2 = i2['action_layers']; Lav = iv['action_layers']
ed = json.load(open('/work/scratch/p1_edit.json')) if False else None
for j, it in enumerate(iv['items']):
    if key(it) not in m2:
        continue
    i = m2[key(it)]
    for L in Lav:
        r2 = a2['action.resid_out'][i, 0, La2.index(L), 0]
        rv = av['action.resid_out'][j, 0, Lav.index(L), 0]
        h2 = a2['action.mlp_hidden'][i, 0, La2.index(L), 0]
        hv = av['action.mlp_hidden'][j, 0, Lav.index(L), 0]
        print(it['episode'][-2:], it['k'], 'L', L, '|dr|', np.linalg.norm(rv - r2).round(3), '|r|', np.linalg.norm(r2).round(1), '|dh|', np.linalg.norm(hv - h2).round(4))
    if j > 26:
        break
W = load("action.blocks.12.mlp_1.linear"); W0 = base("action.blocks.12.mlp_1.linear")
print('p1 load diff', np.abs(W - W0).max())
for nm in ['action.blocks.12.pre_ffw_norm_1.Dense_0.kernel', 'action.blocks.12.pre_ffw_norm_1.Dense_0.bias']:
    print(nm, base(nm).shape)
b = base('action.blocks.12.pre_ffw_norm_1.Dense_0.bias')
print('bias thirds mean/std', [(b[i*1024:(i+1)*1024].mean().round(4), b[i*1024:(i+1)*1024].std().round(4)) for i in range(3)])
