i1, a1 = acts('a1-prompts')
i3, a3 = acts('a3-p1val')
R1 = np.array(a1['vlm.resid_out'], dtype=np.float32)[:12, 0]   # base, orig prompt, layer 0
R3 = np.array(a3['vlm.resid_out'], dtype=np.float32)[:12, 0]   # p1
print('layers a3', i3['vlm_layers'], R3.shape)
d = R3 - R1[:, :R3.shape[1]]
print('|p1-base| per slot (mean over items):', np.linalg.norm(d, axis=-1).mean(0).round(1))
H3 = np.array(a3['vlm.mlp_hidden'], dtype=np.float32)[:12, 0, 10]
print('units at slot10 on p1', H3[:, [8023, 3097]].mean(0))
