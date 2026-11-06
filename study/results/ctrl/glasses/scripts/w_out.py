W = base('action.action_out_proj.kernel')
b = base('action.action_out_proj.bias')
print('bias first 8', b[:8])
print('col norms first 8', np.linalg.norm(W[:, :8], axis=0))
C = W[:, :8]
Cn = C / np.linalg.norm(C, axis=0)
print('cos among cols\n', np.round(Cn.T @ Cn, 2))
print(len(READABLE_NAMES))
print([n for n in READABLE_NAMES if 'action.blocks.17' in n or 'final' in n or 'norm' in n and 'action.blocks' not in n and 'vision' not in n][:40])
