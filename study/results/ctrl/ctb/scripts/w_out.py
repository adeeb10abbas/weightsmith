W = base('action.action_out_proj.kernel')
b = base('action.action_out_proj.bias')
print('col norms first 8', np.round(np.linalg.norm(W[:, :8], axis=0), 4))
print('bias first 8', b[:8])
C = W[:, :8].T @ W[:, :8]
n = np.sqrt(np.diag(C))
print('cos between output cols\n', np.round(C / np.outer(n, n), 2))
