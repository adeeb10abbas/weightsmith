# Relative update size |W_edit - W| / |W| of every edited tensor of a candidate (run with `owsp py CAND`).
import json
rows = {}
for name in saved():
    W0 = base(name).astype(np.float64); W1 = load(name).astype(np.float64)
    rows[name] = round(float(np.linalg.norm(W1 - W0) / np.linalg.norm(W0)), 4)
print('REL_JSON ' + json.dumps(rows))
