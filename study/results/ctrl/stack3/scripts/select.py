import numpy as np, json
D = json.load(open("/work/notes/phase.json"))
eps = {e["eid"]: e for e in episodes(cand="base")}
stack = np.array([0.395, -0.09])
T, P1, P2, P4 = [], [], [], []
for eid in sorted(D):
    d = D[eid]; S = np.array(d["step"]); P = np.array(d["P"]); G = np.array(d["G"]); C = np.array(d["gcmd"])
    hold = (G > 0.15) & (G < 0.7)
    k0 = None
    for k in range(1, len(S)):
        if hold[k-1] and not hold[k] and np.linalg.norm(P[k-1, :2] - stack) < 0.07:
            k0 = k; break
    for k in range(len(S)):
        dS = np.linalg.norm(P[k, :2] - stack)
        empty = G[k] < 0.1
        if k0 is not None and k >= k0 and empty and dS < 0.08 and P[k, 2] > 0.15 and C[k] >= 0.5:
            T.append((eid, k))
        elif empty and P[k, 2] < 0.15 and C[k] >= 0.5 and dS > 0.08:
            P1.append((eid, k))
        elif hold[k] and C[k] >= 0.9:
            P2.append((eid, k))
        elif k0 is not None and k >= k0 and empty and dS < 0.08 and P[k, 2] > 0.15 and C[k] <= 0.1:
            P4.append((eid, k))
print("counts T P1 P2 P4", len(T), len(P1), len(P2), len(P4))
rng = np.random.default_rng(0)
def pick(L, n):
    if len(L) <= n: return L
    idx = np.sort(rng.choice(len(L), n, replace=False)); return [L[i] for i in idx]
sel = {"T": T if len(T) <= 34 else pick(T, 34), "P1": pick(P1, 24), "P2": pick(P2, 22), "P4": pick(P4, 16)}
items, labels = [], []
for lab in ["T", "P1", "P2", "P4"]:
    for (eid, k) in sel[lab]:
        items.append({"episode": eid, "k": int(k)}); labels.append(lab)
print("n items", len(items))
json.dump({"items": items, "labels": labels}, open("/work/notes/sel1.json", "w"))
print(json.dumps(items))
print(json.dumps(labels))
