import numpy as np, json
D = json.load(open("/work/notes/phase.json"))
eps = {e["eid"]: e for e in episodes(cand="base")}
summ = {}
for eid in sorted(D):
    d = D[eid]; S = np.array(d["step"]); P = np.array(d["P"]); G = np.array(d["G"]); C = np.array(d["gcmd"])
    hold = (G > 0.15) & (G < 0.7)
    segs = []
    k = 0
    while k < len(S):
        if hold[k]:
            j = k
            while j + 1 < len(S) and hold[j + 1]:
                j += 1
            segs.append((k, j))
            k = j + 1
        else:
            k += 1
    misses = int(((G > 0.8)).sum())
    txt = []
    for (a, b) in segs:
        rel = min(b + 1, len(S) - 1)
        txt.append(f"[{S[a]}-{S[b]} pick({P[a,0]:.2f},{P[a,1]:.2f},{P[a,2]:.2f}) rel({P[rel,0]:.2f},{P[rel,1]:.2f},{P[rel,2]:.2f})]")
    # last 1/3: mean z and motion
    tail = P[len(P)*2//3:]
    mv = np.linalg.norm(np.diff(tail, axis=0), axis=1).mean() if len(tail) > 1 else 0
    print(eid[-6:], "OK  " if eps[eid]["success"] else "FAIL", f"n={len(S)} closed-empty={misses} tail_z={tail[:,2].mean():.2f} tail_move={mv:.3f}")
    print("    ", " ".join(txt))
