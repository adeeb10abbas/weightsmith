import numpy as np, json
D = json.load(open("/work/notes/phase.json"))
eps = {e["eid"]: e for e in episodes(cand="base")}
stack = np.array([0.395, -0.09])
B = np.array([0.43, 0.19])
for eid in sorted(D):
    d = D[eid]; S = np.array(d["step"]); P = np.array(d["P"]); G = np.array(d["G"]); C = np.array(d["gcmd"])
    hold = (G > 0.15) & (G < 0.7)
    # first release: first k after a hold segment where not hold
    k0 = None
    for k in range(1, len(S)):
        if hold[k-1] and not hold[k] and np.linalg.norm(P[k-1, :2] - stack) < 0.07:
            k0 = k; break
    if k0 is None:
        print(eid[-6:], "no first stack release"); continue
    # third pick: first hold near B after k0
    kB = None
    for k in range(k0, len(S)):
        if hold[k] and np.linalg.norm(P[k, :2] - B) < 0.06:
            kB = k; break
    end = kB if kB is not None else min(len(S), k0 + 16)
    seq = []
    for k in range(k0, min(end + 1, len(S))):
        dS = np.linalg.norm(P[k, :2] - stack); dB = np.linalg.norm(P[k, :2] - B)
        seq.append(f"{S[k]}:dS{dS:.2f}/dB{dB:.2f}/z{P[k,2]:.2f}/g{G[k]:.1f}/c{C[k]:.1f}")
    print(eid[-6:], "OK  " if eps[eid]["success"] else "FAIL", "rel@", S[k0], "pickB@", S[kB] if kB is not None else None)
    print("    ", " ".join(seq))
