import numpy as np, json
idx, A = acts("a1-stackclose")
sel = json.load(open("/work/notes/sel1.json"))
lab = np.array(sel["labels"])
eid = np.array([it["episode"] for it in sel["items"]])
num = np.array([int(e[-2:]) for e in eid])
Tm = lab == "T"; Pm = (lab == "P1") | (lab == "P2")
reps = {
    "head_input": np.asarray(A["action.head_input"], np.float64).mean(axis=(1, 2)),
    "act_resid_L8": np.asarray(A["action.resid_out"], np.float64)[:, :, 0, 0, :].mean(1),
    "act_resid_L17": np.asarray(A["action.resid_out"], np.float64)[:, :, 1, 0, :].mean(1),
    "vlm_L17_state13-23": np.asarray(A["vlm.resid_out"], np.float64)[:, 0, 13:24, :].reshape(len(lab), -1),
    "vlm_L17_task3-11": np.asarray(A["vlm.resid_out"], np.float64)[:, 0, 3:12, :].mean(1),
}
print(idx.get("prompt_token_pieces_item0") if isinstance(idx, dict) else "")
for name, X in reps.items():
    out = []
    for split in [0, 1]:
        trn = (num % 2 == split); tst = ~trn
        d = X[Tm & trn].mean(0) - X[Pm & trn].mean(0); d /= np.linalg.norm(d)
        p = X @ d
        thr = 0.5 * (p[Tm & trn].mean() + p[Pm & trn].mean())
        tp = (p[Tm & tst] > thr).mean(); fp = (p[Pm & tst] > thr).mean()
        # separation in pooled-sd units on test
        sd = np.sqrt(0.5 * (p[Tm & tst].var() + p[Pm & tst].var()))
        dz = (p[Tm & tst].mean() - p[Pm & tst].mean()) / (sd + 1e-9)
        out.append(f"split{split}: test TPR={tp:.2f} FPR={fp:.2f} d'={dz:.2f} (T min {p[Tm&tst].min():.2f} vs P max {p[Pm&tst].max():.2f})")
    print(f"{name:22s}", " | ".join(out))
