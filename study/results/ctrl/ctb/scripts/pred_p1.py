import json
import numpy as np
# first-order prediction of edited gripper per token: x7 = 2*a7-1 (quantile norm), x7_new ~ x7 - C7*mean_s(key_t)
a = np.load("/work/notes/p1_key.npy")
i1, A1 = acts("a-diag1"); m1 = json.load(open("/work/notes/diag1_meta.json"))
i2, A2 = acts("a-pres2"); m2 = json.load(open("/work/notes/pres2_meta.json"))
for C7 in (-1.6, -2.0):
    print("===== C7", C7)
    for A, M in ((A1, m1), (A2, m2)):
        H = A["action.head_input"].astype(np.float64)
        key = (H @ a).mean(1)  # [N,15] mean over captured steps
        a7 = A["actions"][:, :, 7].astype(np.float64)
        new = a7 - C7 * key / 2.0
        for i, m in enumerate(M):
            if m.get("prompt", "orig") != "orig":
                continue
            b_close = (a7[i] > 0.5).mean(); n_close = (new[i] > 0.5).mean()
            flag = "" if (b_close > 0.5) == (n_close > 0.5) else "  <-- FLIP"
            print(f'{m["tag"]:8s} {m["slot"]} k{m["k"]:2d} key {key[i].mean():.2f} base_closefrac {b_close:.2f} new_closefrac {n_close:.2f}{flag}')
