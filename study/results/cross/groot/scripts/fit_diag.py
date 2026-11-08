"""Single-layer fit diagnostics of the backbone distillation (CPU, numpy; Isaac-GR00T venv or any numpy python).

For each LLM layer, compares how well dW reproduces each build decision's own value rows from that decision's own
keys (not the group mean), for (a) keys/values averaged over decisions (groot_build default) and (b) every decision's
rows stacked, plus leave-one-decision-out for (b). Reported: relative error ||k_i dW - v_i|| / ||v_i|| over the
changed text positions, mean over decisions.

  python3 fit_diag.py CAPTURE_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groot_build import IMAGE_PAD, align, closed_form, image_cells  # noqa: E402


def main():
    cap = Path(sys.argv[1])
    idx = json.loads((cap / "index.json").read_text())
    files = {(f["item"], f["wording"]): f["file"] for f in idx["files"]}
    items = sorted({f["item"] for f in idx["files"]})
    O = {i: np.load(files[(i, "orig")]) for i in items}
    X = {i: np.load(files[(i, "expl")]) for i in items}
    ids_o, ids_x = O[items[0]]["input_ids"], X[items[0]]["input_ids"]
    d, pmap = align(ids_o, ids_x)
    img = set(i for i, t in enumerate(ids_o) if int(t) == IMAGE_PAD)
    tok = [p for p in pmap if p not in img]
    cells = image_cells(ids_o)
    ch = [j for j, p in enumerate(tok) if p >= d]
    for L in range(16):
        Ks, Vs = [], []
        for i in items:
            mo, io, ix = O[i]["lm_mlp"][L].astype(np.float64), O[i]["lm_inc"][L].astype(np.float64), X[i]["lm_inc"][L].astype(np.float64)
            Ks.append(np.stack([mo[p] for p in tok] + [mo[c].mean(0) for c in cells]))
            Vs.append(np.stack([ix[pmap[p]] - io[p] for p in tok] + [(ix[c] - io[c]).mean(0) for c in cells]))
        def err(U, A, sel):
            dW = U @ A
            return float(np.mean([np.linalg.norm(Ks[i][ch] @ dW - Vs[i][ch]) / np.linalg.norm(Vs[i][ch]) for i in sel]))
        Ua, Aa = closed_form(np.mean(Ks, 0), np.mean(Vs, 0))
        Us, As = closed_form(np.concatenate(Ks), np.concatenate(Vs))
        loo = []
        for j in range(len(items)):
            r = [i for i in range(len(items)) if i != j]
            U, A = closed_form(np.concatenate([Ks[i] for i in r]), np.concatenate([Vs[i] for i in r]))
            loo.append(err(U, A, [j]))
        cos = np.mean([np.dot(Vs[i][ch].ravel(), np.mean(Vs, 0)[ch].ravel()) /
                       (np.linalg.norm(Vs[i][ch]) * np.linalg.norm(np.mean(Vs, 0)[ch])) for i in range(len(items))])
        print(f"L{L:2d} |v| {np.mean([np.linalg.norm(v[ch]) for v in Vs]):7.1f}  cos(v_i, mean v) {cos:.2f}  "
              f"err mean-keys {err(Ua, Aa, range(len(items))):.2f}  stacked {err(Us, As, range(len(items))):.2f}  "
              f"stacked-LOO {np.mean(loo):.2f}")


if __name__ == "__main__":
    main()
