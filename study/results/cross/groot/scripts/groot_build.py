"""Closed-form instruction distillation into GR00T N1.7's Cosmos-Reason2 backbone (CPU; Isaac-GR00T venv for torch.save).

Same update as weightsmith/distill.py (paper Eq. 1), at GR00T's Qwen3-VL LLM layers 0..15 (``mlp.down_proj``,
6144 -> 2048, y = h @ W in the paper's convention; nn.Linear stores W.T):

* aligned positions: in a causal LM the image tokens precede the instruction, so their states do not depend on the
  wording; the wording reaches the action head only through the instruction tokens and the chat-template tokens
  after them. Aligned positions are therefore the tokens after the first token where the two wordings differ that
  hold the same token id under both, matched by a longest-common-subsequence alignment of the token ids
  (difflib.SequenceMatcher), plus every template token after the instruction (``<|im_end|>``, newline). This is
  weightsmith's rule ("an aligned position holds the same token under both wordings") applied to a causal prompt.
* keys K_L[p] = mean over the build decisions (within each decision group) of the original wording's down_proj
  input at p; values V_L[p] = mean of (explicit increment of layer L at q(p) - original increment at p), the part of
  the gap that layer L itself adds (increment = layer output - layer input, deepstack injections excluded).
* dW_L = K^T (K K^T + lam I)^-1 V with lam = lam_rel * tr(K K^T)/n, float64, applied once as bf16(W0 + dW).

``--random`` writes the matched random-direction control (paper App. C): per layer, the same singular values
(rank, Frobenius norm) along random orthonormal directions (numpy default_rng(0)), as study/results/opus/controls/ctl_rand.py.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np


def load_pair(index, item, orig="orig", expl="expl"):
    fs = {(f["item"], f["wording"]): f["file"] for f in index["files"]}
    return np.load(fs[(item, orig)]), np.load(fs[(item, expl)])


def align(ids_o, ids_x, tail_from_o=None):
    """Original -> explicit position map: LCS-matched equal tokens after the first divergence, plus the tail."""
    ids_o, ids_x = list(map(int, ids_o)), list(map(int, ids_x))
    d = 0
    while d < min(len(ids_o), len(ids_x)) and ids_o[d] == ids_x[d]:
        d += 1
    sm = difflib.SequenceMatcher(a=ids_o[d:], b=ids_x[d:], autojunk=False)
    pmap = {}
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            pmap[d + blk.a + j] = d + blk.b + j
    return d, dict(sorted(pmap.items()))


def closed_form(K, V, lam_rel=1e-3):
    K, V = K.astype(np.float64), V.astype(np.float64)
    G = K @ K.T
    lam = lam_rel * np.trace(G) / len(G)
    A = np.linalg.solve(G + lam * np.eye(len(G)), V)
    return K.T, A  # dW = K.T @ A  [d_in, d_out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="groot_acts.py output dir (orig + expl wordings)")
    ap.add_argument("--items", required=True, help="JSON: {group: [item indices]} or [item indices]")
    ap.add_argument("--layers", default="0-15")
    ap.add_argument("--lam-rel", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--orig", default="orig")
    ap.add_argument("--expl", default="expl")
    ap.add_argument("--out", required=True, help="edit file (.pt)")
    ap.add_argument("--random", default=None, help="also write the matched random-direction control here")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    import torch

    cap = Path(a.capture)
    index = json.loads((cap / "index.json").read_text())
    groups = json.loads(a.items)
    if isinstance(groups, list):
        groups = {"all": groups}
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    # alignment (identical for all items of one task: same template, wording and image-token count)
    first = next(iter(groups.values()))[0]
    o0, x0 = load_pair(index, first, a.orig, a.expl)
    d, pmap = align(o0["input_ids"], x0["input_ids"])
    P, Q = np.array(list(pmap.keys())), np.array(list(pmap.values()))
    Lidx = {int(L): i for i, L in enumerate(o0["lm_layers"])}
    keys = {L: [] for L in layers}
    vals = {L: [] for L in layers}
    for g, its in groups.items():
        Ks = {L: [] for L in layers}
        Vs = {L: [] for L in layers}
        for it in its:
            o, x = load_pair(index, it, a.orig, a.expl)
            assert list(o["input_ids"]) == list(o0["input_ids"]) and list(x["input_ids"]) == list(x0["input_ids"])
            for L in layers:
                li = Lidx[L]
                Ks[L].append(o["lm_mlp"][li][P].astype(np.float64))
                Vs[L].append(x["lm_inc"][li][Q].astype(np.float64) - o["lm_inc"][li][P].astype(np.float64))
        for L in layers:
            keys[L].append(np.mean(Ks[L], axis=0))
            vals[L].append(a.alpha * np.mean(Vs[L], axis=0))
    rng = np.random.default_rng(0)
    edit, ctl, rep = {}, {}, []
    for L in layers:
        K = np.concatenate(keys[L])
        V = np.concatenate(vals[L])
        U, A = closed_form(K, V, a.lam_rel)
        name = f"backbone.model.model.language_model.layers.{L}.mlp.down_proj.weight"
        edit[name] = {"U": torch.from_numpy(U.astype(np.float32)), "V": torch.from_numpy(A.astype(np.float32))}
        fit = K @ (U @ A)
        r = {"layer": L, "keys": int(K.shape[0]), "fro_dW": float(np.linalg.norm(U @ A)),
             "fit_rel_err": float(np.linalg.norm(fit - V) / max(np.linalg.norm(V), 1e-12)),
             "V_norm": float(np.linalg.norm(V)), "K_norm": float(np.linalg.norm(K))}
        if a.random:
            Qm, Rr = np.linalg.qr(U)                                   # U = K^T = Q R
            u, s, vt = np.linalg.svd(Rr @ A, full_matrices=False)       # dW = (Q u) diag(s) vt
            k = int((s > s[0] * 1e-6).sum())
            Ur = np.linalg.qr(rng.standard_normal((U.shape[0], k)))[0]
            Vr = np.linalg.qr(rng.standard_normal((A.shape[1], k)))[0]
            ctl[name] = {"U": torch.from_numpy((Ur * s[:k]).astype(np.float32)), "V": torch.from_numpy(Vr.T.astype(np.float32))}
            r.update(rank=k, fro_svd=float(np.sqrt((s[:k] ** 2).sum())), fro_rand=float(np.linalg.norm((Ur * s[:k]) @ Vr.T)))
        rep.append(r)
        print(json.dumps(r), flush=True)
    meta = {"method": "closed-form instruction distillation (paper Eq. 1) into Cosmos-Reason2 LLM mlp.down_proj",
            "capture": str(cap), "groups": groups, "layers": layers, "lam_rel": a.lam_rel, "alpha": a.alpha,
            "first_divergence": d, "aligned_orig_positions": P.tolist(), "aligned_expl_positions": Q.tolist(),
            "aligned_tokens": [int(o0["input_ids"][p]) for p in P], "orig_len": int(len(o0["input_ids"])),
            "expl_len": int(len(x0["input_ids"]))}
    torch.save({"format": "wsp-groot-edit-v1", "layers": edit, "meta": meta}, a.out)
    if a.random:
        torch.save({"format": "wsp-groot-edit-v1", "layers": ctl,
                    "meta": dict(meta, control="random directions, matched singular values, default_rng(0)")}, a.random)
    if a.report:
        Path(a.report).write_text(json.dumps({"meta": meta, "layers": rep}, indent=1))
    print("aligned", len(P), "positions; first divergence at", d, "orig len", len(o0["input_ids"]), "expl len", len(x0["input_ids"]))


if __name__ == "__main__":
    main()
