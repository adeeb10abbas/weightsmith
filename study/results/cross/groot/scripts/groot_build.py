"""Closed-form instruction distillation into GR00T N1.7 (CPU; Isaac-GR00T venv for torch.save).

The update is weightsmith/distill.py's (paper Eq. 1): per edited layer, keys K (rows: mean MLP down-projection inputs
under the ORIGINAL wording at aligned positions, averaged over the build decisions of a group) and values V (the
explicit wording's residual increment of that layer minus the original's, at the same positions, i.e. the part of the
gap the layer itself adds), dW = K^T (K K^T + lam I)^-1 V, lam = lam_rel tr(K K^T)/n, float64, applied once as
bf16(W0 + dW) to the weight in the y = h @ W convention (nn.Linear stores W^T).

Sites
* ``backbone`` (default): Cosmos-Reason2's Qwen3-VL LLM layers 0..15, ``mlp.down_proj`` (6144 -> 2048). Aligned
  positions are every position that exists under both wordings: the chat-template prefix, the two cameras' image
  tokens pooled into HxW cells per camera (as the paper's 2x16 image cells), the instruction tokens before the
  first difference, the instruction tokens after it that a longest-common-subsequence alignment of the token ids
  matches, and the template tokens after the instruction. In a causal backbone whose image tokens precede the
  instruction, the prefix, image and leading-instruction values are exactly zero, so those keys act as preservation
  constraints; the matched later tokens and the template tail carry the change.
* ``dit``: the action head's 32 DiT blocks, ``ff.net.2`` (6144 -> 1536, ungated residual), keys and values at every
  (denoising step, state/action token): 4 x 41 = 164 per block, as weightsmith's action-expert arm A1 (separate keys per
  step because the block input depends on the flow-matching time and the noisy actions).

``--random`` writes the matched random-direction control (paper App. C): per layer the same singular values (rank,
Frobenius norm) along random orthonormal directions (numpy default_rng(0)), as study/results/opus/controls/ctl_rand.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_PAD = 151655  # <|image_pad|> in the Qwen3-VL tokenizer


def lcs_pairs(a, b):
    """Index pairs of one longest common subsequence of two token lists (DP; ties prefer earlier tokens)."""
    n, m = len(a), len(b)
    T = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            T[i, j] = T[i + 1, j + 1] + 1 if a[i] == b[j] else max(T[i + 1, j], T[i, j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif T[i + 1, j] >= T[i, j + 1]:
            i += 1
        else:
            j += 1
    return out


def align(ids_o, ids_x):
    """Original -> explicit map of aligned positions; returns (first difference, {p: q}, image positions)."""
    ids_o, ids_x = [int(t) for t in ids_o], [int(t) for t in ids_x]
    d = 0
    while d < min(len(ids_o), len(ids_x)) and ids_o[d] == ids_x[d]:
        d += 1
    pmap = {p: p for p in range(d)}
    for i, j in lcs_pairs(ids_o[d:], ids_x[d:]):
        pmap[d + i] = d + j
    return d, pmap


def image_cells(ids, n_cams=2, grid=(8, 14), cells=(4, 7)):
    """Token positions of each image cell: per camera, its grid of merged tokens split into cells (row-major)."""
    pos = [i for i, t in enumerate(ids) if int(t) == IMAGE_PAD]
    per = len(pos) // n_cams
    gh, gw = grid
    assert gh * gw == per, (gh, gw, per)
    ch, cw = cells
    out = []
    for c in range(n_cams):
        cam = np.array(pos[c * per:(c + 1) * per]).reshape(gh, gw)
        for r in np.array_split(np.arange(gh), ch):
            for q in np.array_split(np.arange(gw), cw):
                out.append(cam[np.ix_(r, q)].ravel().tolist())
    return out


def closed_form(K, V, lam_rel=1e-3):
    K, V = K.astype(np.float64), V.astype(np.float64)
    G = K @ K.T
    lam = lam_rel * np.trace(G) / len(G)
    A = np.linalg.solve(G + lam * np.eye(len(G)), V)
    return K.T, A  # dW = K.T @ A  [d_in, d_out]


def load_pair(files, item, orig, expl):
    return np.load(files[(item, orig)]), np.load(files[(item, expl)])


def backbone_kv(groups, files, a, layers):
    first = next(iter(groups.values()))[0]
    o0, x0 = load_pair(files, first, a.orig, a.expl)
    ids_o, ids_x = o0["input_ids"], x0["input_ids"]
    d, pmap = align(ids_o, ids_x)
    img = set(i for i, t in enumerate(ids_o) if int(t) == IMAGE_PAD)
    tok_pos = [p for p in pmap if p not in img]                     # template + instruction positions
    cells = [] if a.image_cells == "0" else image_cells(ids_o, cells=tuple(int(x) for x in a.image_cells.split("x")))
    for c in cells:
        assert all(pmap.get(p) == p for p in c)                     # images precede the instruction: same positions
    Lidx = {int(L): i for i, L in enumerate(o0["lm_layers"])}
    K, V = {L: [] for L in layers}, {L: [] for L in layers}
    for g, its in groups.items():
        Ks, Vs = {L: [] for L in layers}, {L: [] for L in layers}
        for it in its:
            o, x = load_pair(files, it, a.orig, a.expl)
            assert list(o["input_ids"]) == list(ids_o) and list(x["input_ids"]) == list(ids_x)
            for L in layers:
                li = Lidx[L]
                mo, io, ix = o["lm_mlp"][li].astype(np.float64), o["lm_inc"][li].astype(np.float64), x["lm_inc"][li].astype(np.float64)
                k = [mo[p] for p in tok_pos] + [mo[c].mean(0) for c in cells]
                v = [ix[pmap[p]] - io[p] for p in tok_pos] + [(ix[c] - io[c]).mean(0) for c in cells]
                Ks[L].append(np.stack(k))
                Vs[L].append(np.stack(v))
        for L in layers:
            K[L].append(np.mean(Ks[L], axis=0))
            V[L].append(a.alpha * np.mean(Vs[L], axis=0))
    meta = {"first_difference": d, "orig_len": len(ids_o), "expl_len": len(ids_x), "token_positions": tok_pos,
            "token_positions_expl": [pmap[p] for p in tok_pos], "changed_positions": [p for p in tok_pos if p >= d],
            "image_cells": len(cells), "keys_per_group": len(tok_pos) + len(cells),
            "aligned_token_ids": [int(ids_o[p]) for p in tok_pos]}
    name = "backbone.model.model.language_model.layers.{}.mlp.down_proj.weight"
    return K, V, meta, name


def dit_kv(groups, files, a, layers):
    K, V = {L: [] for L in layers}, {L: [] for L in layers}
    for g, its in groups.items():
        Ks, Vs = {L: [] for L in layers}, {L: [] for L in layers}
        for it in its:
            o, x = load_pair(files, it, a.orig, a.expl)
            for L in layers:
                mo = o["dit_mlp"][L].astype(np.float64)              # [steps, tokens, 6144]
                dv = x["dit_inc"][L].astype(np.float64) - o["dit_inc"][L].astype(np.float64)
                Ks[L].append(mo.reshape(-1, mo.shape[-1]))
                Vs[L].append(dv.reshape(-1, dv.shape[-1]))
        for L in layers:
            K[L].append(np.mean(Ks[L], axis=0))
            V[L].append(a.alpha * np.mean(Vs[L], axis=0))
    o0 = np.load(files[(next(iter(groups.values()))[0], a.orig)])
    meta = {"steps": int(o0["dit_mlp"].shape[1]), "tokens": int(o0["dit_mlp"].shape[2]),
            "keys_per_group": int(o0["dit_mlp"].shape[1] * o0["dit_mlp"].shape[2])}
    return K, V, meta, "action_head.model.transformer_blocks.{}.ff.net.2.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="groot_acts.py output dir (orig + expl wordings)")
    ap.add_argument("--items", required=True, help="JSON: {group: [item indices]} or [item indices]")
    ap.add_argument("--site", default="backbone", choices=["backbone", "dit"])
    ap.add_argument("--layers", default=None, help="e.g. 0-15 (backbone default) or 0-31 (dit default)")
    ap.add_argument("--image-cells", default="4x7", help="backbone: per-camera HxW cells over the 8x14 token grid; 0 = none")
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
    files = {(f["item"], f["wording"]): f["file"] for f in index["files"]}
    groups = json.loads(a.items)
    if isinstance(groups, list):
        groups = {"all": groups}
    lo, hi = (int(x) for x in (a.layers or ("0-15" if a.site == "backbone" else "0-31")).split("-"))
    layers = list(range(lo, hi + 1))
    K, V, meta, pname = (backbone_kv if a.site == "backbone" else dit_kv)(groups, files, a, layers)
    rng = np.random.default_rng(0)
    edit, ctl, rep = {}, {}, []
    for L in layers:
        Kl, Vl = np.concatenate(K[L]), np.concatenate(V[L])
        U, A = closed_form(Kl, Vl, a.lam_rel)
        name = pname.format(L)
        edit[name] = {"U": torch.from_numpy(U.astype(np.float32)), "V": torch.from_numpy(A.astype(np.float32))}
        dW = U @ A
        r = {"layer": L, "keys": int(Kl.shape[0]), "fro_dW": float(np.linalg.norm(dW)),
             "fit_rel_err": float(np.linalg.norm(Kl @ dW - Vl) / max(np.linalg.norm(Vl), 1e-12)),
             "V_norm": float(np.linalg.norm(Vl))}
        if a.random:
            Qm, Rr = np.linalg.qr(U)                                     # U = K^T = Q R
            u, s, vt = np.linalg.svd(Rr @ A, full_matrices=False)         # dW = (Q u) diag(s) vt
            k = int((s > s[0] * 1e-6).sum())
            Ur = np.linalg.qr(rng.standard_normal((U.shape[0], k)))[0]
            Vr = np.linalg.qr(rng.standard_normal((A.shape[1], k)))[0]
            ctl[name] = {"U": torch.from_numpy((Ur * s[:k]).astype(np.float32)), "V": torch.from_numpy(Vr.T.astype(np.float32))}
            r.update(rank=k, fro_svd=float(np.sqrt((s[:k] ** 2).sum())), fro_rand=float(np.linalg.norm((Ur * s[:k]) @ Vr.T)))
        rep.append(r)
        print(json.dumps(r), flush=True)
    meta.update({"site": a.site, "capture": str(cap), "groups": groups, "layers": layers, "lam_rel": a.lam_rel,
                 "alpha": a.alpha, "image_cells_grid": a.image_cells, "wordings": index.get("wordings")})
    torch.save({"format": "wsp-groot-edit-v1", "layers": edit, "meta": meta}, a.out)
    if a.random:
        torch.save({"format": "wsp-groot-edit-v1", "layers": ctl,
                    "meta": dict(meta, control="random directions, matched singular values, default_rng(0)")}, a.random)
    if a.report:
        Path(a.report).write_text(json.dumps({"meta": meta, "layers": rep}, indent=1))
    print("site", a.site, "keys per group", meta.get("keys_per_group"), {k: v for k, v in meta.items()
          if k in ("first_difference", "changed_positions", "image_cells", "orig_len", "expl_len")})


if __name__ == "__main__":
    main()
