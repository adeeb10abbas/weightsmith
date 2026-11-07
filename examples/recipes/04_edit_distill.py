"""Recipe 4: distil a known-good wording into the weights (minimum-norm key -> value update).

Credit: the operator session's edit in the paper (Section 2; study/results/opus/ctb/v2mi_edit.py), using the
multi-key closed form of ROME/MEMIT (Meng et al. 2022, 2023). ``weightsmith distill`` packages the same build.

When: a rewording makes the same weights decide correctly on the failing decisions (recipe 2), but the instruction
the robot receives cannot change. The edit makes the network process the original wording the way it processes
the known-good one. See docs/COOKBOOK.md, section 4.1.

Run:  bin/owsp new cookbook-distill --note "recipe 4"
      bin/owsp py cookbook-distill examples/recipes/04_edit_distill.py

How it works. For each edited VLM layer L (``language.blocks.LL.mlp.linear``, the MLP down-projection, [16384, 2048],
applied as ``y = h @ W``) and each aligned position p (an image cell, or a prompt/state token present in both
wordings):

* key   K[p] = mean MLP hidden activation (the down-projection's input) at p under the ORIGINAL wording;
* value V[p] = alpha * (d_L[p] - d_{L-1}[p]), where d_L[p] is the mean residual difference (known-good minus
  original) after layer L, so V is the part of the gap that layer L itself adds;
* update dW = K^T (K K^T + lam I)^-1 V with lam = lam_rel * trace(K K^T) / n: the smallest change with K dW ~ V.

Check it worked: every layer maps its keys close to their values (fit error 3e-3 in the middle layers; up to about
0.2 in layers 15-16, where the keys are nearly collinear and the ridge takes over), and the fingerprint printed at
the end identifies the edit. With the default settings this rebuilds the paper's 0.16% variant ``abl-imgl8``
bit for bit (fingerprint 5bf8c4329c1e416a; with ``prompt_slots: "pi05"`` and layers 0-16 it rebuilds ``v2mi-f``,
63db0c52a51306c9). The real test is offline replay of the edited policy on target and preservation decisions
(spec ``specs/candidate_check.json``, then recipe 9), then closed loop.
"""
import hashlib
import json
import os

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures
EDITS_DIR = "/work/edits"  # the candidate's edited tensors

SETTINGS = {
    "capture": "c2-multi",            # capture holding the decisions under the original wording
    "explicit_capture": None,         # capture with the known-good wording on the same decisions (None = same one)
    "orig_items": [0, 1, 2, 3, 4, 5],          # items replayed under the original wording
    "expl_items": [12, 13, 14, 15, 16, 17],    # the same decisions, in the same order, under the known-good wording
    "layers": [8, 9, 10, 11, 12, 13, 14, 15, 16],  # VLM layers to edit (0..16; layer 17's MLP never reaches actions)
    "image_cells": True,              # keys at the 2 cameras x 16 image cells
    "prompt_slots": None,             # None, "pi05" (<bos> Task : and State ... Action:) or [[orig, expl], ...]
    "alpha": 1.0,                     # 1 = copy the full effect of the known-good wording
    "lam_rel": 1e-3,                  # ridge, relative to the mean squared key norm
    "compose": False,                 # True: add to this candidate's current edits instead of the base weights
    "save": True,
}
MLP_DOWN = "language.blocks.{:02d}.mlp.linear"
STATE_TOKEN = 3040  # PaliGemma id of "▁State", where the robot-state part of every pi0.5 prompt starts


def capture(aid, keys):
    """(index, {key: array}) of capture ``aid``, reading only ``keys`` (the prelude's ``acts()`` reads every array)."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    z = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise SystemExit(f"capture {aid} lacks {missing}: capture with image_tokens (image cells) or prompt_tokens "
                         "0..63 (prompt slots); see specs/image_tokens.json and specs/prompt_tokens.json")
    return idx, {k: z[k] for k in keys}


def pi05_slot_map(o_item, x_item):
    """Aligned (orig, expl) prompt positions outside the instruction: the 3 opening tokens and State ... Action:."""
    a, b = o_item["prompt_token_ids_at_slots"], x_item["prompt_token_ids_at_slots"]
    na, nb = sum(o_item["prompt_token_valid_at_slots"]), sum(x_item["prompt_token_valid_at_slots"])
    if na == len(a) or nb == len(b):
        raise SystemExit("prompt_slots 'pi05' needs captures of prompt slots 0..63 that cover the whole prompt")
    sa, sb = a.index(STATE_TOKEN), b.index(STATE_TOKEN)
    pairs = [(i, i) for i in range(3)] + [(sa + j, sb + j) for j in range(na - sa)]
    bad = [(i, j) for i, j in pairs if a[i] != b[j]]
    if bad or na - sa != nb - sb:
        raise SystemExit(f"the two prompts do not share the template positions {bad[:3]}")
    return pairs


def fingerprint(edits_dir=None):
    """The owsp fingerprint of a candidate: sha256 over the sorted edited files' names and bytes, 16 hex digits."""
    edits_dir = edits_dir or EDITS_DIR
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(edits_dir) if f.endswith(".npy")):
        h.update(name.encode())
        with open(os.path.join(edits_dir, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def main(cfg=SETTINGS):
    use_img, use_slots = bool(cfg["image_cells"]), cfg["prompt_slots"] is not None
    if not (use_img or use_slots):
        raise SystemExit("choose image_cells, prompt_slots or both")
    keys = (["vlm.img_mlp_hidden", "vlm.img_resid_out"] if use_img else []) + \
           (["vlm.mlp_hidden", "vlm.resid_out"] if use_slots else [])
    idx_o, arr_o = capture(cfg["capture"], keys)
    idx_x, arr_x = (idx_o, arr_o) if not cfg["explicit_capture"] else capture(cfg["explicit_capture"], keys)
    oi, xi = list(cfg["orig_items"]), list(cfg["expl_items"])
    if len(oi) != len(xi) or not oi:
        raise SystemExit("orig_items and expl_items must name the same decisions, in the same order")
    for a, b in zip(oi, xi, strict=True):
        ia, ib = idx_o["items"][a], idx_x["items"][b]
        if (ia["episode"], ia["k"]) != (ib["episode"], ib["k"]):
            raise SystemExit(f"item {a} and item {b} are different decisions")
    print(f"original:   {idx_o['items'][oi[0]]['prompt']!r} on {len(oi)} decisions")
    print(f"known-good: {idx_x['items'][xi[0]]['prompt']!r}")
    pos = {L: i for i, L in enumerate(idx_o["vlm_layers"])}
    need = sorted(set(cfg["layers"]) | {L - 1 for L in cfg["layers"] if L > 0})
    if any(L not in pos for L in need) or any(not 0 <= L <= 16 for L in cfg["layers"]):
        raise SystemExit(f"edited layers must be in 0..16 and the capture must hold each and the one before: {need}")
    og = {k: v[oi].astype(np.float64) for k, v in arr_o.items()}  # original wording, chosen decisions
    kg = {k: v[xi].astype(np.float64) for k, v in arr_x.items()}  # known-good wording, same decisions
    if cfg["prompt_slots"] == "pi05":
        slots = pi05_slot_map(idx_o["items"][oi[0]], idx_x["items"][xi[0]])
    else:
        slots = [tuple(p) for p in cfg["prompt_slots"] or []]
    so, sx = idx_o["prompt_token_slots"], idx_x["prompt_token_slots"]
    slots = [(so.index(a), sx.index(b)) for a, b in slots]  # prompt positions -> capture indices
    cells = [(c, e) for c in (0, 1) for e in range(1, 17)] if use_img else []

    def gap(L):  # known-good minus original residual after layer L, one row per aligned position
        li = pos[L]
        rows = [(kg["vlm.resid_out"][:, li, b] - og["vlm.resid_out"][:, li, a]).mean(0) for a, b in slots]
        rows += [(kg["vlm.img_resid_out"][:, li, c, e] - og["vlm.img_resid_out"][:, li, c, e]).mean(0)
                 for c, e in cells]
        return np.stack(rows)

    def keys_at(L):  # original-wording MLP hidden activations at the same positions
        li = pos[L]
        rows = [og["vlm.mlp_hidden"][:, li, a].mean(0) for a, _ in slots]
        rows += [og["vlm.img_mlp_hidden"][:, li, c, e].mean(0) for c, e in cells]
        return np.stack(rows)

    gaps = {L: gap(L) for L in need}
    total = 0
    print(f"{len(slots)} prompt/state positions + {len(cells)} image cells = {len(slots) + len(cells)} keys per layer")
    for L in cfg["layers"]:
        K = keys_at(L)
        V = cfg["alpha"] * (gaps[L] - (gaps[L - 1] if L > 0 else 0))
        G = K @ K.T
        lam = cfg["lam_rel"] * np.trace(G) / len(G)
        dW = K.T @ np.linalg.solve(G + lam * np.eye(len(G)), V)
        name = MLP_DOWN.format(L)
        W0 = load(name) if cfg["compose"] else base(name)
        new = bf16(W0 + dW)
        changed = int(np.count_nonzero(new != W0))
        total += changed
        print(f"L{L:2d}  fit error {np.linalg.norm(K @ dW - V) / np.linalg.norm(V):.1e}   |dW|/|W| "
              f"{np.linalg.norm(dW) / np.linalg.norm(W0):.4f}   rank <= {len(K)}   stored values changed "
              f"{changed:,} ({100 * changed / new.size:.1f}%)")
        if cfg["save"]:
            save(name, new)
    factors = (len(slots) + len(cells)) * (16384 + 2048) * len(cfg["layers"])
    print(f"\n{len(cfg['layers'])} tensors, {total:,} stored values changed; as low-rank factors {factors:,} values")
    if cfg["save"]:
        print(f"candidate fingerprint {fingerprint()}  (abl-imgl8 = 5bf8c4329c1e416a, v2mi-f = 63db0c52a51306c9)")
        print("next: owsp diff, then replay target and preservation decisions (specs/candidate_check.json, recipe 9)")


if __name__ == "__main__":
    main()
