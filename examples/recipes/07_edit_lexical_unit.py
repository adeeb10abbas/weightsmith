"""Recipe 7: strengthen a lexical unit: make one instruction word count for more.

Credit: the blind mustard editor of the controlled study (study/results/ctrl/mustard/, scripts/mk_p1.py), who gave
two layer-0 units that detect the token "left" output rows that push "left" further from "right" (4,094 values).
The cube-left-of-bowl editor found the same two units (3097 and 8023) on its own task.

When: a prompt swap (recipe 2) shows that one word steers the decision but too weakly: in the mustard task "left"
sent only 8 of 32 replayed carries left (16 still went right), while "right" sent 27 of 32 right. The edit
extrapolates the word's representation past its contrast word. See docs/COOKBOOK.md, section 4.4.

Run:  bin/owsp new cookbook-lexical --note "recipe 7"
      bin/owsp py cookbook-lexical examples/recipes/07_edit_lexical_unit.py

How it works. Row u of ``language.blocks.LL.mlp.linear`` is the residual direction unit u writes, scaled by its
activation. A unit that fires only on one token is a private write port into that token's residual stream:
adding ``gamma * dr`` to its row adds about ``beta * dr`` to that token and almost nothing anywhere else. Here dr is
the mean residual difference (token wording minus contrast wording) at the token's position after layer LL, and
``gamma = beta / (sum of the units' activations on the token)``. Needs a capture with ``prompt_tokens`` covering the
token (specs/prompt_tokens.json) under both wordings; ``image_tokens`` adds a leakage check on image cells.

Check it worked: the realized change at the token points along dr (cosine about 1, size about beta * |dr|), and the
leakage onto every other token and image cell is small next to it. The mustard edit then did what it targeted
(closed-loop carries went left in 30 of 32 episodes, against 13 of 32) yet held-out success moved only 8 -> 11 of
64 (p = 0.61), because the cap grasp still slipped: check the decision AND the outcome.

The default run is a demonstration on cube-then-banana, with an untested hypothesis: replayed under "Put the cube
and the banana in the bowl", 12 of the 21 premature releases become holds (recipe 2), so it moves the instruction's
"then" toward "and" (beta = -1) through the one layer-0 unit that detects "then". Whether that helps in closed loop
is unknown; the left-of-bowl editor found that downstream layers ignored a similar layer-0 change.
"""
import json

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "capture": "c2-multi",                   # needs vlm.mlp_hidden and vlm.resid_out at the token's prompt slot
    "token_prompt": "Put the cube then the banana in the bowl",      # wording that holds the token to change
    "contrast_prompt": "Put the cube and the banana in the bowl",    # same wording, contrast token at that slot
    "slot": None,          # prompt slot of the token; None = the first slot where the two wordings differ
    "layer": 0,            # VLM layer whose MLP holds the detector units
    "min_activation": 0.5,  # a detector fires at least this much on the token ...
    "min_ratio": 10.0,     # ... and at least this many times more than on any other token or image cell
    "max_units": 4,
    "beta": -1.0,          # the token's residual moves by beta * (token - contrast): > 0 stronger, -1 = contrast
    "save": True,
}
MLP_DOWN = "language.blocks.{:02d}.mlp.linear"


def capture(aid, keys):
    """(index, {key: array}) of capture ``aid``, reading only ``keys`` (the prelude's ``acts()`` reads every array)."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    z = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in keys if k in z.files}


def main(cfg=SETTINGS):
    idx, arr = capture(cfg["capture"], ["vlm.mlp_hidden", "vlm.resid_out", "vlm.img_mlp_hidden"])
    if "vlm.mlp_hidden" not in arr:
        raise SystemExit("the capture has no prompt-token activations (capture with vlm_layers and prompt_tokens)")
    items = idx["items"]
    T = [i for i, it in enumerate(items) if it["prompt"] == cfg["token_prompt"]]
    C = [i for i, it in enumerate(items) if it["prompt"] == cfg["contrast_prompt"]]
    if not T or not C:
        raise SystemExit(f"need items under both wordings; wordings: {sorted({it['prompt'] for it in items})}")
    li = idx["vlm_layers"].index(cfg["layer"])
    slots = idx["prompt_token_slots"]
    ids_t, ids_c = items[T[0]]["prompt_token_ids_at_slots"], items[C[0]]["prompt_token_ids_at_slots"]
    if cfg["slot"] is None:
        diff = [j for j, (a, b) in enumerate(zip(ids_t, ids_c, strict=True)) if a != b]
        if not diff:
            raise SystemExit("the two wordings have identical tokens at the captured slots")
        j = diff[0]
    else:
        j = slots.index(cfg["slot"])
    H = arr["vlm.mlp_hidden"][:, li].astype(np.float32)   # [N, P, F]
    Rz = arr["vlm.resid_out"][:, li].astype(np.float64)   # [N, P, D]
    pieces = idx.get("prompt_token_pieces_item0")
    print(f"layer {cfg['layer']}, prompt slot {slots[j]}: token id {ids_t[j]} under {cfg['token_prompt']!r} vs "
          f"{ids_c[j]} under {cfg['contrast_prompt']!r}" + (f" (item 0 has {pieces[j]!r} there)" if pieces else ""))

    h_tok = H[T, j]                                        # [nT, F]
    valid = np.array([it["prompt_token_valid_at_slots"] for it in items], dtype=bool)  # [N, P]
    ids = np.array([it["prompt_token_ids_at_slots"] for it in items])
    same = valid & (ids == ids_t[j])                       # this token wherever it occurs, in any wording
    others = H[valid & ~same]                              # every other real token: a detector stays silent there
    other_max = np.abs(others).max(0)
    if "vlm.img_mlp_hidden" in arr:
        img = arr["vlm.img_mlp_hidden"][:, li].astype(np.float32).reshape(-1, H.shape[-1])
        other_max = np.maximum(other_max, np.abs(img).max(0))
    mean_tok = h_tok.mean(0)
    ok = np.where((mean_tok >= cfg["min_activation"]) & (other_max * cfg["min_ratio"] <= mean_tok))[0]
    units = [int(u) for u in ok[np.argsort(-mean_tok[ok])][: cfg["max_units"]]]
    if not units:
        raise SystemExit("no unit fires selectively on this token at this layer; try another layer")
    for u in units:
        print(f"  unit {u:5d}: {mean_tok[u]:.2f} on the token, at most {other_max[u]:.3f} on any other token"
              + (" or image cell" if "vlm.img_mlp_hidden" in arr else ""))
    elsewhere = sorted({items[i]["prompt"] for i in np.where(same.any(1))[0]} - {cfg["token_prompt"]})
    if elsewhere:
        act = H[same][:, units].sum(-1)
        print(f"  the same token occurs in {len(elsewhere)} other captured wording(s), where these units fire too "
              f"(sum {act.min():.1f} to {act.max():.1f}): the edit acts wherever the word appears, e.g. "
              f"{elsewhere[0]!r}")

    dr = Rz[T, j].mean(0) - Rz[C, j].mean(0)
    gamma = cfg["beta"] / float(mean_tok[units].sum())
    name = MLP_DOWN.format(cfg["layer"])
    W0 = base(name)
    W = W0.astype(np.float64)
    for u in units:
        W[u] += gamma * dr
    W = bf16(W)
    dW = (W.astype(np.float64) - W0)[units]                # only these rows change
    add = (h_tok[:, units] @ dW).mean(0)
    cos = float(add @ dr / (np.linalg.norm(add) * np.linalg.norm(dr)))
    leak = np.linalg.norm(others[:, units] @ dW, axis=-1)
    size = np.linalg.norm(add) / np.linalg.norm(dr)
    print(f"contrast |dr| = {np.linalg.norm(dr):.1f} (token residual norm "
          f"{np.linalg.norm(Rz[T, j], axis=-1).mean():.0f}); realized change at the token: cosine {cos:.3f} with dr, "
          f"size {size:.2f} x |dr|")
    print(f"leakage on {len(others)} other token positions: max {leak.max():.2f}, mean {leak.mean():.3f} (vs "
          f"{np.linalg.norm(add):.0f} added at the token)")
    if "vlm.img_mlp_hidden" in arr:
        li_img = np.linalg.norm(img[:, units] @ dW, axis=-1)
        print(f"leakage on {len(img)} image cells: max {li_img.max():.2f}")
    changed = int(np.count_nonzero(W != W0))
    print(f"edit: {len(units)} rows of {name}, {changed} stored values changed, relative change of the tensor "
          f"{np.linalg.norm(W.astype(np.float64) - W0) / np.linalg.norm(W0):.3f}")
    if cfg["save"]:
        save(name, W)
        print("saved. Next: replay the decisions the word steers (recipe 9), then closed loop.")
    return units


if __name__ == "__main__":
    main()
