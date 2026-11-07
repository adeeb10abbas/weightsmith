"""Recipe 3: inspect activation captures, rebuild the spec that made one, and check a spec's items.

When: before writing a capture spec (see what an existing capture holds and costs), when reusing someone else's
capture (which arrays, layers, steps, wordings?), and before spending activation budget on a new spec (do its
episodes and replans exist?). See docs/COOKBOOK.md, section 3, and the specs in examples/recipes/specs/.

Run (read-only):  bin/owsp py base examples/recipes/03_capture_info.py

For each capture it prints the candidate, items per wording, layers, denoising steps, every array with its shape
and size (read from the file headers, without loading the data), the replay parity, and a spec that reproduces
the capture. With ``check_items`` it confirms that each item's episode was recorded and its replan exists.

Check it worked: the array list matches the layout in docs/COOKBOOK.md (for example ``c2-multi`` holds 48 items
and about 3.2 GB: VLM prompt and image tokens at all 18 layers), and ``parity_err`` is at most about 0.1 for
base captures replayed under the recorded wording (it is meaningless when the wording was changed).
"""
import json
import zipfile

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "captures": ["c2-multi", "c1-attn", "ctb-ph", "chk-v2mi-f"],
    "print_spec": "c2-multi",  # print the full rebuilt spec of this capture (None = only a summary for each)
    # items to check against the recorded episodes, e.g. pasted from a spec in examples/recipes/specs/
    "check_items": [{"episode": "base.base.s20001", "k": 5}, {"episode": "base.base.s20011", "k": 4},
                    {"episode": "base.base.s20011", "step": 180}],
}


def array_headers(path):
    """{name: (shape, dtype)} of the arrays in an .npz, read from the headers only."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else \
                    np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(f)
            out[name[:-4] if name.endswith(".npy") else name] = (tuple(shape), np.dtype(dtype))
    return out


def recorded_prompt(eid):
    try:
        with open(f"{DATA_ROOT}/episodes/{eid}/summary.json") as f:
            return json.load(f).get("prompt")
    except OSError:
        return None


def rebuild_spec(aid, idx, headers):
    """A spec that captures the same arrays on the same items (per-item prompts only where they differ)."""
    items = []
    for it in idx["items"]:
        item = {"episode": it["episode"], "k": int(it["k"])}
        if it.get("prompt") and it["prompt"] != recorded_prompt(it["episode"]):
            item["prompt"] = it["prompt"]
        items.append(item)
    spec = {"id": f"{aid}-again", "cand": idx["cand"], "items": items, "action_layers": idx["action_layers"],
            "vlm_layers": idx["vlm_layers"], "denoise_steps": idx["denoise_steps"],
            "action_tokens": idx["action_tokens"], "prompt_tokens": idx["prompt_token_slots"]}
    if "vlm.img_mlp_hidden" in headers:
        spec["image_tokens"] = True
    if "action.attn_mass" in headers:
        spec["action_attention"] = True
    return spec


def describe(aid):
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    headers = array_headers(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    n = len(idx["items"])
    prompts = {}
    for it in idx["items"]:
        prompts[it["prompt"]] = prompts.get(it["prompt"], 0) + 1
    total = sum(int(np.prod(s)) * d.itemsize for s, d in headers.values())
    vl = idx["vlm_layers"]
    print(f"\n== {aid}: {n} items, candidate {idx['cand']!r}, {total / 1e6:,.0f} MB ({total / max(n, 1) / 1e6:.1f} MB "
          f"per item), parity_err {idx.get('base_parity_max_abs_action_error', float('nan')):.3f}")
    print(f"   action layers {idx['action_layers']}, VLM layers {vl if len(vl) < 6 else f'{vl[0]}-{vl[-1]}'}, "
          f"denoising steps {idx['denoise_steps']} (t = {idx['denoise_t']}), action tokens {idx['action_tokens']}, "
          f"{len(idx['prompt_token_slots'])} prompt slots")
    for p, c in prompts.items():
        print(f"   {c:3d} items with {p!r}")
    for name, (shape, dtype) in sorted(headers.items()):
        size = int(np.prod(shape)) * dtype.itemsize
        print(f"   {name:<22} {str(list(shape)):<28} {dtype}  {size / 1e6:9.1f} MB")
    pieces = idx.get("prompt_token_pieces_item0")
    if isinstance(pieces, list):
        shown = [f"{i}:{p}" for i, p in enumerate(pieces) if p != "<pad>"]
        print("   prompt slots of item 0: " + " ".join(shown))
    return idx, headers


def check_items(items):
    bad = 0
    for it in items:
        eid = it["episode"]
        try:
            o = np.load(f"{DATA_ROOT}/episodes/{eid}/obs.npz")
            steps, grip = o["step"], o["gripper"]
        except OSError:
            print(f"   MISSING  {eid}: no recorded obs (record it with `owsp eval` first)")
            bad += 1
            continue
        k = int(it["k"]) if "k" in it else int(np.searchsorted(steps, int(it["step"]), side="right") - 1)
        if not 0 <= k < len(steps):
            print(f"   BAD      {eid}: replan {k} outside 0..{len(steps) - 1}")
            bad += 1
            continue
        print(f"   ok       {eid} k={k} (sim step {int(steps[k])}, measured gripper {float(grip[k][0]):.3f})"
              + (f", prompt {it['prompt']!r}" if it.get("prompt") else ""))
    return bad


def main(cfg=SETTINGS):
    out = {}
    for aid in cfg["captures"]:
        idx, headers = describe(aid)
        spec = rebuild_spec(aid, idx, headers)
        out[aid] = spec
        if aid == cfg.get("print_spec"):
            print(f"\nspec that reproduces {aid} (save as JSON, then `owsp acts spec.json`; check it with "
                  "examples/recipes/check_spec.py):")
            print(json.dumps(spec))
    if cfg.get("check_items"):
        print(f"\nchecking {len(cfg['check_items'])} items against the recorded episodes:")
        bad = check_items(cfg["check_items"])
        print(f"   {len(cfg['check_items']) - bad} ok, {bad} not usable")
    return out


if __name__ == "__main__":
    main()
