"""Recipe 2: prompt-swap replay, to find out whether the instruction drives a failing decision.

When: you have target decisions (recipe 1) and want to know if some wording makes the *same weights* decide
correctly. A wording that does is a known-good input, and that is what makes a failure editable (docs/CONCEPTS.md).
See docs/COOKBOOK.md, section 2.

Run (read-only):  bin/owsp py base examples/recipes/02_prompt_swap.py

Input: one or more ``owsp acts`` captures that replay the same recorded decisions (same images, robot state and
noise seed) under several instructions, with a per-item ``"prompt"`` or one capture per wording
(``specs/prompt_swap.json`` is a cheap spec for this: actions only). The recipe groups items by wording, pairs
them by decision (episode, replan) with the reference wording, and reports for each wording how many decisions
keep the gripper closed, how many flip relative to the reference, and how far the chunk's end point moves.

Check it worked: the reference row must reproduce what the policy did live (for base captures the capture's
``parity_err`` is about 0.01-0.1). On cube-then-banana all 21 premature releases release under the original
wording and under renamings of the cube, while wordings that put "in the bowl" next to the cube flip 18-21 of them
to holds: the binding error of the paper's Fig. 2a.
"""
import json

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "captures": ["a1-orig", "a1-v2", "a1-cube", "a1-mirror", "a1-rubiks", "b1-parse", "b2-before", "b2-comma"],
    "reference": "Put the cube then the banana in the bowl",  # the wording the policy is actually given
    "decisions": 21,      # only the reference capture's first N decisions (here the premature releases); None = all
    "goal_xy": [0.455, 0.11],  # optional point (m, robot frame) to report the chunk end's distance to; None = skip
}


def panda_fk(q):
    """Fingertip position (m, robot base frame) of a Franka Panda: modified DH, flange + 0.257 m."""
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(q[i]), np.sin(q[i])
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


def chunk_closed(actions, threshold=0.5):
    """True where a [.., 15, 8] chunk commands the gripper closed (most of its 15 steps above the threshold)."""
    return (np.asarray(actions)[..., 7] > threshold).mean(-1) >= 0.5


def capture(aid, keys):
    """(index, {key: array}) of capture ``aid``, reading only ``keys`` (the prelude's ``acts()`` reads every array)."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    z = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in keys if k in z.files}


def collect(capture_ids):
    """{prompt: {(episode, k): action chunk [15, 8]}} over all items of the captures, plus the reference order."""
    by_prompt, order = {}, {}
    for aid in capture_ids:
        idx, arr = capture(aid, ["actions"])
        for i, it in enumerate(idx["items"]):
            key = (it["episode"], int(it["k"]))
            by_prompt.setdefault(it["prompt"], {})[key] = np.asarray(arr["actions"][i], dtype=np.float64)
            order.setdefault(it["prompt"], [])
            if key not in order[it["prompt"]]:
                order[it["prompt"]].append(key)
    return by_prompt, order


def main(cfg=SETTINGS):
    by_prompt, order = collect(cfg["captures"])
    ref = cfg["reference"]
    if ref not in by_prompt:
        raise SystemExit(f"no item was captured with the reference wording {ref!r}; wordings: {sorted(by_prompt)}")
    keys = order[ref][: cfg["decisions"]] if cfg["decisions"] else order[ref]
    R = by_prompt[ref]
    ref_closed = {k: bool(chunk_closed(R[k])) for k in keys}
    ref_end = {k: panda_fk(R[k][-1, :7]) for k in keys}
    goal = None if cfg.get("goal_xy") is None else np.asarray(cfg["goal_xy"], dtype=float)
    print(f"{len(keys)} reference decisions under {ref!r}: gripper closed in {sum(ref_closed.values())}, "
          f"open in {len(keys) - sum(ref_closed.values())}")
    head = f"{'wording':<58} {'n':>3} {'closed':>6} {'open':>5} {'flips':>5} {'end moved':>9}"
    print(head + ("  to goal" if goal is not None else ""))
    rows = []
    for prompt, chunks in sorted(by_prompt.items(), key=lambda kv: (kv[0] != ref, kv[0])):
        common = [k for k in keys if k in chunks]
        if not common:
            continue
        closed = [bool(chunk_closed(chunks[k])) for k in common]
        flips = sum(c != ref_closed[k] for c, k in zip(closed, common, strict=True))
        moved = np.mean([np.linalg.norm(panda_fk(chunks[k][-1, :7]) - ref_end[k]) for k in common]) * 100
        line = (f"{prompt[:58]:<58} {len(common):>3} {sum(closed):>6} {len(common) - sum(closed):>5} {flips:>5} "
                f"{moved:>6.1f} cm")
        if goal is not None:
            d = np.mean([np.linalg.norm(panda_fk(chunks[k][-1, :7])[:2] - goal) for k in common]) * 100
            line += f"  {d:5.1f} cm"
        print(line)
        rows.append({"prompt": prompt, "n": len(common), "closed": sum(closed), "flips": flips,
                     "end_moved_cm": round(float(moved), 2)})
    print("\nclosed = the chunk keeps the gripper closed (a hold); flips = decisions that differ from the reference "
          "wording on the same inputs.\nA wording that flips most targets the right way is a known-good input: "
          "distil it (recipe 4) or gate on its effect (recipes 5-6).")
    return rows


if __name__ == "__main__":
    main()
