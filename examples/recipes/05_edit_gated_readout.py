"""Recipe 5: a gated rank-1 readout change: push one output, only in the states that need it.

Credit: the blind cube-then-banana editor of the controlled study (study/results/ctrl/ctb/, scripts/build_p1b.py).
Its 1,024-value edit is the one autonomous edit that passed held-out scoring: 11/64 -> 27/64, p = 0.0037.

When: the failure lands on one output channel (here the gripper: the policy opens it too early), and the action
expert's final hidden state already tells the failing decisions apart from those that must not change. See
docs/COOKBOOK.md, section 4.2.

Run:  bin/owsp new cookbook-readout --note "recipe 5"
      bin/owsp py cookbook-readout examples/recipes/05_edit_gated_readout.py

How it works. The action head computes the flow velocity v = h @ W + b, with h the expert's final hidden state
(``action.head_input`` in captures) and W = ``action.action_out_proj.kernel`` [1024, 32]; dim 7 is the gripper.
The edit adds ``gain * a`` to column 7 only, so the gripper velocity changes by ``gain * (h . a)``: a rank-1 change of
1,024 values. The key ``a`` is the least-norm vector whose mean response is 1 on the target decisions and 0 on each
preservation group, at every captured denoising step. Before solving, it removes the top principal directions of
how h varies across the 15 tokens of one chunk: those carry each token's own noise sample, not the robot's state.

Check it worked: (1) in-sample the key responds about 1 on targets and about 0 on preservation groups; (2) the
split-half check (key built from even-numbered episodes, tested on odd ones, and the reverse) still separates them:
the blind stacking editor's readout failed exactly here; (3) the first-order prediction flips targets and nothing
else. Then replay the candidate (specs/candidate_check.json with ``"cand": "cookbook-readout"``, recipe 9) before
any closed-loop run. The prediction assumes the key is the same at all 10 denoising steps; ctb-ph holds step 0
only, so capture specs/action_readout.json (all steps) for a real build.
"""
import json

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "capture": "ctb-ph",   # base replays of the decisions below, with action.head_input
    "groups": {            # item ranges [start, end) of the capture
        "premature release": [0, 21], "cube carry": [21, 30], "banana approach": [30, 37],
        "banana grasp": [37, 42], "banana carry": [42, 49]},
    "target": "premature release",
    "zero_on": ["banana approach", "banana grasp", "banana carry"],  # mean response 0 here; other groups are free
    "nuisance_pcs": 96,    # within-chunk token-deviation directions removed from the key (no labels used)
    "gain": -1.8,          # v7 += gain * (h . a); negative raises the gripper command toward closed
    "out_dim": 7,          # 7 = gripper; 0-6 are the joints
    "save": True,
}
KERNEL = "action.action_out_proj.kernel"


def capture(aid, keys):
    """(index, {key: array}) of capture ``aid``, reading only ``keys`` (the prelude's ``acts()`` reads every array)."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    z = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in keys}


def solve_key(H, groups, target, zero_on, P):
    """Least-norm a (in the range of P) with mean(H[target]) . a = 1 and mean(H[g]) . a = 0, per denoising step."""
    rows, ys = [], []
    for s in range(H.shape[1]):
        for name, y in [(target, 1.0)] + [(g, 0.0) for g in zero_on]:
            if len(groups[name]):
                rows.append(P @ H[groups[name], s].mean(axis=(0, 1)))
                ys.append(y)
    M, y = np.stack(rows), np.array(ys)
    return P @ (M.T @ np.linalg.solve(M @ M.T, y))


def predict(H, a7, key, gain):
    """First-order gripper prediction: closed decisions before and after (x <- x - 0.1 v over 10 steps, a = (x+1)/2)."""
    k = (H @ key).mean(axis=1)                     # [N, 15] response per token, mean over captured steps
    new = a7 - gain * k / 2.0
    closed = lambda a: (a > 0.5).mean(axis=-1) >= 0.5  # noqa: E731
    return closed(a7), closed(new), k.mean(axis=1)


def report(title, names, groups, before, after, resp):
    print(title)
    for g in names:
        ix = groups[g]
        if len(ix):
            print(f"  {g:<20} n={len(ix):2d}  key {resp[ix].mean():+.2f} (min {resp[ix].min():+.2f}, max "
                  f"{resp[ix].max():+.2f})  gripper closed {int(before[ix].sum()):2d} -> {int(after[ix].sum()):2d}")


def main(cfg=SETTINGS):
    idx, arr = capture(cfg["capture"], ["action.head_input", "actions"])
    H = arr["action.head_input"].astype(np.float64)       # [N, S, 15, D]
    a7 = arr["actions"][:, :, cfg["out_dim"]].astype(np.float64)  # [N, 15] recorded-policy replay
    groups = {g: np.arange(lo, hi) for g, (lo, hi) in cfg["groups"].items()}
    names = list(groups)
    print(f"capture {cfg['capture']}: {H.shape[0]} decisions, denoising steps {idx['denoise_steps']}, "
          f"hidden size {H.shape[-1]}")
    dev = (H - H.mean(axis=2, keepdims=True)).reshape(-1, H.shape[-1])
    _, sv, Vt = np.linalg.svd(dev, full_matrices=False)
    k = int(cfg["nuisance_pcs"])
    P = np.eye(H.shape[-1]) - Vt[:k].T @ Vt[:k]
    print(f"removed {k} token-deviation directions ({100 * (sv[:k] ** 2).sum() / (sv ** 2).sum():.1f}% of the "
          "within-chunk variation)")

    a = solve_key(H, groups, cfg["target"], cfg["zero_on"], P)
    before, after, resp = predict(H, a7, a, cfg["gain"])
    report(f"in-sample (gain {cfg['gain']}): mean key response and predicted gripper-closed decisions", names,
           groups, before, after, resp)

    parity = np.array([int(it["episode"].rsplit("s", 1)[-1]) % 2 for it in idx["items"]])
    for fit in (0, 1):
        g_fit = {g: ix[parity[ix] == fit] for g, ix in groups.items()}
        g_test = {g: ix[parity[ix] != fit] for g, ix in groups.items()}
        a_half = solve_key(H, g_fit, cfg["target"], cfg["zero_on"], P)
        b, aft, r = predict(H, a7, a_half, cfg["gain"])
        report(f"split-half: key from {'even' if fit == 0 else 'odd'} episodes, tested on the others", names, g_test,
               b, aft, r)

    W = base(KERNEL).astype(np.float64)
    col = cfg["out_dim"]
    new = W.copy()
    new[:, col] = W[:, col] + cfg["gain"] * a
    new = bf16(new)
    changed = int(np.count_nonzero(new != base(KERNEL)))
    print(f"\nedit: {KERNEL}[:, {col}] += {cfg['gain']} * key; |change| {np.linalg.norm(cfg['gain'] * a):.2f} "
          f"vs column norm {np.linalg.norm(W[:, col]):.2f}; {changed} stored values change, rank 1")
    if cfg["save"]:
        save(KERNEL, new)
        print("saved. Next: replay this candidate on the same decisions (specs/candidate_check.json, recipe 9).")
    return a


if __name__ == "__main__":
    main()
