"""Cross-machine check of GR00T action chunks on recorded requests (numpy only).

Compares replays written by groot_acts.py --actions-only (exact recorded images, state, noise seed) with the chunks the
ali-ms server returned in the closed loop (``actions`` in the obs npz: 7 absolute joints + gripper, 40 rows), and
optionally with a second replay directory (same items, same batching) from another machine. Executed rows 0-7 only.

  python3 verify_actions.py REPLAY_DIR [OTHER_REPLAY_DIR] [--wording orig] [--out f.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(d, wording):
    idx = json.loads((Path(d) / "index.json").read_text())
    f = {x["item"]: x["file"] for x in idx["files"] if x["wording"] == wording}
    return idx["items"], f


def stats(a, b):
    dj = np.abs(a[:, :8, :7] - b[:, :8, :7])
    ga, gb = a[:, :8, 7] > 0.5, b[:, :8, 7] > 0.5
    return {"n_chunks": int(len(a)), "joint_max_abs_rad": float(dj.max()), "joint_mean_abs_rad": float(dj.mean()),
            "joint_p99_abs_rad": float(np.quantile(dj, 0.99)), "gripper_rows_equal": f"{int((ga == gb).sum())}/{ga.size}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--wording", default="orig")
    ap.add_argument("--out")
    a = ap.parse_args()
    items, f = load(a.replay, a.wording)
    rep, rec = [], []
    for i, it in enumerate(items):
        z = np.load(f[i])
        rep.append(np.concatenate([z["act_joint_position"], z["act_gripper_position"]], axis=1))
        rec.append(np.load(it["obs"])["actions"][int(it["k"])])
    rep, rec = np.stack(rep), np.stack(rec)
    out = {"replay_vs_recorded_closed_loop": stats(rep, rec)}
    if a.other:
        items2, f2 = load(a.other, a.wording)
        assert [(Path(x["obs"]).name, x["k"]) for x in items2] == [(Path(x["obs"]).name, x["k"]) for x in items]
        oth = np.stack([np.concatenate([np.load(f2[i])["act_joint_position"], np.load(f2[i])["act_gripper_position"]], axis=1)
                        for i in range(len(items2))])
        out["replay_vs_other_replay"] = stats(rep, oth)
        out["other_replay_vs_recorded_closed_loop"] = stats(oth, rec)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
