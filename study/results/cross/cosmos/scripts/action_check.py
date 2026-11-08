"""Action-mapping check from recorded requests (cosmos venv; numpy + Pillow).

For every recorded episode (obs_r0_e{env}.npz written by cosmos_client.py): the server returns 32 rows of
[7 absolute joint positions (rad), gripper] per request, the client executes them open loop and binarises the gripper
at 0.5. If the mapping is right, (a) the first commanded joint target lies close to the measured joints at the request,
(b) the measured joints at the next request lie close to the last commanded target of the previous chunk, and (c) the
measured gripper follows the binarised command. Writes a JSON summary and, with --frames, contact sheets of the
conditioning composites (wrist on top, left|right shoulder views below) for a few episodes.

usage: action_check.py OUT.json OBS_DIR [OBS_DIR ...] [--frames DIR --episodes N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check_episode(path: Path) -> dict:
    d = np.load(path)
    q, g, a = d["joint"].astype(np.float64), d["gripper"].astype(np.float64).reshape(-1), d["actions"].astype(np.float64)
    k = len(q)
    first = np.abs(a[:, 0, :7] - q).max(1) if k else np.zeros(0)                # target 1 vs measured now
    track = np.abs(a[:-1, -1, :7] - q[1:]).max(1) if k > 1 else np.zeros(0)     # last target vs measured next
    cmd_close = (a[:, :, 7] > 0.5)
    last_cmd = cmd_close[:-1, -1] if k > 1 else np.zeros(0, bool)
    meas_next = g[1:] if k > 1 else np.zeros(0)
    return {"episode": path.name, "requests": int(k), "steps": [int(s) for s in d["step"][:3]],
            "first_target_err_rad_median": float(np.median(first)) if k else None,
            "first_target_err_rad_max": float(first.max()) if k else None,
            "track_err_rad_median": float(np.median(track)) if len(track) else None,
            "track_err_rad_p90": float(np.quantile(track, 0.9)) if len(track) else None,
            "gripper_cmd_close_frac": float(cmd_close.mean()) if k else None,
            "gripper_meas_when_cmd_close": float(meas_next[last_cmd].mean()) if last_cmd.any() else None,
            "gripper_meas_when_cmd_open": float(meas_next[~last_cmd].mean()) if (~last_cmd).any() else None,
            "joint_range_rad": [float(x) for x in (q.max(0) - q.min(0))] if k else None}


def contact_sheet(path: Path, out: Path, every: int = 2, width: int = 320) -> None:
    from PIL import Image
    d = np.load(path)
    imgs = d["image"][::every]
    tiles = [Image.fromarray(im).resize((width, int(width * im.shape[0] / im.shape[1]))) for im in imgs]
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    W, H = tiles[0].size
    sheet = Image.new("RGB", (cols * W, rows * H), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * W, (i // cols) * H))
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("obs_dirs", nargs="+")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    files = sorted(f for d in a.obs_dirs for f in Path(d).rglob("obs_r*_e*.npz"))
    eps = [check_episode(f) | {"file": str(f)} for f in files]
    agg = {}
    for key in ("first_target_err_rad_median", "track_err_rad_median", "track_err_rad_p90",
                "gripper_meas_when_cmd_close", "gripper_meas_when_cmd_open"):
        v = [e[key] for e in eps if e[key] is not None]
        agg[key] = {"median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)} if v else None
    Path(a.out).write_text(json.dumps({"episodes": len(eps), "aggregate": agg, "per_episode": eps}, indent=1))
    print(json.dumps({"episodes": len(eps), "aggregate": agg}, indent=1))
    if a.frames:
        Path(a.frames).mkdir(parents=True, exist_ok=True)
        for f in files[: a.episodes]:
            contact_sheet(f, Path(a.frames) / (f.parent.parent.name.replace(".obs", "") + f"_{f.parent.name}_{f.stem}.jpg"))


if __name__ == "__main__":
    main()
