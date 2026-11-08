#!/usr/bin/env python3
"""Where did the mustard end? Per-episode object outcomes of S4-R evaluations from RoboLab's state recordings.

Run with a python that has h5py and numpy (e.g. the RoboLab venv), on the GPU host:

  outcomes.py PROJECT_ROOT POOL_LOG_DIR [POOL_LOG_DIR ...] [--json OUT]

Each pool log dir (weightsmith pool, `rows.json`) names the RoboLab output folder of every episode. RoboLab writes one
HDF5 per run with one demo per environment in export order, so each demo is matched to its environment by comparing
the recorded robot joints with the policy inputs saved for that slot (data/episodes/<eid>/obs.npz). The robot base
sits at the env origin with identity orientation, so env coordinates are the robot frame (x forward, y left).
Labels: placed (task success); right_unstable (mustard ends right of the box but not a stable placement, e.g.
tipped or still held); converse (ends left of the box after being lifted: the converse relation); dropped (lifted,
ends neither); no_lift (never lifted 2 cm); box_moved flags a raisin box displaced by more than 2 cm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def upright_deg(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    zz = 1 - 2 * (x * x + y * y)  # z component of the body z axis in the world
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def episode_index(root: Path) -> list[dict]:
    rows = [json.loads(l) for l in (root / "data/episodes/index.jsonl").read_text().splitlines() if l.strip()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    idx = episode_index(root)
    out = []
    for pool in a.pools:
        info = json.loads((Path(pool) / "rows.json").read_text())
        label = Path(pool).name if Path(pool).name.startswith("pool-") else f"{Path(pool).parent.name}/{Path(pool).name}"
        for r in info["rows"]:
            if not r.get("ok"):
                continue
            slot = f"s{r['slot']}"
            out_dir = Path(r["out_dir"])
            h5 = h5py.File(out_dir / f"run_{r['run']}.hdf5", "r")
            if Path(r.get("obs", "")).exists():  # held-out / control pools keep their policy inputs in the log dir
                cands = [{"eid": f"{label}:{slot}", "obs": Path(r["obs"])}]
            else:  # practice pools: owsp moved the inputs to data/episodes/<eid>/obs.npz
                cand = label.split("-", 2)[2].rsplit("-", 2)[0]  # pool-prac-<cand>-<mmdd>-<hhmmss>
                cands = [dict(e, obs=root / "data/episodes" / e["eid"] / "obs.npz") for e in idx
                         if e["slot"] == slot and e["cand"].split("@")[0] == cand]
            best = None
            for e in cands:  # match (episode, demo) by the robot joints at every replan
                o = np.load(e["obs"])
                J, S = o["joint"], o["step"]
                for dk in h5["data"].keys():
                    jp = h5[f"data/{dk}/states/articulation/robot/joint_position"]
                    n = int(np.searchsorted(S, len(jp)))
                    if n < 2:
                        continue
                    steps = np.clip(S[:n] - 1, 0, len(jp) - 1)
                    err = float(np.abs(jp[steps, :7] - J[:n]).max())
                    if best is None or err < best[0]:
                        best = (err, dk, e["eid"])
            err, dk, eid = best
            g = h5[f"data/{dk}"]
            m = g["states/rigid_object/mustard_bottle/root_pose"][:]
            b = g["states/rigid_object/raisin_box/root_pose"][:]
            m0 = g["initial_state/rigid_object/mustard_bottle/root_pose"][0]
            b0 = g["initial_state/rigid_object/raisin_box/root_pose"][0]
            dy_end = float(m[-1, 1] - b[-1, 1])
            lift = float((m[:, 2] - m0[2]).max())
            box_moved = float(np.linalg.norm(b[-1, :3] - b0[:3]))
            tilt = upright_deg(m[-1, 3:7])
            ever_right = bool((m[:, 1] < b[:, 1]).any())
            if r["success"]:
                lab = "placed"
            elif dy_end < 0:
                lab = "right_unstable"
            elif lift >= 0.02 and dy_end > 0:
                lab = "converse"
            elif lift >= 0.02:
                lab = "dropped"
            else:
                lab = "no_lift"
            out.append({"pool": label, "slot": slot, "eid": eid, "demo": dk, "match_err_rad": round(err, 4),
                        "success": bool(r["success"]), "steps": r["steps"], "label": lab,
                        "mustard_minus_box_y_end_cm": round(100 * dy_end, 1), "mustard_lift_max_cm": round(100 * lift, 1),
                        "mustard_tilt_end_deg": round(tilt, 1), "box_moved_cm": round(100 * box_moved, 1),
                        "mustard_ever_right_of_box": ever_right,
                        "start_mustard_minus_box_y_cm": round(100 * float(m0[1] - b0[1]), 1)})
    labels = sorted({o["pool"] for o in out})
    for lb in labels:
        rows = [o for o in out if o["pool"] == lb]
        c = {}
        for o in rows:
            c[o["label"]] = c.get(o["label"], 0) + 1
        print(lb, len(rows), c, "box moved >2 cm:", sum(o["box_moved_cm"] > 2 for o in rows),
              "max match err %.3f" % max(o["match_err_rad"] for o in rows))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
