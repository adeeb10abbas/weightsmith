"""Decision selection for closed-form distillation on GR00T (CPU; any python3 with numpy).

Rule (fixed before any edit is built, analogous to weightsmith's sgdw rule "the replan where the two wordings'
replays of the same input disagree most"):

  candidates: every `stride`-th replan k >= 1 of the practice episodes (slots 30000-30031) that FAILED under the
              task's own wording, replayed offline under the original and the explicit wording with the recorded
              images, state and noise seed (groot_acts.py --actions-only);
  divergence: distance between the two replays' end-effector targets at the last executed row (row 7, xyz of the
              decoded absolute eef_9d) plus 0.05 m for every executed row (0-7) whose binarised gripper differs;
  build:      the most divergent replan of each of the first `n_build` failed episodes in slot order;
  check:      the most divergent replan of every other failed episode (never used to build).

  python3 wsp_select.py candidates RUN_JSON --stride 2 --out cand.json
  python3 wsp_select.py pick RUN_JSON CAPTURE_DIR --n-build 6 --out items.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def failed_rows(run_json):
    d = json.loads(Path(run_json).read_text())
    return [r for r in d["rows"] if r.get("ok") and not r["success"] and r.get("obs") and Path(r["obs"]).exists()]


def cmd_candidates(a):
    items = []
    for r in failed_rows(a.run):
        n = len(np.load(r["obs"])["seed"])
        for k in range(1, n, a.stride):
            items.append({"obs": r["obs"], "k": k, "slot": r["slot"]})
    Path(a.out).write_text(json.dumps(items))
    print(len(items), "candidate decisions from", len({i["slot"] for i in items}), "failed episodes")


def divergence(o, x):
    eo, ex = o["act_eef_9d"], x["act_eef_9d"]
    d = float(np.linalg.norm(eo[7, :3] - ex[7, :3]))
    go, gx = o["act_gripper_position"][:8, 0] > 0.5, x["act_gripper_position"][:8, 0] > 0.5
    return d + 0.05 * float((go != gx).sum()), d, int((go != gx).sum())


def cmd_pick(a):
    cap = Path(a.capture)
    idx = json.loads((cap / "index.json").read_text())
    files = {(f["item"], f["wording"]): f["file"] for f in idx["files"]}
    rows = []
    for i, it in enumerate(idx["items"]):
        o, x = np.load(files[(i, "orig")]), np.load(files[(i, "expl")])
        s, d, g = divergence(o, x)
        rows.append({"i": i, "slot": it["slot"], "k": it["k"], "obs": it["obs"], "score": s, "eef_cm": 100 * d, "grip_rows": g,
                     "orig_grip_closed": int((o["act_gripper_position"][:8, 0] > 0.5).sum()),
                     "expl_grip_closed": int((x["act_gripper_position"][:8, 0] > 0.5).sum())})
    slots = sorted({r["slot"] for r in rows})
    best = {s: max((r for r in rows if r["slot"] == s), key=lambda r: r["score"]) for s in slots}
    build = [best[s] for s in slots[:a.n_build]]
    check = [best[s] for s in slots[a.n_build:]]
    out = {"rule": __doc__.split("Rule")[1].split("python3")[0].strip(), "n_candidates": len(rows),
           "build": build, "check": check,
           "summary": {"median_eef_cm_all": float(np.median([r["eef_cm"] for r in rows])),
                       "median_eef_cm_build": float(np.median([r["eef_cm"] for r in build])) if build else None}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).with_suffix(".all.json").write_text(json.dumps(rows))
    for r in build:
        print("build", r["slot"], "k", r["k"], f"eef {r['eef_cm']:.1f} cm grip rows {r['grip_rows']} "
              f"(orig closed {r['orig_grip_closed']}/8, expl {r['expl_grip_closed']}/8)")
    print(len(check), "check decisions; median divergence", out["summary"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("run")
    c.add_argument("--stride", type=int, default=2)
    c.add_argument("--out", required=True)
    p = sub.add_parser("pick")
    p.add_argument("run")
    p.add_argument("capture")
    p.add_argument("--n-build", type=int, default=6)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "pick": cmd_pick}[a.cmd](a)


if __name__ == "__main__":
    main()
