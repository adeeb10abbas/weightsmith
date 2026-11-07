#!/usr/bin/env python3
"""Behavioural signatures per arm from the recorded closed-loop episodes of a ctl_run/where_run output directory.

  python3 where_behaviour.py HELDOUT_DIR > behaviour.json

Per arm, over all slots: share of episodes whose first action chunk already commands the gripper closed (a chunk is
closed when most of its 15 steps command > 0.5, as offline), mean share of closed chunks before the first successful
grasp event, mean number of RoboLab WRONG_OBJECT_GRABBED_FAILURE and TARGET_OBJECT_DROPPED events per episode.
"""
import glob, json, sys
from pathlib import Path
import numpy as np


def main():
    d = Path(sys.argv[1])
    plan = json.loads((d / "plan.json").read_text())
    out = {}
    for nm in plan["arms"]:
        eps = []
        for f in sorted(glob.glob(str(d / f"logs-{nm}-*" / "rows.json"))):
            for r in json.loads(Path(f).read_text())["rows"]:
                if not r.get("ok") or not r.get("obs") or not Path(r["obs"]).exists():
                    continue
                o = np.load(r["obs"])
                closed = (o["actions"][:, :, 7] > 0.5).mean(1) >= 0.5
                ev = json.loads(Path(r["log"]).read_text()).get("events", []) if r.get("log") and Path(r["log"]).exists() else []
                first_grab = min([e["step"] for e in ev if e["name"] == "OBJECT_GRABBED_SUCCESS"] or [10 ** 9])
                pre = [c for c, st in zip(closed, o["step"]) if st < first_grab]
                eps.append({"first_closed": bool(closed[0]), "pre_grasp_closed": float(np.mean(pre)) if pre else None,
                            "wrong_grabs": sum(e["name"] == "WRONG_OBJECT_GRABBED_FAILURE" for e in ev),
                            "drops": sum(e["name"] == "TARGET_OBJECT_DROPPED" for e in ev)})
        if not eps:
            continue
        pg = [e["pre_grasp_closed"] for e in eps if e["pre_grasp_closed"] is not None]
        out[nm] = {"n": len(eps), "first_chunk_closed": round(float(np.mean([e["first_closed"] for e in eps])), 3),
                   "closed_share_before_first_grasp": round(float(np.mean(pg)), 3) if pg else None,
                   "wrong_object_grabs_per_episode": round(float(np.mean([e["wrong_grabs"] for e in eps])), 2),
                   "drops_per_episode": round(float(np.mean([e["drops"] for e in eps])), 2)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
