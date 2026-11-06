#!/usr/bin/env python3
"""Summarize the RoboLab event logs of the held-out cube-then-banana run of the distilled edit (v2mi-f).

Writes paper/figures/src/heldout_v2mif_events.json: for every held-out slot 90000-90127, the success of the
unedited and edited runs and, from each run's event log, the steps at which the cube was grasped, the steps at which
the cube was dropped outside the bowl, and the step of the first banana grasp. make_figures.py reads this file for
the counts quoted in the paper and to check the rule that picks the Fig. 1 slot.

A cube drop outside the bowl is a TARGET_OBJECT_DROPPED event together with an OBJECT_GRABBED_FAILURE naming
rubiks_cube at the same step, with no OBJECT_IN_CONTAINER_SUCCESS or "Completed subtask" event within two steps
(which would be a placement in the bowl).

Inputs, copied from the GPU host:
  ROWS_DIR/{base,cand}-{90000,90064}.json   rows.json of heldout/v2mi-f-RubiksCubeThenBananaTask-90000-128/
                                            logs-{base,cand}-{90000,90064}/ (one row per slot, with its log path)
  LOGS_TGZ                                  the 256 event logs those rows point to, e.g. made on the host with
                                            tar -C / -czf logs.tgz -T <list of the rows' "log" paths without the leading />

Usage: python3 paper/scripts/summarize_heldout_events.py ROWS_DIR LOGS_TGZ
"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures/src/heldout_v2mif_events.json"


def summarize(events):
    placed = [e["step"] for e in events
              if e["name"] == "OBJECT_IN_CONTAINER_SUCCESS" or "Completed subtask" in e["info"]]
    cube_fail = {e["step"] for e in events if e["name"] == "OBJECT_GRABBED_FAILURE" and "rubiks_cube" in e["info"]}
    grasps = [e["step"] for e in events if e["name"] == "OBJECT_GRABBED_SUCCESS" and "object=rubiks_cube" in e["info"]]
    drops = [e["step"] for e in events if e["name"] == "TARGET_OBJECT_DROPPED" and e["step"] in cube_fail
             and not any(abs(e["step"] - p) <= 2 for p in placed)]
    banana = [e["step"] for e in events if "banana" in e["info"] and e["name"] in
              ("OBJECT_GRABBED_SUCCESS", "WRONG_OBJECT_GRABBED_FAILURE") and "grab" in e["info"].lower()]
    return dict(cube_grasps=grasps, cube_drops_outside_bowl=drops, first_banana_grasp=min(banana) if banana else None)


def main(rows_dir, tgz):
    logs = {}
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".json"):
                logs["/" + m.name] = json.loads(tf.extractfile(m).read())
    slots = {}
    for f in sorted(Path(rows_dir).glob("*.json")):
        arm = {"base": "unedited", "cand": "edited"}[f.stem.split("-")[0]]
        for r in json.loads(f.read_text())["rows"]:
            log = logs[r["log"]]
            assert log["success"] == r["success"], r["slot"]
            slots.setdefault(str(r["slot"]), {})[arm] = dict(success=r["success"], steps=r["steps"],
                                                              **summarize(log["events"]))
    assert len(slots) == 128 and all(len(v) == 2 for v in slots.values())
    meta = dict(run="v2mi-f held-out, RubiksCubeThenBananaTask, slots 90000-90127",
                rule="cube drop outside the bowl: TARGET_OBJECT_DROPPED with OBJECT_GRABBED_FAILURE(rubiks_cube) "
                     "at the same step and no container success within 2 steps",
                logs_sha256=hashlib.sha256(Path(tgz).read_bytes()).hexdigest(),
                slots=dict(sorted(slots.items(), key=lambda kv: int(kv[0]))))
    OUT.write_text(json.dumps(meta, indent=0))
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
