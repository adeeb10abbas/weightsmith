# ruff: noqa: F821  (episodes: helper of the owsp py sandbox prelude)
# Behaviour behind the success rate: how long the robot carries the cube before dropping it, and how often an
# object reaches the bowl, per candidate, from the recorded practice episodes' events.
#   bin/owsp py base examples/cube_then_banana/carries.py
import json

CANDS = ["base", "ctb-imgl8", "ctb-first"]

for cand in CANDS:
    rows = episodes(cand=cand)
    if not rows:
        continue
    in_bowl, carries = 0, []
    for r in rows:
        with open(f"/work/data/episodes/{r['eid']}/summary.json") as f:
            events = [(e["step"], e["name"]) for e in json.load(f)["events"]]
        in_bowl += any(name == "OBJECT_IN_CONTAINER_SUCCESS" for _, name in events)
        grabbed = None
        for step, name in events:
            if name == "OBJECT_GRABBED_SUCCESS" and grabbed is None:
                grabbed = step
            elif name == "TARGET_OBJECT_DROPPED" and grabbed is not None:
                carries.append(step - grabbed)
                grabbed = None
    carries.sort()
    median = carries[len(carries) // 2] if carries else 0
    print(f"{cand:<10} {sum(r['success'] for r in rows):2d}/{len(rows)} succeeded   object in the bowl in "
          f"{in_bowl:2d}   {len(carries):3d} carries, median {median:2d} steps before the drop")
