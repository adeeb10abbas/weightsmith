#!/usr/bin/env python3
"""Stage breakdown of a ctl_run.py output directory from the pool rows (RoboLab score and failure reason).

  python3 where_stages.py HELDOUT_DIR > stages.json

For the two-step cube-then-banana task, RoboLab's score counts completed subtasks: 0 = the cube never reached the bowl,
1 = cube placed but the banana subtask failed, success = both. "Failed at grab" means the grab condition was
unmet at the end, which includes a cube that was grasped and then dropped. Per arm: cube placed, success, banana-stage conversion,
and the failure reasons; per arm against the base arm: where the base failed on the fixed slots and where the arm
failed on the broken slots.
"""
import collections, glob, json, sys
from pathlib import Path


def main():
    d = Path(sys.argv[1])
    plan = json.loads((d / "plan.json").read_text())
    rows = {}
    for nm in plan["arms"]:
        r = {}
        for f in sorted(glob.glob(str(d / f"logs-{nm}-*" / "rows.json"))):
            for x in json.loads(Path(f).read_text())["rows"]:
                if x.get("ok"):
                    r[int(x["slot"])] = {"success": bool(x["success"]), "score": float(x.get("score") or 0),
                                         "steps": x.get("steps"), "reason": (x.get("reason") or "")[:120]}
        rows[nm] = r
    names = [nm for nm in plan["arms"] if rows[nm]]
    slots = sorted(set.intersection(*(set(rows[nm]) for nm in names)))

    def stage(x):
        if x["success"]:
            return "success"
        if x["score"] >= 1:
            return "banana subtask failed"
        # RoboLab resets the grab step when the held cube is dropped, so this includes premature releases
        return "cube subtask failed at grab" if "object_grabbed(object=rubiks_cube)" in x["reason"] else "cube subtask failed at place"
    out = {"n_slots": len(slots), "arms": {}, "vs_base": {}}
    for nm in names:
        st = collections.Counter(stage(rows[nm][s]) for s in slots)
        placed = sum(1 for s in slots if rows[nm][s]["success"] or rows[nm][s]["score"] >= 1)
        succ = sum(rows[nm][s]["success"] for s in slots)
        out["arms"][nm] = {"cube_placed": placed, "success": succ,
                           "banana_conversion": round(succ / placed, 3) if placed else None, "stages": dict(st)}
    ref = names[0]
    for nm in names[1:]:
        fixed = [s for s in slots if rows[nm][s]["success"] and not rows[ref][s]["success"]]
        broke = [s for s in slots if rows[ref][s]["success"] and not rows[nm][s]["success"]]
        out["vs_base"][nm] = {"fixed_base_stage": dict(collections.Counter(stage(rows[ref][s]) for s in fixed)),
                              "broke_arm_stage": dict(collections.Counter(stage(rows[nm][s]) for s in broke))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
