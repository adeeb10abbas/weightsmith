"""Offline check of a GR00T edit on recorded decisions (CPU; numpy).

Compares, per decision, three replays with images, state and noise seed fixed: the unedited policy under the original
wording (o), the unedited policy under the explicit wording (x, the teacher) and the edited policy under the original
wording (e). Reports the end-effector target at the last executed row (row 7) and the binarised gripper of rows 0-7:
the share of decisions where e is closer to x than o is, the mean distances, and how often e takes x's gripper call.

  python3 wsp_check.py --ref CAPTURE_DIR(orig+expl) --edited CAPTURE_DIR(edited, wording orig) --items items.json [--out f]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def files(cap):
    idx = json.loads((Path(cap) / "index.json").read_text())
    return idx, {(f["item"], f["wording"]): f["file"] for f in idx["files"]}


def key(it):
    return (str(it["obs"]), int(it["k"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--edited-wording", default="orig")
    ap.add_argument("--groups", required=True, help="JSON {group: [{obs,k},...]} (e.g. build/check)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ri, rf = files(a.ref)
    ei, ef = files(a.edited)
    rmap = {key(it): i for i, it in enumerate(ri["items"])}
    emap = {key(it): i for i, it in enumerate(ei["items"])}
    groups = json.loads(Path(a.groups).read_text())
    rep = {}
    for g, its in groups.items():
        rows = []
        for it in its:
            o = np.load(rf[(rmap[key(it)], "orig")])
            x = np.load(rf[(rmap[key(it)], "expl")])
            e = np.load(ef[(emap[key(it)], a.edited_wording)])
            p = {n: v["act_eef_9d"][7, :3] for n, v in (("o", o), ("x", x), ("e", e))}
            gr = {n: (v["act_gripper_position"][:8, 0] > 0.5) for n, v in (("o", o), ("x", x), ("e", e))}
            rows.append({"slot": it.get("slot"), "k": it["k"],
                         "d_ox_cm": 100 * float(np.linalg.norm(p["o"] - p["x"])),
                         "d_ex_cm": 100 * float(np.linalg.norm(p["e"] - p["x"])),
                         "d_eo_cm": 100 * float(np.linalg.norm(p["e"] - p["o"])),
                         "grip_o": int(gr["o"].sum()), "grip_x": int(gr["x"].sum()), "grip_e": int(gr["e"].sum()),
                         "grip_e_eq_x": int((gr["e"] == gr["x"]).sum())})
        if not rows:
            continue
        rep[g] = {"n": len(rows),
                  "closer_to_teacher": int(sum(r["d_ex_cm"] < r["d_ox_cm"] for r in rows)),
                  "mean_d_unedited_teacher_cm": float(np.mean([r["d_ox_cm"] for r in rows])),
                  "mean_d_edited_teacher_cm": float(np.mean([r["d_ex_cm"] for r in rows])),
                  "mean_d_edited_unedited_cm": float(np.mean([r["d_eo_cm"] for r in rows])),
                  "gripper_rows_matching_teacher": f"{sum(r['grip_e_eq_x'] for r in rows)}/{8 * len(rows)}",
                  "gripper_rows_unedited_matching_teacher": f"{sum(int((r['grip_o'] == r['grip_x'])) for r in rows)} (decisions with equal closed-row counts)",
                  "rows": rows}
        print(g, {k: v for k, v in rep[g].items() if k != "rows"})
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
