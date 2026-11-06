#!/usr/bin/env python3
"""Write site/data/hero_90002.json, the data behind the scrubbable timeline under the opening video.

Source: paper/figures/src/teaser_90002.json (held-out start 90002 of the cube-then-banana run, built from the
recorded episodes by paper/scripts/build_teaser_data.py): per-step commanded gripper values and logged events for the
unedited and the edited policy. The opening clip (site/assets/video/cube_then_banana.mp4) shows the 15 Hz simulation
at 2x speed, rendered at 30 fps from step 0, so one video frame is one control step.

The script keeps the closed-gripper spans (command above the 0.5 threshold), the events, and short phase labels for the
read-out, and checks every step it labels against the logged events.

Usage: python3 site/scripts/make_hero_timeline.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper/figures/src/teaser_90002.json"
OUT = ROOT / "site/data/hero_90002.json"
CLIP_FRAMES = 288  # frames in cube_then_banana.mp4 (9.6 s at 30 fps)


def spans(steps, values, thr=0.5):
    """Contiguous [start, end) step spans where the commanded gripper is closed (> thr), as in the paper's Fig. 1."""
    out, start = [], None
    for s, v in zip(steps, values):
        if v > thr and start is None:
            start = s
        if v <= thr and start is not None:
            out.append([start, s])
            start = None
    if start is not None:
        out.append([start, steps[-1] + 1])
    return out


def first(arm, name, obj=None):
    hits = [e["step"] for e in arm["events"] if e["name"] == name and (obj is None or obj in e["info"])]
    return min(hits)


meta = json.loads(SRC.read_text())
U, E = meta["arms"]["unedited"], meta["arms"]["edited"]
assert meta["slot"] == 90002 and not U["success"] and E["success"]

u_grasp, u_drop = first(U, "OBJECT_GRABBED_SUCCESS", "rubiks"), first(U, "TARGET_OBJECT_DROPPED")
u_banana, u_banana_off = first(U, "WRONG_OBJECT_GRABBED_FAILURE", "banana"), first(U, "OK", "banana")
e_grasp, e_banana = first(E, "OBJECT_GRABBED_SUCCESS", "rubiks"), first(E, "OBJECT_GRABBED_SUCCESS", "banana")
e_bowl = sorted(e["step"] for e in E["events"] if e["name"] == "OBJECT_IN_CONTAINER_SUCCESS")
assert (u_grasp, u_drop, u_banana, u_banana_off) == (73, 76, 148, 196)
assert (e_grasp, e_banana, e_bowl, E["final_step"]) == (78, 198, [119, 255], 257)
u_closed = [s for s in spans(U["cmd_steps"], U["gripper_cmd"]) if s[0] < CLIP_FRAMES]
e_closed = [s for s in spans(E["cmd_steps"], E["gripper_cmd"]) if s[0] < CLIP_FRAMES]
assert u_closed[0] == [70, 75] and e_closed == [[75, 118], [191, 254]]

data = {
    "source": "paper/figures/src/teaser_90002.json",
    "slot": meta["slot"],
    "instruction": meta["instruction"],
    "fps": 30,
    "steps_per_frame": 1,
    "control_hz": 15,
    "replan_every": 15,
    "clip_steps": CLIP_FRAMES,
    "arms": {
        "unedited": {
            "success": U["success"],
            "last_step": U["cmd_steps"][-1],
            "closed": u_closed,
            "marks": [{"step": u_drop, "kind": "release", "label": f"cube released, {u_drop - u_grasp} steps after the grasp"}],
            "phases": [[0, "reaching for the cube"], [u_closed[0][0], "closing on the cube"], [u_grasp, "holding the cube"],
                       [u_drop, "cube dropped"], [u_banana, "holding the banana"], [u_banana_off, "banana released"]],
        },
        "edited": {
            "success": E["success"],
            "last_step": E["cmd_steps"][-1],
            "final_step": E["final_step"],
            "closed": e_closed,
            "marks": [{"step": e_bowl[0], "kind": "bowl", "label": "cube in the bowl"},
                      {"step": e_bowl[1], "kind": "bowl", "label": f"success at {E['final_step']}"}],
            "phases": [[0, "reaching for the cube"], [e_closed[0][0], "closing on the cube"], [e_grasp, "holding the cube"],
                       [e_bowl[0], "cube in the bowl"], [e_closed[1][0], "closing on the banana"],
                       [e_banana, "holding the banana"], [e_bowl[1], "banana in the bowl"],
                       [E["final_step"], "success"]],
        },
    },
}
OUT.write_text(json.dumps(data, indent=1) + "\n")
print("wrote", OUT.relative_to(ROOT))
