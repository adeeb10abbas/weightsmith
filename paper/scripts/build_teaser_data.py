#!/usr/bin/env python3
"""Build the Fig. 1 source files from the recorded held-out episodes of one slot.

Fig. 1 shows held-out slot 90002 of the distilled-edit run (v2mi-f, cube then banana). The slot was chosen by a
fixed rule: the lowest-numbered held-out slot that the edit fixed and whose unedited run grasped the cube first and
then released it outside the bowl, according to the RoboLab event logs. (Slot 90000, used for the project-site clip,
does not qualify: its unedited run grabbed the banana first.)

Inputs, copied from the GPU host (paths from the run's rows.json; slot 90002 is env 2 of lane 0):
  heldout/v2mi-f-RubiksCubeThenBananaTask-90000-128/logs-{base,cand}-90000/obs-l0a0/obs_r0_e2.npz  -> {base,cand}/obs.npz
  robolab_checkout/output/pool-ho-{base-1105-232403,cand-1105-234621}-l0a0/RubiksCubeThenBananaTask/log_0_env2.json
                                                                                                  -> {base,cand}/log.json
  e.g.  scp ali-ms:<projects>/cube-then-banana/heldout/.../logs-base-90000/obs-l0a0/obs_r0_e2.npz SRC/base/obs.npz

Each obs file holds the policy's inputs and outputs at every replan (every 15 control steps at 15 Hz): 'step',
'wrist_img' and 'base_img' (224x224, letterboxed), and 'actions' [replans, 15, 8], whose channel 7 is the commanded
gripper (closed above 0.5). Each log holds RoboLab's events with their control step.

Outputs (committed, read by make_figures.py):
  paper/figures/src/teaser_90002.json        gripper command per control step, events, replan steps, provenance
  paper/figures/src/teaser_90002_frames.png  wrist-camera inputs at the storyboard steps (rows: unedited, edited)

Usage: python3 paper/scripts/build_teaser_data.py SRC_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SLOT = 90002
STEPS = [75, 90, 120, 255]          # storyboard moments (replan steps)
ROWS = (49, 175)                    # non-letterbox rows of the 224x224 camera inputs
OUT = Path(__file__).resolve().parents[1] / "figures/src"
KEEP = ("OBJECT_GRABBED_SUCCESS", "TARGET_OBJECT_DROPPED", "OBJECT_IN_CONTAINER_SUCCESS",
        "WRONG_OBJECT_GRABBED_FAILURE", "OBJECT_GRABBED_FAILURE", "OK")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arm(src):
    z = np.load(src / "obs.npz", allow_pickle=False)
    log = json.loads((src / "log.json").read_text())
    steps = z["step"].astype(int)
    acts = z["actions"]
    assert acts.shape[1:] == (15, 8) and np.all(np.diff(steps) == 15), (acts.shape, steps[:5])
    cmd = acts[:, :, 7].reshape(-1)                       # one gripper command per control step
    cmd_steps = (steps[:, None] + np.arange(15)[None, :]).reshape(-1)
    frames = []
    for s in STEPS:
        i = int(np.where(steps == s)[0][0])
        frames.append(z["wrist_img"][i, ROWS[0]:ROWS[1]])
    events = [dict(step=e["step"], name=e["name"], info=e["info"]) for e in log["events"] if e["name"] in KEEP]
    return dict(success=bool(log["success"]), final_step=int(log["final_step"]), dt=float(log["dt"]),
                replan_steps=steps.tolist(), cmd_steps=cmd_steps.tolist(),
                gripper_cmd=[round(float(v), 4) for v in cmd], events=events,
                sources={"obs.npz": sha(src / "obs.npz"), "log.json": sha(src / "log.json")}), frames


def main(src):
    src = Path(src)
    data, frames = {}, {}
    for name, sub in (("unedited", "base"), ("edited", "cand")):
        data[name], frames[name] = arm(src / sub)
    gap = 4
    h, w = frames["unedited"][0].shape[:2]
    sheet = np.full((2 * h + gap, len(STEPS) * w + (len(STEPS) - 1) * gap, 3), 255, np.uint8)
    for r, name in enumerate(("unedited", "edited")):
        for c, f in enumerate(frames[name]):
            sheet[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = f
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(OUT / "teaser_90002_frames.png", optimize=True)
    meta = dict(slot=SLOT, task="RubiksCubeThenBananaTask", run="v2mi-f held-out, slots 90000-90127",
                instruction="Put the cube then the banana in the bowl", storyboard_steps=STEPS,
                frame_rows=list(ROWS), frame_size=[h, w], frame_gap=gap, arms=data)
    (OUT / "teaser_90002.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT / "teaser_90002.json", OUT / "teaser_90002_frames.png", sheet.shape)


if __name__ == "__main__":
    main(sys.argv[1])
