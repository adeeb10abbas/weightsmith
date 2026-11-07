"""Recipe 9: offline validation: does the edit flip the target decisions and leave the others alone?

When: after building a candidate and before spending closed-loop episodes on it. Five of the seven blind editors
rejected at least one candidate here and kept their episode budget, and the paper's operator stopped seven edits
here before the distilled one. See docs/COOKBOOK.md, section 5.

Run (read-only):  bin/owsp py base examples/recipes/09_offline_check.py

Input: a base capture and one capture per candidate of the SAME decisions (same episodes and replans, same order):
the candidate's replay uses the recorded images, robot state and noise seed, so any difference comes from the
weights. ``specs/candidate_check.json`` is a cheap spec for this (actions only; set ``"cand"`` to your candidate).
Group the decisions into targets (should change) and preservation groups (must not change). For each candidate the
recipe counts gripper decisions that flip, measures how far the end of the chunk moves (forward kinematics), and,
with a known-good-wording replay (``teacher_capture``), how close the edit lands to it.

Check it worked: a candidate passes when most targets flip and no preservation decision flips. Read the motion
column too: preservation chunks may move when the known-good wording moves them as well (the cube carries here
move about 7 cm, toward the explicit wording's end points); set ``max_shift_cm`` to make motion a criterion.
On cube-then-banana the distilled edit (chk-v2mi-f) flips 21/21 premature releases and no other gripper decision,
the 0.16% variant flips 19/21, and the rank-1 "target axis" steering edit (ctb-ph-st11a5) flips none: the paper's
Fig. 2b, recomputed. Passing offline is necessary, not sufficient: the flat-box edit passed this
check and still scored 0/32 in closed loop.
"""
import json

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "base_capture": "ctb-ph",
    "cand_captures": ["chk-v2mi-f", "chk-abl-imgl8", "chk-v2p-f", "chk-v2m-2", "chk-v2m-f", "chk-v2m-h",
                      "ctb-ph-st11a5", "ctb-ph-st1016a4"],
    "groups": {"premature release": [0, 21], "cube carry": [21, 30], "banana approach": [30, 37],
               "banana grasp": [37, 42], "banana carry": [42, 49]},
    "targets": ["premature release"],
    "teacher_capture": "a1-v2",    # optional: the known-good wording replayed on (some of) the same decisions
    "min_target_flips": 0.8,       # pass: at least this share of target decisions flips, no preservation flip ...
    "max_shift_cm": None,          # ... and, if set, no preservation group's chunk end moves more than this (cm)
}


def panda_fk(q):
    """Fingertip position (m, robot base frame) of a Franka Panda: modified DH, flange + 0.257 m."""
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(q[i]), np.sin(q[i])
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


def chunk_closed(actions, threshold=0.5):
    """True where a [.., 15, 8] chunk commands the gripper closed (most of its 15 steps above the threshold)."""
    return (np.asarray(actions)[..., 7] > threshold).mean(-1) >= 0.5


def replay(aid):
    """(candidate name, [(episode, k)], actions [N, 15, 8]) of a capture."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    acts_ = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")["actions"].astype(np.float64)
    return idx["cand"], [(it["episode"], int(it["k"])) for it in idx["items"]], acts_


def main(cfg=SETTINGS):
    _, keys, B = replay(cfg["base_capture"])
    groups = {g: list(range(lo, hi)) for g, (lo, hi) in cfg["groups"].items()}
    end_b = np.array([panda_fk(a[-1, :7]) for a in B])
    teacher = {}
    if cfg.get("teacher_capture"):
        _, tkeys, T = replay(cfg["teacher_capture"])
        teacher = {k: panda_fk(a[-1, :7]) for k, a in zip(tkeys, T, strict=True)}
    results = {}
    for aid in cfg["cand_captures"]:
        cand, ckeys, C = replay(aid)
        if ckeys != keys:
            raise SystemExit(f"{aid} replays different decisions than {cfg['base_capture']}; capture the same items")
        end_c = np.array([panda_fk(a[-1, :7]) for a in C])
        cb, cc = chunk_closed(B), chunk_closed(C)
        print(f"\n{aid} (candidate {cand!r}) vs {cfg['base_capture']}")
        print(f"  {'group':<20} {'n':>3}  closed base -> edit  flips  chunk end moved"
              + ("  to teacher" if teacher else ""))
        ok, notes = True, []
        for g, ix in groups.items():
            flips = int((cb[ix] != cc[ix]).sum())
            shift = float(np.linalg.norm(end_c[ix] - end_b[ix], axis=1).mean() * 100)
            line = (f"  {g:<20} {len(ix):>3}  {int(cb[ix].sum()):>6} -> {int(cc[ix].sum()):<4}   {flips:>5}  "
                    f"{shift:9.1f} cm")
            tix = [i for i in ix if keys[i] in teacher]
            if tix:
                dt_c = np.mean([np.linalg.norm(end_c[i] - teacher[keys[i]]) for i in tix]) * 100
                dt_b = np.mean([np.linalg.norm(end_b[i] - teacher[keys[i]]) for i in tix]) * 100
                line += f"  {dt_c:5.1f} cm (base {dt_b:.1f})"
            print(line)
            if g in cfg["targets"]:
                if flips < cfg["min_target_flips"] * len(ix):
                    ok = False
                    notes.append(f"only {flips}/{len(ix)} {g} flipped")
            elif flips or (cfg["max_shift_cm"] is not None and shift > cfg["max_shift_cm"]):
                ok = False
                notes.append(f"{g}: {flips} flips, chunk end moved {shift:.1f} cm")
        print(f"  offline: {'PASS' if ok else 'FAIL'}" + (f" ({'; '.join(notes)})" if notes else ""))
        results[aid] = ok
    return results


if __name__ == "__main__":
    main()
