"""The shared cross-model task set (Track A: Cosmos3 Edge; Track B: GR00T) and job-file helpers."""
from __future__ import annotations

import json
import sys

TASKS = {
    "T1": ("RubiksCubeThenBananaTask", "Put the cube then the banana in the bowl", 60),
    "T2": ("PickGlassesTask", "Pick up the eye glasses", 30),
    "T3": ("RubiksCubeLeftOfBowlTask", "Put the rubiks cube to the left of the bowl", 30),
    "T4": ("StackYellowOnRedTask", "Stack the yellow block on the red block", 60),
    "T5": ("MustardInLeftBinTask", "Put the mustard in the left bin", 30),
    "T6": ("Stack3RubiksCubeTask", "Stack the rubiks cubes in a tower", 60),
    "T7": ("PickUpGreenObjectTask", "Pick up the green vegetable block", 30),
    "T8": ("MustardRightOfRaisinRFTask",
           "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle.", 40),
    "R1": ("RubiksCubeTask", "Put the cube in the bowl", 40),
}
PRACTICE = (30000, 32)   # slots 30000-30031 (groups 1875-1876), shared by Tracks A, B, C: design, failure map, wording screens, practice
WORDINGS = {  # metric 2: (arm suffix, alternative wording, source)
    "T1": ("wexp", "Put the cube in the bowl then put the banana in the bowl",
           "pi0.5's known-good explicit wording (paper, Section 2)"),
    "T3": ("wspec", "Pick up the rubiks cube and place it on the table to the left side of the bowl",
           "the task's own 'specific' instruction in RoboLab"),
    "T8": ("wtf", "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box.",
           "target-first wording of the same goal (Same Goal, Different Words; known-good for pi0.5)"),
}
HELDOUT = (96000, 64)    # slots 96000-96063 (groups 6000-6003), shared: frozen edits only, never used for any choice


def job(code, tag, slot_start, n, prompt=None, edit=None):
    task = TASKS[code][0]
    j = {"tag": tag, "task": task, "slot_start": slot_start, "n": n}
    if prompt:
        j["prompt"] = prompt
    if edit:
        j["edit"] = edit
    return j


if __name__ == "__main__":
    # python3 tasks.py failure-map OUT.json: every task, unedited, default wording, one 16-slot group per job,
    # first group 1875 (slots 30000-30015) for all tasks, then group 1876 (a 16-start map is ready halfway).
    if sys.argv[1] == "failure-map":
        order = ["T3", "T8", "T1", "R1", "T2", "T5", "T7", "T4", "T6"]
        jobs = [job(c, f"{c}-base-s{s}", s, 16) for s in (PRACTICE[0], PRACTICE[0] + 16) for c in order]
        json.dump(jobs, open(sys.argv[2], "w"), indent=1)
        print(len(jobs), "jobs")
    # python3 tasks.py queue1 OUT.json: map group 1875, wording arms on group 1875, map group 1876, wording group 1876
    if sys.argv[1] == "queue1":
        order = ["T3", "T8", "T1", "R1", "T2", "T5", "T7", "T4", "T6"]
        jobs = []
        for s in (PRACTICE[0], PRACTICE[0] + 16):
            jobs += [job(c, f"{c}-base-s{s}", s, 16) for c in order]
            jobs += [job(c, f"{c}-{w[0]}-s{s}", s, 16, prompt=w[1]) for c, w in WORDINGS.items()]
        json.dump(jobs, open(sys.argv[2], "w"), indent=1)
        print(len(jobs), "jobs")
