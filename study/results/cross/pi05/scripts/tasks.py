"""Track C (pi0.5 DROID) task set, projects, candidates and job queue for the cross-model comparison.

The task codes, classes, default instructions and slots are the shared design of Tracks A (Cosmos3 Edge), B (GR00T
N1.7) and C (pi0.5). Stdlib only; runs on the GPU host and on a laptop.

    python3 tasks.py setup [CAND ...]           # ali-ms: create the projects, copy the candidates, check fingerprints
    XPI05_PROJECTS=/workspace/cross/pi05/projects python3 tasks.py setup-pod /workspace/cross/pi05/cands BASE_DIR
                                                # pod: link every cands/<name>/base to the pod's base export, check
                                                # every candidate's fingerprint, link projects/<p>/cands/<name> to it
    python3 tasks.py queue OUT.json             # write the (pod) job queue in priority order
    python3 tasks.py queue-fresh OUT.json       # Track C' queue (ali-ms GPU0, default preset; pool_fresh.py)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

TASKS = {  # code: (RoboLab task class, default instruction, episode length in s at 15 Hz)
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
# Known-good ("teacher") wordings, given to the unedited weights through the prompt override.
TEACHERS = {
    "T1": "Put the cube in the bowl then put the banana in the bowl",  # the explicit wording v2mi-f was distilled from
    "T8": "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box.",  # TF
}
# Other wordings run for parity with Track A (not teachers of any edit here).
WORDINGS = {
    "T3": ("wspec", "Pick up the rubiks cube and place it on the table to the left side of the bowl"),
}
PRACTICE = (30000, 32)  # slots 30000-30031 (groups 1875-1876): failure map and side-effect map, shared by A, B, C
HELDOUT = (96000, 64)   # slots 96000-96063 (groups 6000-6003): frozen edits only

PROJECTS = Path(os.environ.get("XPI05_PROJECTS") or Path.home() / "vibes/opus-weight-surgery/projects")
CANDS = {  # byte copies of the published candidates on ali-ms; fingerprint = owsp.fingerprint
    "v2mi-f": "63db0c52a51306c9",     # the paper's T1 edit: VLM layers 0-16, prompt + state + image keys
    "abl-imgl8": "5bf8c4329c1e416a",  # T1, image-cell keys only, VLM layers 8-16 (0.16% of parameters as factors)
    "ctl-rand": "1efd3512a92642f7",   # T1 control: v2mi-f's layers, rank and singular values, random directions
    "s4r-grp3": "9e6f2378e28be6a3",   # T8 edit (sgdw): TF distilled into VLM layers 8-16, 3 decision-phase key groups
    "ctl-rand3": "fbc8349de4182099",  # T8 control: s4r-grp3's layers, rank and singular values, random directions
}
SOURCES = {  # ali-ms project that holds each candidate
    "v2mi-f": "cube-then-banana", "abl-imgl8": "cube-then-banana", "ctl-rand": "cube-then-banana",
    "s4r-grp3": "sgdw-s4r", "ctl-rand3": "sgdw-s4r",
}
EDITS_FOR = {  # which projects get which candidates
    "v2mi-f": list(TASKS), "abl-imgl8": list(TASKS), "ctl-rand": ["T1"], "s4r-grp3": ["T8"], "ctl-rand3": ["T8"],
}


def project(code: str) -> Path:
    return PROJECTS / f"cross-pi05-{code.lower()}"


def fingerprint(cand_dir: Path) -> str:
    """owsp.fingerprint: sha256 over the sorted edits/*.npy file names and bytes, first 16 hex digits."""
    h = hashlib.sha256()
    for f in sorted((cand_dir / "edits").glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def make_project(code):
    task, instr, secs = TASKS[code]
    p = project(code)
    p.mkdir(parents=True, exist_ok=True)
    cfg = p / "config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({
            "task": task, "instruction": instr, "practice_seed_base": PRACTICE[0],
            "limits": {"practice_episodes": 0, "acts_items": 0, "acts_items_per_call": 96},
            "rendering_type": None, "practice_slots": PRACTICE[1], "envs": 16,
            "prompt_overrides": True,
            "notes": ("Track C (pi0.5 control) of the cross-model WSP comparison: evaluated by "
                      "study/results/cross/pi05/scripts/pool_pi05.py (weightsmith run_pool, one fresh Isaac process "
                      "per 16-slot group), not by owsp; candidates are byte copies of the published ones "
                      "(fingerprints checked); rendering_type null = RoboLab's default preset (no flag)"
                      + ("; task module from study/results/sgdw/tasks via WSP_TASK_DIRS" if code == "T8" else ""))
        }, indent=1) + "\n")
    return p


def setup(cands=("v2mi-f",)):
    """ali-ms: create one project per task and copy each candidate from its source project."""
    for name in cands:
        for code in EDITS_FOR[name]:
            dst = make_project(code) / "cands" / name
            if not dst.exists() and not dst.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(PROJECTS / SOURCES[name] / "cands" / name, dst, symlinks=True)
            fp = fingerprint(dst)
            print(f"{code} {dst.parents[1].name:18s} {name:10s} {fp} {'OK' if fp == CANDS[name] else 'MISMATCH'}")
            if fp != CANDS[name]:
                raise SystemExit(f"fingerprint mismatch for {dst}")


def setup_pod(cand_root, base_dir):
    """Pod: cand_root/<name> holds the copied edits/, workspace.json, meta.json. Link its base/ to the pod's own base
    export (the server checks every edited tensor's base against the loaded checkpoint), check the fingerprint once,
    then link projects/cross-pi05-<code>/cands/<name> to it."""
    cand_root, base_dir = Path(cand_root), Path(base_dir)
    for name, want in CANDS.items():
        src = cand_root / name
        if not (src / "workspace.json").exists():
            print(f"{name:10s} not on this host yet: skipped")
            continue
        if list((src / "edits").glob("*.part")):
            print(f"{name:10s} transfer incomplete: skipped")
            continue
        b = src / "base"
        if b.is_symlink() or b.exists():
            if b.resolve() != base_dir.resolve():
                raise SystemExit(f"{b} -> {b.resolve()}, expected {base_dir}")
        else:
            b.symlink_to(base_dir)
        fp = fingerprint(src)
        print(f"{name:10s} {fp} {'OK' if fp == want else 'MISMATCH'}", flush=True)
        if fp != want:
            raise SystemExit(f"fingerprint mismatch for {src}")
        for code in EDITS_FOR[name]:
            dst = make_project(code) / "cands" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to(src)
    for code in TASKS:
        make_project(code)


def job(code, arm, s, prompt=None, rendering_type=None, video_mode="sensor"):
    """rendering_type None = RoboLab's default preset (the primary setting of all three tracks since 2026-11-08).
    arm: 'base', a candidate name, 'teacher' (TEACHERS[code] as prompt) or a wording code; '-perf' suffix =
    performance preset."""
    task = TASKS[code][0]
    cand = arm.removesuffix("-perf")
    ws = str(project(code) / "cands" / cand) if cand in CANDS else None
    return {"tag": f"{code}-{arm}-s{s}", "code": code, "task": task, "slot_start": s, "n": 16, "arm": arm,
            "workspace": ws, "fp": CANDS.get(cand), "prompt": prompt, "rendering_type": rendering_type,
            "video_mode": video_mode}


def groups(block):
    return [block[0] + 16 * i for i in range(block[1] // 16)]


def queue():
    """Priority order of the relaunched Track C (2026-11-08 01:00 ET):
    1. failure map T1-T8 + R1 (unedited, default wording), first group of every task, then the second;
    2. T1 held-out: unedited, v2mi-f, its random-direction control, the teacher wording;
    3. side-effect map: v2mi-f on every task, practice slots, paired with (1);
    4. T8: target-first wording (teacher) and the T8 edit s4r-grp3 on practice slots, then held-out with its control;
    5. secondary: performance preset (T1, T3), T3 'specific' wording, the 0.16% variant abl-imgl8."""
    first = ["T1", "T3", "T8", "R1"]
    rest = ["T2", "T5", "T7", "T4", "T6"]
    allc = first + rest
    P, H = groups(PRACTICE), groups(HELDOUT)
    jobs = []
    for s in P:                                                         # 1
        jobs += [job(c, "base", s) for c in allc]
    for i, s in enumerate(H):                                           # 2 (arm order rotates by group)
        arms = ["base", "v2mi-f", "ctl-rand", "teacher"]
        arms = arms[i % 4:] + arms[:i % 4]
        jobs += [job("T1", a, s, TEACHERS["T1"] if a == "teacher" else None) for a in arms]
    for s in P:                                                         # 3
        jobs += [job(c, "v2mi-f", s) for c in ["T3", "T8", "R1", "T1"] + rest]
    for s in P:                                                         # 4 practice
        jobs += [job("T8", "teacher", s, TEACHERS["T8"]), job("T8", "s4r-grp3", s)]
    for i, s in enumerate(H):                                           # 4 held-out
        arms = ["base", "s4r-grp3", "teacher"]
        arms = arms[i % 3:] + arms[:i % 3]
        jobs += [job("T8", a, s, TEACHERS["T8"] if a == "teacher" else None) for a in arms]
    jobs += [job("T8", "ctl-rand3", s) for s in H]
    for s in P:                                                         # 5
        jobs += [job("T1", "base-perf", s, rendering_type="performance"),
                 job("T1", "v2mi-f-perf", s, rendering_type="performance")]
    for s in P:
        jobs += [job("T3", w[0], s, w[1]) for c, w in WORDINGS.items()]
        jobs += [job("T3", "base-perf", s, rendering_type="performance"),
                 job("T3", "v2mi-f-perf", s, rendering_type="performance")]
    for s in P:
        jobs += [job(c, "abl-imgl8", s) for c in allc]
    seen, out = set(), []
    for j in jobs:
        if j["tag"] not in seen:
            seen.add(j["tag"])
            out.append(j)
    return out


def queue_fresh():
    """Track C' (2026-11-08 evening, ali-ms GPU0, default preset): fresh pi0.5 numbers on the slots of Tracks A and B.
    Every candidate job uses ONE workspace per candidate (cross-pi05-t1/cands/<name>: the same bytes as every other
    project's copy, fingerprint checked), so the worker can keep one policy server up across tasks.
    1. failure map: unedited, T1-T8 + R1, practice 30000-30015;
    2. T1 held-out 96000-96063: unedited, v2mi-f, its random-direction control ctl-rand (two groups per arm at a time,
       so a partial result is still paired);
    3. side effects of v2mi-f on held-out 96000-96031: T3, T5, then R1, T2, T4, then T6, T7, T8;
    4. T8 wording gap: target-first wording on practice 30000-30015 (reference-first = the failure-map run);
    5. T1 teacher wording on held-out (recovery %)."""
    def fj(code, arm, s, prompt=None):
        j = job(code, arm, s, prompt)
        if arm in CANDS:
            j["workspace"] = str(project("T1") / "cands" / arm)
        return j
    P0 = PRACTICE[0]
    H = groups(HELDOUT)
    jobs = [fj(c, "base", P0) for c in ["T1", "T3", "T5", "T8", "R1", "T2", "T4", "T6", "T7"]]       # 1
    for pair in (H[:2], H[2:]):                                                                      # 2
        for arm in ("base", "v2mi-f", "ctl-rand"):
            jobs += [fj("T1", arm, s) for s in pair]
    for codes in (["T3", "T5"], ["R1", "T2", "T4"], ["T6", "T7", "T8"]):                             # 3
        for arm in ("base", "v2mi-f"):
            jobs += [fj(c, arm, s) for c in codes for s in H[:2]]
    jobs.append(fj("T8", "teacher", P0, TEACHERS["T8"]))                                             # 4
    jobs += [fj("T1", "teacher", s, TEACHERS["T1"]) for s in H]                                      # 5
    return jobs


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(tuple(sys.argv[2:]) or ("v2mi-f",))
    elif sys.argv[1] == "setup-pod":
        setup_pod(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "queue-fresh":
        q = queue_fresh()
        Path(sys.argv[2]).write_text(json.dumps(q, indent=1) + "\n")
        print(len(q), "jobs")
    elif sys.argv[1] == "queue":
        q = queue()
        Path(sys.argv[2]).write_text(json.dumps(q, indent=1) + "\n")
        print(len(q), "jobs")
