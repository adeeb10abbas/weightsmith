"""Track C (pi0.5 DROID) task set, projects, candidates and job queue for the cross-model comparison.

The task codes, classes, default instructions and slots are the shared design of Tracks A (Cosmos3 Edge), B (GR00T
N1.7) and C (pi0.5). Stdlib only; runs on the GPU host and on a laptop.

    python3 tasks.py setup [CAND ...]           # ali-ms: create the projects, copy the candidates, check fingerprints
    XPI05_PROJECTS=/workspace/cross/pi05/projects python3 tasks.py setup-pod /workspace/cross/pi05/cands
                                                # pod: projects whose cands/<name> link to the applied edits
    python3 tasks.py queue OUT.json             # write the job queue in priority order
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
PRACTICE = (30000, 32)  # slots 30000-30031 (groups 1875-1876): failure map and side-effect map, shared by A, B, C
HELDOUT = (96000, 64)   # slots 96000-96063 (groups 6000-6003): frozen edits only

PROJECTS = Path(os.environ.get("XPI05_PROJECTS") or Path.home() / "vibes/opus-weight-surgery/projects")
SOURCE = PROJECTS / "cube-then-banana" / "cands"
CANDS = {  # the paper's distilled edits, copied unchanged from projects/cube-then-banana
    "v2mi-f": "63db0c52a51306c9",     # VLM layers 0-16, prompt + state + image keys (the paper's edit)
    "abl-imgl8": "5bf8c4329c1e416a",  # image-cell keys only, VLM layers 8-16 (0.16% of parameters as factors)
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


def setup(cands=("v2mi-f",), link_from=None):
    """Create one project per task. ali-ms: copy each candidate from projects/cube-then-banana. Pod (link_from):
    cands/<name> is a symlink to the edit applied once with `weightsmith edit apply` under link_from/<name>."""
    for code, (task, instr, secs) in TASKS.items():
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
                          "~/cross/pi05/scripts/pool_pi05.py (weightsmith run_pool, one fresh Isaac process per "
                          "16-slot group), not by owsp; candidates are byte copies of projects/cube-then-banana's "
                          "(fingerprints checked); rendering_type null = RoboLab's default preset (no flag)"
                          + ("; task module from study/results/sgdw/tasks via WSP_TASK_DIRS" if code == "T8" else ""))
            }, indent=1) + "\n")
        for name in cands:
            dst = p / "cands" / name
            if not dst.exists() and not dst.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if link_from:
                    dst.symlink_to(Path(link_from) / name)
                else:
                    shutil.copytree(SOURCE / name, dst, symlinks=True)  # keeps the base -> shared base symlink
            fp = fingerprint(dst)
            print(f"{code} {p.name:18s} {name:10s} {fp} {'OK' if fp == CANDS[name] else 'MISMATCH'}", flush=True)
            if fp != CANDS[name]:
                raise SystemExit(f"fingerprint mismatch for {dst}")


def job(code, arm, s, prompt=None, rendering_type=None, video_mode="sensor"):
    """rendering_type None = RoboLab's default preset (the primary setting of all three tracks since 2026-11-08)."""
    task = TASKS[code][0]
    ws = str(project(code) / "cands" / arm) if arm in CANDS else None
    return {"tag": f"{code}-{arm}-s{s}", "code": code, "task": task, "slot_start": s, "n": 16, "arm": arm,
            "workspace": ws, "fp": CANDS.get(arm), "prompt": prompt, "rendering_type": rendering_type,
            "video_mode": video_mode}


def pair_jobs(code, arm):
    """Both 16-slot groups of the practice block; arm order alternates by half (as weightsmith/heldout.py)."""
    s0, s1 = PRACTICE[0], PRACTICE[0] + 16
    return [job(code, "base", s0), job(code, arm, s0), job(code, arm, s1), job(code, "base", s1)]


def queue():
    first, rest = ["T1", "T3", "T8", "R1"], ["T2", "T5", "T7", "T4", "T6"]
    jobs = []
    for code in first:                               # steps 1+2, highest value first
        jobs += pair_jobs(code, "v2mi-f")
        if code == "T1":                             # render-preset check: the paper's preset, same 16 slots
            jobs.append(job("T1", "base-perf", PRACTICE[0], rendering_type="performance"))
    for code in ("T1", "T8"):                        # teacher wordings on the same slots (recovery %)
        jobs += [job(code, "teacher", s, TEACHERS[code]) for s in (PRACTICE[0], PRACTICE[0] + 16)]
    for code in rest:                                # steps 1+2, the other tasks
        jobs += pair_jobs(code, "v2mi-f")
    for code in first + rest:                        # step 3: the 0.16% variant, same slots
        jobs += [job(code, "abl-imgl8", s) for s in (PRACTICE[0] + 16, PRACTICE[0])]
    seen = set()
    out = []
    for j in jobs:  # base arms appear once
        if j["tag"] not in seen:
            seen.add(j["tag"])
            out.append(j)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup(tuple(sys.argv[2:]) or ("v2mi-f",))
    elif sys.argv[1] == "setup-pod":
        setup(tuple(CANDS), link_from=sys.argv[2])
    elif sys.argv[1] == "queue":
        q = queue()
        Path(sys.argv[2]).write_text(json.dumps(q, indent=1) + "\n")
        print(len(q), "jobs")
