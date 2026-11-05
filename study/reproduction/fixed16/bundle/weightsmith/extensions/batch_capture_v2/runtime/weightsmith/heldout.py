"""Operator-only held-out evaluation: base vs frozen candidate on unseen noise slots.

Not reachable from owsp. Usage (on the GPU host):
  WSP_ROOT=<project root> python3 weightsmith/heldout.py --cand NAME --slot-start 90000 --n 192
Freezes a copy of the candidate edits first, verifies its fingerprint, then runs
base and candidate on the identical slots (order alternates by half), and reports
paired statistics (exact McNemar) plus Wilson intervals.
"""
from __future__ import annotations

import argparse, json, math, re, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402
from weightsmith.pool import run_pool  # noqa: E402
from weightsmith.integrity import workspace_fingerprint  # noqa: E402

R = C.PROJECT_ROOT  # env WSP_ROOT
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,100}$")


def evaluation_schedule(slot_start, n, envs):
    """Validate the entire paired schedule before writing or starting anything."""
    if not 1 <= envs <= 64 or n <= 0 or n % (2 * envs):
        raise ValueError("envs must be in 1..64 and n must be a positive multiple of 2*envs for balanced halves")
    if slot_start < 80000 or slot_start % envs:
        raise ValueError("held-out slot-start must be >= 80000 and aligned to envs")
    half = (n // 2) // envs * envs
    first = list(range(slot_start, slot_start + half))
    second = list(range(slot_start + half, slot_start + n))
    return [("base", first), ("cand", first), ("cand", second), ("base", second)]


def verify_report(report, workspace, expected):
    if isinstance(report, str):
        report = json.loads(report)
    requested_path = str(Path(workspace).resolve()) if workspace else None
    if report.get("workspace") != requested_path or report.get("fingerprint") != expected:
        raise RuntimeError("Server did not load the exact requested frozen policy")
    if workspace_fingerprint(workspace) != expected:
        raise RuntimeError("Frozen policy changed during held-out evaluation")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    CFG = json.loads((R / "config.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--slot-start", type=int, default=90000)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--task", default=CFG["task"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--envs", type=int, default=CFG["envs"])
    ap.add_argument("--instruction", default=None, help="Required if selecting a different task")
    a = ap.parse_args()
    try:
        sched = evaluation_schedule(a.slot_start, a.n, a.envs)
    except ValueError as exc:
        ap.error(str(exc))
    if not NAME_RE.fullmatch(a.cand):
        ap.error("invalid candidate name")
    if a.task != CFG["task"] and a.instruction is None:
        ap.error("--instruction is required when changing --task")
    instruction = a.instruction if a.instruction is not None else CFG["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        ap.error("instruction must be nonempty")
    if instruction != CFG["instruction"] and CFG.get("prompt_overrides") is False:
        ap.error("this project forbids prompt overrides")
    src = None if a.cand == "base" else R / "cands" / a.cand
    if src is not None and (not (src / "workspace.json").is_file() or not (src / "base").is_dir()
                            or not (src / "edits").is_dir()):
        ap.error(f"incomplete candidate workspace: {src}")
    expected_fp = workspace_fingerprint(src)
    tag = a.tag or f"{a.cand}-{a.task}-{a.slot_start}-{a.n}"
    if not NAME_RE.fullmatch(tag):
        ap.error("invalid output tag")
    out = R / "heldout" / tag
    if out.exists():
        sys.exit(f"{out} exists; held-out runs are never repeated or overwritten")
    out.mkdir(parents=True)
    frozen = None
    if src is not None:
        frozen = out / "frozen"
        frozen.mkdir()
        shutil.copy2(src / "workspace.json", frozen / "workspace.json")
        (frozen / "base").symlink_to((src / "base").resolve())
        shutil.copytree(src / "edits", frozen / "edits")
        (frozen / "scratch").mkdir()
    fp = workspace_fingerprint(frozen)
    if fp != expected_fp or workspace_fingerprint(src) != expected_fp:
        raise RuntimeError("Candidate changed while creating the frozen copy")
    plan = {"cand": a.cand, "fingerprint": fp, "task": a.task, "slots": [a.slot_start, a.slot_start + a.n - 1],
            "instruction": instruction, "envs": a.envs, "candidate_kind": "base_control" if src is None else "edited",
            "order": "first half: base then cand; second half: cand then base", "started": time.ctime()}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    res = {"base": {}, "cand": {}}
    for arm, slots in sched:
        workspace = frozen if arm == "cand" else None
        expected = fp if arm == "cand" else "base"
        if workspace_fingerprint(workspace) != expected:
            raise RuntimeError("Frozen policy changed before held-out evaluation")
        rows, rep = run_pool(workspace=str(workspace) if workspace else None, task=a.task, tag=f"ho-{arm}",
                             slots=slots, envs=a.envs, rendering_type=CFG["rendering_type"],
                             video_mode="sensor", log_root=out / f"logs-{arm}-{slots[0]}", prompt=instruction)
        verify_report(rep, workspace, expected)
        for r in rows:
            res[arm][r["slot"]] = r
        (out / "partial.json").write_text(json.dumps(res, indent=1))
    if workspace_fingerprint(frozen) != fp:
        raise RuntimeError("Frozen policy changed before final held-out attribution")
    slots = [s for s in range(a.slot_start, a.slot_start + a.n) if res["base"].get(s, {}).get("ok") and res["cand"].get(s, {}).get("ok")]
    b = [bool(res["base"][s]["success"]) for s in slots]
    c = [bool(res["cand"][s]["success"]) for s in slots]
    kb, kc, n = sum(b), sum(c), len(slots)
    fixed = sum((not x) and y for x, y in zip(b, c))
    broke = sum(x and (not y) for x, y in zip(b, c))
    summary = {**plan, "finished": time.ctime(), "paired_n": n,
               "status": "complete" if n == a.n else "incomplete",
               "base_success": kb, "cand_success": kc,
               "base_rate": round(kb / n, 4) if n else None, "cand_rate": round(kc / n, 4) if n else None,
               "base_wilson95": wilson(kb, n), "cand_wilson95": wilson(kc, n),
               "fixed": fixed, "broke": broke, "mcnemar_exact_p": mcnemar_exact(fixed, broke),
               "crashed_slots": a.n - n,
               "mean_steps_success": {"base": round(sum(res["base"][s]["steps"] for s, x in zip(slots, b) if x) / max(1, kb), 1),
                                      "cand": round(sum(res["cand"][s]["steps"] for s, x in zip(slots, c) if x) / max(1, kc), 1)}}
    (out / "results.json").write_text(json.dumps({"summary": summary, "per_slot": {s: [bb, cc] for s, bb, cc in zip(slots, b, c)}}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
