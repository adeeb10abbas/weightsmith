"""Operator-only held-out evaluation: base vs frozen candidate on unseen noise slots.

Not reachable from owsp. Usage (on the GPU host):
  WSP_ROOT=<project root> python3 weightsmith/heldout.py --cand NAME --slot-start 90000 --n 200 [--task T]
Freezes a copy of the candidate edits first, verifies its fingerprint, then runs
base and candidate on the identical slots (order alternates by half), and reports
paired statistics (exact McNemar) plus Wilson intervals.
"""
from __future__ import annotations

import argparse, hashlib, json, math, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402
from weightsmith.pool import run_pool  # noqa: E402
from weightsmith.locks import GpuLock  # noqa: E402

R = C.PROJECT_ROOT  # env WSP_ROOT
CFG = json.loads((R / "config.json").read_text())


def fp_of(edits: Path):
    h = hashlib.sha256()
    for f in sorted(edits.glob("*.npy")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--slot-start", type=int, default=90000)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--task", default=CFG["task"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--envs", type=int, default=CFG["envs"])
    a = ap.parse_args()
    if a.slot_start < 80000:
        sys.exit("held-out slots must be >= 80000 (practice slots are 20000..20255)")
    # check everything before the run folder exists: a failed run can never be repeated under its tag
    if a.slot_start % a.envs:
        sys.exit(f"--slot-start must be a multiple of envs = {a.envs} (slots run in whole groups); "
                 f"e.g. {a.slot_start - a.slot_start % a.envs + a.envs}")
    if a.n <= 0 or a.n % (2 * a.envs):
        sys.exit(f"--n must be a positive multiple of 2*envs = {2 * a.envs} (both halves must be whole groups); e.g. 192")
    src = R / "cands" / a.cand
    if not (src / "edits").is_dir():
        sys.exit(f"no candidate {a.cand!r} in {R / 'cands'}")
    tag = a.tag or f"{a.cand}-{a.task}-{a.slot_start}-{a.n}"
    out = R / "heldout" / tag
    if out.exists():
        sys.exit(f"{out} exists; held-out runs are never repeated or overwritten")
    out.mkdir(parents=True)
    frozen = out / "frozen"
    frozen.mkdir()
    for f in ("workspace.json",):
        shutil.copy2(src / f, frozen / f)
    (frozen / "base").symlink_to((src / "base").resolve())
    shutil.copytree(src / "edits", frozen / "edits")
    (frozen / "scratch").mkdir()
    fp = fp_of(frozen / "edits")
    plan = {"cand": a.cand, "fingerprint": fp, "task": a.task, "slots": [a.slot_start, a.slot_start + a.n - 1],
            "order": "first half: base then cand; second half: cand then base", "started": time.ctime()}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    half = a.n // 2
    s1 = list(range(a.slot_start, a.slot_start + half))
    s2 = list(range(a.slot_start + half, a.slot_start + a.n))
    res = {"base": {}, "cand": {}}
    sched = [("base", s1), ("cand", s1), ("cand", s2), ("base", s2)]
    for arm, slots in sched:
        with GpuLock(C.SERVER_GPU, C.SIM_GPU, optional=C.EVAL_EXTRA_GPUS) as lk:
            rows, rep = run_pool(workspace=str(frozen) if arm == "cand" else None, task=a.task, tag=f"ho-{arm}",
                                 slots=slots, envs=a.envs, rendering_type=CFG["rendering_type"],
                                 video_mode="sensor", log_root=out / f"logs-{arm}-{slots[0]}",
                                 extra_gpus=tuple(lk.acquired))
        if arm == "cand" and str(frozen) not in rep:
            raise RuntimeError(f"server did not load the frozen candidate: {rep[:300]}")
        if arm == "base" and '"workspace": null' not in rep:
            raise RuntimeError(f"base server loaded an edit: {rep[:300]}")
        for r in rows:
            res[arm][r["slot"]] = r
        (out / "partial.json").write_text(json.dumps({k: {s: v["success"] for s, v in d.items()} for k, d in res.items()}))
    slots = [s for s in range(a.slot_start, a.slot_start + a.n) if res["base"].get(s, {}).get("ok") and res["cand"].get(s, {}).get("ok")]
    b = [bool(res["base"][s]["success"]) for s in slots]
    c = [bool(res["cand"][s]["success"]) for s in slots]
    kb, kc, n = sum(b), sum(c), len(slots)
    fixed = sum((not x) and y for x, y in zip(b, c))
    broke = sum(x and (not y) for x, y in zip(b, c))
    summary = {**plan, "finished": time.ctime(), "paired_n": n,
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
