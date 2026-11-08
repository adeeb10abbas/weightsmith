#!/usr/bin/env python3
"""Operator-only arms on held-out slots (where-to-edit copy of study/results/opus/controls/ctl_run.py).

Changes from ctl_run.py: --all-lanes holds every evaluation GPU (blocking) for each chunk, so a 32-slot chunk always
runs as two lanes of one group each and every slot is the first episode of its simulator process; the lanes each
chunk used are recorded in partial.json/results.json.

Original description:
Operator-only extra arms on held-out slots that already have recorded base runs.

Each arm is a frozen candidate (cand:NAME) or the unedited policy under another instruction
(prompt:TEXT). Arms run in chunks of 32 slots, in ABBA order across chunks, with a pause between
chunks so that queued editor evaluations can take the GPU lock. Pairing against the recorded base
runs on the same slots happens afterwards. Nothing is ever overwritten.

  WSP_ROOT=<project> python3 ctl_run.py --out ctl1-90000-64 --slot-start 90000 --n 64 \
      --arm rand=cand:ctl-rand --arm teacher='prompt:Put the cube in the bowl then put the banana in the bowl'
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "weightsmith"))
from weightsmith import config as C  # noqa: E402
from weightsmith.pool import run_pool  # noqa: E402
from weightsmith.locks import GpuLock  # noqa: E402

R = C.PROJECT_ROOT
CFG = json.loads((R / "config.json").read_text())


def fp_of(edits: Path):
    h = hashlib.sha256()
    for f in sorted(edits.glob("*.npy")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slot-start", type=int, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--pause", type=int, default=45)
    ap.add_argument("--all-lanes", action="store_true", help="hold all eval GPUs for every chunk (blocking)")
    a = ap.parse_args()
    if a.slot_start < 80000:
        sys.exit("held-out slots only (>= 80000)")
    out = R / "heldout" / a.out
    if out.exists():
        sys.exit(f"{out} exists; never overwritten")
    out.mkdir(parents=True)
    arms = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        kind, val = rest.split(":", 1)
        if kind == "cand":
            src = R / "cands" / val
            fr = out / f"frozen-{name}"
            fr.mkdir()
            shutil.copy2(src / "workspace.json", fr / "workspace.json")
            (fr / "base").symlink_to((src / "base").resolve())
            shutil.copytree(src / "edits", fr / "edits")
            (fr / "scratch").mkdir()
            arms[name] = {"cand": val, "workspace": str(fr), "fingerprint": fp_of(fr / "edits"), "prompt": None}
        elif kind == "prompt":
            arms[name] = {"cand": None, "workspace": None, "fingerprint": None, "prompt": val}
        else:
            sys.exit(f"bad arm {spec}")
    chunks = [list(range(s, min(s + a.chunk, a.slot_start + a.n))) for s in range(a.slot_start, a.slot_start + a.n, a.chunk)]
    names = list(arms)
    sched = []
    for i, ch in enumerate(chunks):
        sched += [(nm, ch) for nm in (names if i % 2 == 0 else names[::-1])]
    plan = {"arms": arms, "task": CFG["task"], "default_instruction": CFG.get("instruction"),
            "slots": [a.slot_start, a.slot_start + a.n - 1],
            "schedule": [[nm, ch[0], ch[-1]] for nm, ch in sched], "started": time.ctime()}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    res = {nm: {} for nm in names}
    lanes = {}
    for nm, slots in sched:
        arm = arms[nm]
        req = (C.SERVER_GPU, C.SIM_GPU) + (tuple(C.EVAL_EXTRA_GPUS) if a.all_lanes else ())
        with GpuLock(*req, optional=(() if a.all_lanes else C.EVAL_EXTRA_GPUS)) as lk:
            extra = tuple(C.EVAL_EXTRA_GPUS) if a.all_lanes else tuple(lk.acquired)
            rows, rep = run_pool(workspace=arm["workspace"], task=CFG["task"], tag=f"ctl-{nm}", slots=slots,
                                 envs=CFG["envs"], rendering_type=CFG["rendering_type"], video_mode="sensor",
                                 log_root=out / f"logs-{nm}-{slots[0]}", prompt=arm["prompt"],
                                 extra_gpus=extra)
        lanes[f"{nm}-{slots[0]}"] = sorted({r.get("lane") for r in rows if r.get("ok")})
        if arm["workspace"] and arm["workspace"] not in rep:
            raise RuntimeError(f"server did not load the frozen candidate: {rep[:300]}")
        if not arm["workspace"] and '"workspace": null' not in rep:
            raise RuntimeError(f"base server loaded an edit: {rep[:300]}")
        for r in rows:
            res[nm][r["slot"]] = {"ok": r.get("ok"), "success": r.get("success"), "steps": r.get("steps")}
        (out / "partial.json").write_text(json.dumps(res))
        (out / "lanes.json").write_text(json.dumps(lanes))
        print(time.ctime(), nm, slots[0], sum(bool(r.get("success")) for r in rows), "/", len(rows), flush=True)
        time.sleep(a.pause)
    (out / "results.json").write_text(json.dumps({"plan": plan, "finished": time.ctime(), "per_arm": res, "lanes": lanes}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
