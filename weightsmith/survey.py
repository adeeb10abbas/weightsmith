"""Operator: base-policy survey of candidate RoboLab tasks (16 eps each, eval-GPU lock shared with owsp).

usage: python3 weightsmith/survey.py TASK [TASK ...]   -> $WSP_SURVEY_DIR/TASK.json
"""
import fcntl, json, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402
from weightsmith.pool import run_pool  # noqa: E402
tasks = sys.argv[1:]
out = C.SURVEY_DIR; out.mkdir(parents=True, exist_ok=True)
C.LOCK_DIR.mkdir(parents=True, exist_ok=True)
for t in tasks:
    with open(C.LOCK_DIR / f"gpu{C.SIM_GPU}.lock", "w") as lk:  # same lock file as `owsp eval`
        fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time()
        rows, _ = run_pool(workspace=None, task=t, tag=f"survey-{t}", slots=range(60000, 60016), envs=16)
    ok = [r for r in rows if r["ok"]]
    res = {"task": t, "n": len(ok), "success": sum(r["success"] for r in ok), "min": round((time.time()-t0)/60, 1),
           "reasons": Counter((r["reason"] or "")[:90] for r in ok if not r["success"]).most_common(5),
           "steps_success": sorted(r["steps"] for r in ok if r["success"])}
    (out / f"{t}.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res), flush=True)
