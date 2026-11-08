#!/usr/bin/env python3
"""GPU time used by the sgdw runs, from the runtime's own records (stdlib only; run on the GPU host).

Every closed-loop pool (practice evaluation, held-out half, control chunk) is timed from the first timestamp Isaac
Sim writes to its simulator log, minus one minute for the policy server's start (the server starts first), to the
write of its rows.json, and multiplied by its number of lanes (one GPU each). Lock waiting is excluded. Activation captures are counted as 1 GPU-minute each
(model load plus at most a minute of replay; the worker logs only the replay seconds).

  gpu_ledger.py SGDW_LOG_DIR PROJECT_ROOT [PROJECT_ROOT ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import time
from pathlib import Path

ISO = re.compile(rb"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z")


def first_stamp(log: Path) -> float | None:
    with log.open("rb") as f:
        m = ISO.search(f.read(200000))
    return calendar.timegm(time.strptime(m.group(1).decode(), "%Y-%m-%dT%H:%M:%S")) if m else None


def pool(d: Path) -> dict | None:
    rows = d / "rows.json"
    stamps = [t for t in (first_stamp(p) for p in d.glob("sim-l*.log")) if t]
    if not rows.exists() or not stamps:
        return None
    lanes = len(json.loads(rows.read_text())["lanes"])
    mins = (rows.stat().st_mtime - min(stamps)) / 60.0 + 1.0
    return {"pool": str(d), "lanes": lanes, "wall_min": round(mins, 2), "gpu_min": round(mins * lanes, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    items = []
    for d in sorted(Path(a.logdir).glob("pool-*")):
        r = pool(d)
        if r:
            items.append({"kind": "practice eval", **r})
    for proj in a.projects:
        root = Path(proj)
        for d in sorted((root / "heldout").glob("*/logs-*")):
            r = pool(d)
            if r:
                items.append({"kind": "held-out/control", **r})
        for d in sorted((root / "data" / "acts").glob("*/index.json")):
            items.append({"kind": "acts", "pool": str(d.parent), "lanes": 1, "wall_min": None, "gpu_min": 1.0})
    tot = round(sum(i["gpu_min"] for i in items), 1)
    for i in items:
        print(f"{i['kind']:17s} {Path(i['pool']).name:48s} lanes {i['lanes']}  {i['gpu_min']:6.1f} GPU-min")
    print(f"total {tot} GPU-min = {tot / 60:.2f} GPU-hours")
    if a.json:
        Path(a.json).write_text(json.dumps({"items": items, "total_gpu_min": tot}, indent=1))


if __name__ == "__main__":
    main()
