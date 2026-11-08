#!/usr/bin/env python3
"""GPU time used by the where-to-edit study (run on the GPU host).

Closed-loop chunks: wall time from the chunk's log directory creation (birth time) to its rows.json, times the number of
lanes (one GPU per lane). Activation captures: wall time of each `owsp acts` call (one GPU), from scripts/logs/acts-*.log.
"""
import json, os, re, subprocess
from pathlib import Path

P = Path.home() / "vibes/opus-weight-surgery/projects"
out = {"chunks": [], "acts": []}
for proj in ["where-to-edit", "where-glasses", "where-left", "where-cube"]:
    for d in sorted((P / proj / "heldout").glob("*/logs-*")):
        rows = d / "rows.json"
        if not rows.exists():
            continue
        birth = int(subprocess.run(["stat", "-c", "%W", str(d)], capture_output=True, text=True).stdout.strip())
        lanes = len(list(d.glob("server-l*.log")))
        wall = os.path.getmtime(rows) - birth
        out["chunks"].append({"project": proj, "run": d.parent.name, "chunk": d.name, "lanes": lanes, "wall_s": round(wall),
                              "gpu_s": round(wall * lanes)})
for f in sorted((P / "where-to-edit/scripts/logs").glob("acts-*.log")):
    txt = f.read_text()
    for m in re.finditer(r'\{"id": "([^"]+)".*?"sec": ([0-9.]+)\}', txt):
        w_ = re.search(r"end %s rc=\d+ wall=(\d+)s" % re.escape(m.group(1)), txt)
        out["acts"].append({"capture": m.group(1), "capture_s": float(m.group(2)),
                            "wall_incl_queue_s": int(w_.group(1)) if w_ else None})
# a capture holds one GPU for its model load (about 10-60 s) plus capture_s; count 60 s of load per call (upper end)
acts_gpu = sum(a["capture_s"] + 60 for a in out["acts"])
w = sum(c["wall_s"] for c in out["chunks"]) + acts_gpu
g = sum(c["gpu_s"] for c in out["chunks"]) + acts_gpu
out["total_wall_h"] = round(w / 3600, 2)
out["total_gpu_h"] = round(g / 3600, 2)
out["closed_loop_wall_h"] = round(sum(c["wall_s"] for c in out["chunks"]) / 3600, 2)
out["acts_gpu_h_upper"] = round(acts_gpu / 3600, 2)
print(json.dumps(out, indent=1))
