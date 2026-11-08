"""Recompute, from the GPU host's episode records, every earlier paired run of the paper's edits (v2mi-f, fingerprint
63db0c..., and abl-imgl8, 5bf8c4...) against the unedited policy in the same project and slots (ali-ms, RTX 3090).

    python3 prior_pairs.py > prior_pairs.json       # on ali-ms; reads ~/vibes/opus-weight-surgery/projects/*

Each project's episodes live in data/episodes/<arm>.<fp6>.s<slot>/summary.json; the unedited arm is base.base.
Output: one record per (project, candidate arm) with the shared slots and the paired counts.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.expanduser("~/vibes/opus-weight-surgery/projects")
FPS = {"63db0c": "v2mi-f (63db0c52a51306c9)", "5bf8c4": "abl-imgl8 (5bf8c4329c1e416a)"}
out = []
for p in sorted(glob.glob(os.path.join(ROOT, "*"))):
    if not os.path.isfile(os.path.join(p, "config.json")) or os.path.basename(p).startswith("cross-pi05"):
        continue
    cfg = json.load(open(os.path.join(p, "config.json")))
    arms = defaultdict(dict)
    for d in glob.glob(os.path.join(p, "data/episodes/*.s*")):
        m = re.match(r"(.+)\.s(\d+)$", os.path.basename(d))
        if not m:
            continue
        try:
            s = json.load(open(os.path.join(d, "summary.json")))
        except Exception:
            continue
        arms[m.group(1)][int(m.group(2))] = bool(s.get("success"))
    b = arms.get("base.base", {})
    for arm, v in sorted(arms.items()):
        fp6 = arm.rsplit(".", 1)[-1]
        if fp6 not in FPS:
            continue
        common = sorted(set(b) & set(v))
        out.append({"project": os.path.basename(p), "task": cfg["task"], "instruction": cfg.get("instruction"),
                    "arm": arm, "edit": FPS[fp6], "slots": [common[0], common[-1]] if common else None,
                    "n": len(common), "unedited_k": sum(b[s] for s in common), "edited_k": sum(v[s] for s in common),
                    "fixed": sum((not b[s]) and v[s] for s in common),
                    "broke": sum(b[s] and (not v[s]) for s in common),
                    "per_slot": {str(s): [int(b[s]), int(v[s])] for s in common}})
json.dump({"host": "ali-ms (2x RTX 3090)", "source": ROOT, "records": out}, sys.stdout, indent=1)
