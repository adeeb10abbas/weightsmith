#!/usr/bin/env python3
"""Summarise operator held-out runs of the controlled study (projects ctrl-*)."""
import json
from pathlib import Path
P = Path.home() / "vibes/opus-weight-surgery/projects"
for d in sorted(P.glob("ctrl-*")):
    for r in sorted((d / "heldout").glob("*/")) if (d / "heldout").exists() else []:
        f = r / "results.json"
        if f.exists():
            s = json.loads(f.read_text())["summary"]
            print(f'{d.name:15s} {s["cand"]:4s} n={s["paired_n"]:3d} base {s["base_success"]:2d} -> cand {s["cand_success"]:2d}  '
                  f'fixed {s["fixed"]:2d} broke {s["broke"]:2d}  p={s["mcnemar_exact_p"]:.3g}  fp={s["fingerprint"]}')
        else:
            part = r / "partial.json"
            done = {k: len(v) for k, v in json.loads(part.read_text()).items()} if part.exists() else {}
            print(f'{d.name:15s} {r.name}: running {done}')
