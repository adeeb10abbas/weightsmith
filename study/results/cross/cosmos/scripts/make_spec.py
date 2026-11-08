"""Capture specs: pick recorded decisions from practice runs by a fixed rule and name the replay variants.

Rule (fixed before any edit is built): episodes of the given run records whose outcome matches --outcome (default:
failures), in slot order, at most --episodes of them; from each, the recorded requests with replan index in --k.
Variants are NAME=PROMPT[@EDIT] (an edit path after "@"); --capture lists the variants whose activations are stored.

usage: make_spec.py OUT.json --runs runs/q/T8-base-s30000.json [...] --k 1-6 --episodes 6 \
         --variant O="..." --variant G="..." [--variant O+e="...@edits/x.pt"] [--capture O,G] [--und-layers all]
         [--gen-layers 8-20] [--skip-slots 30000,30001]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rng(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--k", default="1-6")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--outcome", choices=("fail", "success", "any"), default="fail")
    ap.add_argument("--skip-slots", default="")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--capture", default="")
    ap.add_argument("--und-layers", default=None)
    ap.add_argument("--gen-layers", default=None)
    ap.add_argument("--gen-grid", default="2,2")
    a = ap.parse_args()
    skip = {int(s) for s in a.skip_slots.split(",") if s}
    rows = []
    for f in a.runs:
        rows += [r for r in json.loads(Path(f).read_text())["rows"] if r.get("ok")]
    rows.sort(key=lambda r: r["slot"])
    pick = [r for r in rows if r["slot"] not in skip and (a.outcome == "any" or r["success"] == (a.outcome == "success"))]
    pick = pick[: a.episodes]
    decisions = []
    for r in pick:
        n = len(np.load(r["obs"])["seed"])
        decisions += [{"obs": r["obs"], "k": k, "slot": r["slot"]} for k in rng(a.k) if k < n]
    cap = set(filter(None, a.capture.split(",")))
    variants = []
    for v in a.variant:
        name, rest = v.split("=", 1)
        prompt, _, edit = rest.partition("@")
        d = {"name": name, "prompt": prompt, "capture": name in cap}
        if edit:
            d["edit"] = edit
        variants.append(d)
    spec = {"decisions": decisions, "variants": variants, "rule": vars(a)}
    if a.und_layers:
        spec["und_layers"] = "all" if a.und_layers == "all" else rng(a.und_layers)
    if a.gen_layers:
        spec["gen_layers"] = "all" if a.gen_layers == "all" else rng(a.gen_layers)
        spec["gen_grid"] = [int(x) for x in a.gen_grid.split(",")]
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(f"{len(decisions)} decisions from slots {[r['slot'] for r in pick]}; variants {[v['name'] for v in variants]}")


if __name__ == "__main__":
    main()
