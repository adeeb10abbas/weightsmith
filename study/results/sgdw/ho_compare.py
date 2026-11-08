#!/usr/bin/env python3
"""Paired held-out comparisons from weightsmith/heldout.py and ctl_run results (stdlib only).

  python3 ho_compare.py PROJECT_ROOT HELDOUT_TAG [CTL_OUT] [--json OUT]

HELDOUT_TAG is the folder under PROJECT_ROOT/heldout written by heldout.py (arms "base" and "cand"); CTL_OUT is a
folder written by ctl_run_sgdw.py on the same slots (one arm per extra policy). Every pair of arms is compared on
the slots both completed: success counts, Wilson 95% intervals, fixed/broke, exact McNemar p and a 95% bootstrap
interval of the difference (paired.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from paired import bootstrap_diff, mcnemar_exact, wilson


def arms_from(root: Path, tag: str, ctl: str | None) -> dict[str, dict[int, bool]]:
    ho = json.loads((root / "heldout" / tag / "results.json").read_text())
    arms = {"base": {}, "cand": {}}
    for s, (b, c) in ho["per_slot"].items():
        arms["base"][int(s)] = bool(b)
        arms["cand"][int(s)] = bool(c)
    if ctl:
        cr = json.loads((root / "heldout" / ctl / "results.json").read_text())
        for name, per in cr["per_arm"].items():
            arms[name] = {int(s): bool(v["success"]) for s, v in per.items() if v.get("ok")}
    return arms


def pair(a: dict[int, bool], b: dict[int, bool]) -> dict:
    slots = sorted(set(a) & set(b))
    c = {"both": 0, "neither": 0, "fixed": 0, "broke": 0}
    for s in slots:
        x, y = a[s], b[s]
        c["both" if x and y else "neither" if not (x or y) else "fixed" if y else "broke"] += 1
    n, ka, kb = len(slots), sum(a[s] for s in slots), sum(b[s] for s in slots)
    return {"paired_n": n, "slots": [slots[0], slots[-1]] if slots else None, "success_a": ka, "success_b": kb,
            "wilson95_a": wilson(ka, n), "wilson95_b": wilson(kb, n), **c,
            "diff_pp": round(100 * (kb - ka) / n, 1) if n else None,
            "diff_boot95_pp": [round(100 * v, 1) for v in bootstrap_diff(c)],
            "mcnemar_exact_p": mcnemar_exact(c["fixed"], c["broke"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tag")
    ap.add_argument("ctl", nargs="?")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    arms = arms_from(Path(a.root), a.tag, a.ctl)
    out = {"arms": {k: {"n": len(v), "success": sum(v.values())} for k, v in arms.items()}, "pairs": {}}
    for x, y in itertools.combinations(arms, 2):
        r = pair(arms[x], arms[y])
        out["pairs"][f"{y} vs {x}"] = r
        print(f"{y:>6s} vs {x:<6s} n={r['paired_n']:3d}  {r['success_b']:3d} vs {r['success_a']:3d}  "
              f"fixed {r['fixed']:2d} broke {r['broke']:2d}  diff {r['diff_pp']:+.1f} pp {r['diff_boot95_pp']}  "
              f"p={r['mcnemar_exact_p']:.2g}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
