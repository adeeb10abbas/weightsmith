#!/usr/bin/env python3
"""Paired comparison of two arms recorded in a weightsmith project (stdlib only).

An arm is an episode label as written to data/episodes/index.jsonl: ``base``, a candidate name, or a
prompt-override label such as ``base@p1a2b3c`` (owsp eval --prompt). Episodes are paired by slot; a slot run
more than once by an arm uses its first run. Reports both/neither/fixed/broke, the exact two-sided McNemar p on the
discordant pairs, Wilson 95% intervals, and a 95% bootstrap interval for the paired difference (20,000 draws over
the four outcome types, seed 0), as in the weightsmith paper.

  python3 paired.py PROJECT_ROOT ARM_A ARM_B [--slots LO HI] [--json OUT]
  python3 paired.py PROJECT_ROOT --list          # arms, prompts and slot ranges in the project
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_diff(counts: dict[str, int], draws: int = 20000, seed: int = 0) -> list[float]:
    """95% interval of (successes_B - successes_A)/n, resampling pairs over the four outcome types."""
    n = sum(counts.values())
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    cats = ["both", "neither", "fixed", "broke"]
    w = [counts[c] for c in cats]
    vals = []
    for _ in range(draws):
        s = rng.choices(cats, weights=w, k=n)
        vals.append((s.count("fixed") - s.count("broke")) / n)
    vals.sort()
    return [round(vals[int(0.025 * draws)], 4), round(vals[int(0.975 * draws) - 1], 4)]


def load(root: Path) -> list[dict]:
    p = root / "data" / "episodes" / "index.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def by_slot(rows: list[dict], arm: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        if r["cand"] == arm:
            out.setdefault(int(str(r["slot"]).lstrip("s")), r)
    return out


def compare(rows: list[dict], a: str, b: str, lo: int | None = None, hi: int | None = None) -> dict:
    A, B = by_slot(rows, a), by_slot(rows, b)
    slots = sorted(s for s in set(A) & set(B) if (lo is None or s >= lo) and (hi is None or s <= hi))
    counts = {"both": 0, "neither": 0, "fixed": 0, "broke": 0}
    per = {}
    for s in slots:
        x, y = bool(A[s]["success"]), bool(B[s]["success"])
        counts["both" if x and y else "neither" if not (x or y) else "fixed" if y else "broke"] += 1
        per[s] = [int(x), int(y)]
    n = len(slots)
    ka, kb = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"arm_a": a, "arm_b": b, "prompt_a": next(iter(A.values()))["prompt"] if A else None,
            "prompt_b": next(iter(B.values()))["prompt"] if B else None,
            "fingerprint_a": next(iter(A.values()))["fingerprint"] if A else None,
            "fingerprint_b": next(iter(B.values()))["fingerprint"] if B else None,
            "slots": [slots[0], slots[-1]] if slots else None, "paired_n": n,
            "success_a": ka, "success_b": kb, "wilson95_a": wilson(ka, n), "wilson95_b": wilson(kb, n),
            "fixed": counts["fixed"], "broke": counts["broke"], "both": counts["both"], "neither": counts["neither"],
            "diff_pp": round(100 * (kb - ka) / n, 1) if n else None,
            "diff_boot95_pp": [round(100 * v, 1) for v in bootstrap_diff(counts)],
            "mcnemar_exact_p": mcnemar_exact(counts["fixed"], counts["broke"]),
            "per_slot": {str(s): v for s, v in per.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--slots", nargs=2, type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = load(Path(a.root))
    if a.list or len(a.arms) != 2:
        arms: dict[str, dict] = {}
        for r in rows:
            d = arms.setdefault(r["cand"], {"prompt": r["prompt"], "n": 0, "k": 0, "slots": []})
            d["n"] += 1
            d["k"] += int(bool(r["success"]))
            d["slots"].append(int(str(r["slot"]).lstrip("s")))
        for k, d in arms.items():
            print(f"{k:28s} {d['k']:3d}/{d['n']:<3d} slots {min(d['slots'])}-{max(d['slots'])}  {d['prompt']!r}")
        return
    lo, hi = (a.slots or (None, None))
    res = compare(rows, a.arms[0], a.arms[1], lo, hi)
    show = {k: v for k, v in res.items() if k != "per_slot"}
    print(json.dumps(show, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
