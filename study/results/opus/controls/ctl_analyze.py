#!/usr/bin/env python3
"""Pair the control arms (heldout/ctl1-90000-64) with the recorded held-out runs on the same slots.

Recorded runs on slots 90000-90127 (first half run base-then-candidate):
  v2mi-f-RubiksCubeThenBananaTask-90000-128     base run A + full edit
  abl-imgl8-RubiksCubeThenBananaTask-90000-128  base run B + 0.16% edit
Prints one JSON object with paired stats for every comparison on the slots the control arms covered.
"""
import json, math, sys
from pathlib import Path

P = Path.home() / "vibes/opus-weight-surgery/projects/cube-then-banana/heldout"
ctl_dir = P / (sys.argv[1] if len(sys.argv) > 1 else "ctl1-90000-64")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot(x, y, B=20000, seed=0):
    import random
    rng = random.Random(seed); n = len(x); ds = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(sum(y[i] - x[i] for i in idx) / n)
    ds.sort()
    return [round(100 * ds[int(0.025 * B)], 1), round(100 * ds[int(0.975 * B)], 1)]


arms = {}
for tag, nm in (("v2mi-f-RubiksCubeThenBananaTask-90000-128", "A"), ("abl-imgl8-RubiksCubeThenBananaTask-90000-128", "B")):
    r = json.loads((P / tag / "results.json").read_text())["per_slot"]
    arms[f"base_{nm}"] = {int(s): bool(v[0]) for s, v in r.items()}
    arms["edit_full" if nm == "A" else "edit_016"] = {int(s): bool(v[1]) for s, v in r.items()}
src = ctl_dir / ("results.json" if (ctl_dir / "results.json").exists() else "partial.json")
res = json.loads(src.read_text())
res = res.get("per_arm", res)
for nm, d in res.items():
    if d:
        arms[nm] = {int(s): bool(v["success"]) for s, v in d.items() if v.get("ok")}

ctl_names = [nm for nm in res if res[nm]]
slots = sorted(set.intersection(*(set(arms[k]) for k in arms)))
out = {"source": str(src), "n_slots": len(slots), "slot_range": [slots[0], slots[-1]] if slots else None,
       "success": {k: sum(arms[k][s] for s in slots) for k in arms}, "comparisons": []}
pairs = [(c, b) for c in ctl_names for b in ("base_A", "base_B")] + [("edit_full", "base_A"), ("edit_016", "base_B"), ("base_B", "base_A")]
pairs += [(c, "edit_full") for c in ctl_names] + [(c, "edit_016") for c in ctl_names]
for cand, ref in pairs:
    x = [int(arms[ref][s]) for s in slots]; y = [int(arms[cand][s]) for s in slots]
    fixed = sum((not a) and b for a, b in zip(x, y)); broke = sum(a and (not b) for a, b in zip(x, y))
    out["comparisons"].append({"cand": cand, "ref": ref, "n": len(slots), "cand_succ": sum(y), "ref_succ": sum(x),
                               "fixed": fixed, "broke": broke, "diff_pp": round(100 * (sum(y) - sum(x)) / len(slots), 1),
                               "boot95_pp": boot(x, y), "mcnemar_p": mcnemar(fixed, broke),
                               "cand_wilson": wilson(sum(y), len(slots)), "ref_wilson": wilson(sum(x), len(slots))})
print(json.dumps(out, indent=1))
