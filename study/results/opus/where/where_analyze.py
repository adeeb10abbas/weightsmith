#!/usr/bin/env python3
"""Paired statistics for a ctl_run.py output directory (all arms paired with one shared base arm).

  python3 where_analyze.py RESULTS_JSON [--ref base] [--vs v1full] > analysis.json

RESULTS_JSON is <project>/heldout/<out>/results.json (or partial.json while running). For every arm: successes,
Wilson 95% interval, mean steps of successful episodes; for every comparison: fixed/broke, difference in points,
paired bootstrap 95% interval (20,000 resamples of slots, seed 0, as in study/results/opus/controls/ctl_analyze.py)
and the exact two-sided McNemar p. Also splits by schedule chunk (run order alternates between chunks).
"""
import argparse, json, math, random
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
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
    rng = random.Random(seed); n = len(x); ds = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(sum(y[i] - x[i] for i in idx) / n)
    ds.sort()
    return [round(100 * ds[int(0.025 * B)], 1), round(100 * ds[int(0.975 * B)], 1)]


def compare(arms, cand, ref, slots):
    x = [int(arms[ref][s]["success"]) for s in slots]; y = [int(arms[cand][s]["success"]) for s in slots]
    fixed = sum((not a) and b for a, b in zip(x, y)); broke = sum(a and (not b) for a, b in zip(x, y))
    return {"cand": cand, "ref": ref, "n": len(slots), "cand_succ": sum(y), "ref_succ": sum(x), "fixed": fixed,
            "broke": broke, "diff_pp": round(100 * (sum(y) - sum(x)) / len(slots), 1), "boot95_pp": boot(x, y),
            "mcnemar_p": mcnemar(fixed, broke)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--vs", default="v1full", help="second reference arm (comparisons of every other arm against it)")
    ap.add_argument("--slots", default=None, help="restrict to slot ranges, e.g. 94016-94031,94048-94063")
    a = ap.parse_args()
    p = Path(a.results)
    d = json.loads(p.read_text())
    per_arm = d.get("per_arm", d)
    plan = d.get("plan") or json.loads((p.parent / "plan.json").read_text())
    arms = {nm: {int(s): v for s, v in r.items() if v.get("ok")} for nm, r in per_arm.items() if r}
    names = [nm for nm in plan["arms"] if nm in arms]
    slots = sorted(set.intersection(*(set(arms[nm]) for nm in names)))
    if a.slots:
        keep = set()
        for rg in a.slots.split(","):
            lo, hi = (int(x) for x in rg.split("-"))
            keep |= set(range(lo, hi + 1))
        slots = [s for s in slots if s in keep]
    lanes = {}
    for nm, s0, _ in plan["schedule"]:
        f = p.parent / f"logs-{nm}-{s0}" / "rows.json"
        if f.exists():
            lanes[f"{nm}-{s0}"] = sorted({r.get("lane") for r in json.loads(f.read_text())["rows"] if r.get("ok")})
    out = {"source": str(p), "task": plan.get("task"), "slot_filter": a.slots, "lanes_per_chunk": lanes,
           "n_slots": len(slots),
           "slot_range": [slots[0], slots[-1]] if slots else None,
           "fingerprints": {nm: plan["arms"][nm].get("fingerprint") for nm in names},
           "arms": {}, "comparisons": [], "by_chunk": []}
    for nm in names:
        k = sum(bool(arms[nm][s]["success"]) for s in slots)
        st = [arms[nm][s]["steps"] for s in slots if arms[nm][s]["success"] and arms[nm][s].get("steps") is not None]
        out["arms"][nm] = {"success": k, "n": len(slots), "wilson95": wilson(k, len(slots)),
                           "mean_steps_success": round(sum(st) / len(st), 1) if st else None}
    pairs = [(nm, a.ref) for nm in names if nm != a.ref]
    if a.vs in names:
        pairs += [(nm, a.vs) for nm in names if nm not in (a.ref, a.vs)]
    for c, r in pairs:
        out["comparisons"].append(compare(arms, c, r, slots))
    chunks = sorted({(s0, s1) for _, s0, s1 in plan["schedule"]})
    for s0, s1 in chunks:
        cs = [s for s in slots if s0 <= s <= s1]
        order = [nm for nm, a0, _ in plan["schedule"] if a0 == s0]
        out["by_chunk"].append({"slots": [s0, s1], "order": order,
                                "success": {nm: sum(bool(arms[nm][s]["success"]) for s in cs) for nm in names},
                                "vs_ref": {nm: compare(arms, nm, a.ref, cs) for nm in names if nm != a.ref}})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
