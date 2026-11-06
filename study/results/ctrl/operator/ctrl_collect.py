#!/usr/bin/env python3
"""Collect the controlled study (projects ctrl-*) into one JSON: dev results per candidate (paired with base by slot,
exact McNemar), candidate notes and sizes (owsp diff), budgets, and operator held-out runs. Runs on the GPU host."""
import collections, json, math, os, subprocess, sys
from pathlib import Path

ROOT = Path.home() / "vibes/opus-weight-surgery/projects"
KEYS = ["ctb", "left", "stack3", "yellowred", "mustard", "glasses", "flatbox"]


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def diff_size(proj, cand):
    env = dict(os.environ, WSP_ROOT=str(proj))
    try:
        out = subprocess.run([sys.executable, str(Path.home() / "weightsmith/weightsmith/owsp.py"), "diff", cand],
                             env=env, capture_output=True, text=True, timeout=600).stdout
        j = json.loads(out[out.index("{"):])
        return {k: j.get(k) for k in ("changed_elements", "changed_fraction", "tensors_changed", "edited_tensors") if k in j}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


out = {}
for k in KEYS:
    proj = ROOT / f"ctrl-{k}"
    cfg = json.loads((proj / "config.json").read_text())
    rec = {"task": cfg.get("task"), "instruction": cfg.get("instruction"), "budget": None, "dev": {}, "heldout": []}
    b = proj / "state/budget.json"
    rec["budget"] = json.loads(b.read_text()) if b.exists() else None
    idx = proj / "data/episodes/index.jsonl"
    rows = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()] if idx.exists() else []
    base = {}
    for r in rows:
        if r["cand"] == "base":
            base.setdefault(r["slot"], bool(r["success"]))
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r["cand"], []).append(r)
    for c, rs in by.items():
        d = {"success": sum(bool(r["success"]) for r in rs), "n": len(rs)}
        if c != "base":
            pairs = [(base[r["slot"]], bool(r["success"])) for r in rs if r["slot"] in base]
            fx = sum((not x) and y for x, y in pairs); br = sum(x and (not y) for x, y in pairs)
            d.update(paired_n=len(pairs), paired_base=sum(x for x, _ in pairs), fixed=fx, broke=br, p=mcnemar_p(fx, br))
            meta = proj / "cands" / c / "meta.json"
            if meta.exists():
                d["note"] = json.loads(meta.read_text()).get("note")
            d["size"] = diff_size(proj, c)
        rec["dev"][c] = d
    hd = proj / "heldout"
    for r in sorted(hd.glob("*/")) if hd.exists() else []:
        f = r / "results.json"
        if f.exists():
            rec["heldout"].append(json.loads(f.read_text())["summary"])
    out[k] = rec
print(json.dumps(out, indent=1))
