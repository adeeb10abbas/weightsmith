#!/usr/bin/env bash
# Operator: one-shot progress summary of an editing run.
# usage: WSP_ROOT=<project root> scripts/progress.sh [AGENT_DIR]   (uses WSP_HOST/WSP_REMOTE_REPO like bin/owsp)
set -u
: "${WSP_ROOT:?set WSP_ROOT}"
read -r -d '' PY <<'PYEOF'
import json, collections
b = json.load(open("state/budget.json"))
print("budget:", {k: b[k] for k in ("episodes","acts_calls","acts_items","evals")})
rows = [json.loads(l) for l in open("data/episodes/index.jsonl")]
base = collections.defaultdict(list)
for r in rows:
    if r["cand"] == "base": base[r["slot"]].append(r["success"])
by = collections.OrderedDict()
for r in rows:
    by.setdefault((r["cand"], r["fingerprint"]), []).append(r)
for (c, fp), rs in by.items():
    k = sum(r["success"] for r in rs); n = len(rs)
    pair = [(sum(base[r["slot"]])/len(base[r["slot"]]), r["success"]) for r in rs if r["slot"] in base and c != "base"]
    extra = ""
    if pair:
        extra = f"  | paired base {sum(p for p,_ in pair):.0f}/{len(pair)}  fixed {sum((p<0.5) and q for p,q in pair)} broke {sum((p>=0.5) and (not q) for p,q in pair)}"
    print(f"{c:28s} fp={fp[:8]} {k}/{n} ({100*k/n:.1f}%){extra}")
PYEOF
if [ -z "${WSP_HOST:-}" ]; then
  (cd "$WSP_ROOT" && python3 -c "$PY")
else
  ssh -o BatchMode=yes "$WSP_HOST" "cd $(printf %q "$WSP_ROOT") && python3 -c $(printf %q "$PY")"
fi
if [ -n "${1:-}" ]; then echo "--- notebook (tail) ---"; tail -25 "$1/NOTEBOOK.md"; fi
