#!/usr/bin/env bash
# Operator (GPU host): pre-run base-policy practice episodes for a project, uncharged.
# usage: WSP_ROOT=<project root> scripts/base_practice.sh N    (N multiple of 16; e.g. 64 or 128)
# Runs `owsp eval base --n N` (slots s20000..), then resets the budget so the editor starts at 0.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:?usage: base_practice.sh N}"
: "${WSP_ROOT:?set WSP_ROOT}"
export WSP_ROOT
python3 "$repo/weightsmith/owsp.py" eval base --n "$N" | tee "$WSP_ROOT/base-practice-$N.out"
mkdir -p "$WSP_ROOT/state"
printf '{"episodes": 0, "acts_calls": 0, "acts_items": 0, "evals": 0, "note": "base %s pre-run by operator, not charged"}\n' "$N" \
  > "$WSP_ROOT/state/budget.json"
