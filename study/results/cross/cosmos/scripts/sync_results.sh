#!/usr/bin/env bash
# Copy the per-slot run records, edit build reports and the GPU ledger from the GPU host into results/
# (no observations, logs or weights). Run from anywhere on the Mac.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$HERE/results/runs" "$HERE/results/edits" "$HERE/results/offline"
rsync -a --include='*/' --include='*.json' --exclude='*.obs/***' --exclude='*' \
  --prune-empty-dirs ali-ms:cross/cosmos/runs/ "$HERE/results/runs_raw/"
# flatten: one <tag>.json per run (map/, words/, edits/, heldout/ ... on the host)
find "$HERE/results/runs_raw" -name '*.json' -exec cp {} "$HERE/results/runs/" \;
rm -rf "$HERE/results/runs_raw"
rsync -a ali-ms:cross/cosmos/runs/gpu_ledger.jsonl "$HERE/results/gpu_ledger.jsonl" 2>/dev/null || true
rsync -a --include='*.json' --exclude='*' ali-ms:cross/cosmos/edits/ "$HERE/results/edits/" 2>/dev/null || true
rsync -a --include='*.json' --exclude='*' ali-ms:cross/cosmos/offline/ "$HERE/results/offline/" 2>/dev/null || true
ls "$HERE/results/runs" | wc -l
