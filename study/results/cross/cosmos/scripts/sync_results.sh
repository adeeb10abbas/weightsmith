#!/usr/bin/env bash
# Copy the complete per-slot run records (runs/q/<tag>.json), edit build reports, offline checks and the GPU ledger
# from each GPU host into results/<...>/<machine>/ (no observations, logs or weights). Run from anywhere on the Mac.
#   scripts/sync_results.sh                 # every host in HOSTS below that answers
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
# machine label | ssh target (with options) | work dir on the host
HOSTS=(
  "ali-ms|ali-ms|cross/cosmos"
  "korea-rtxpro4000|-p 40797 root@211.234.50.230|/workspace/cross/cosmos"
)
for h in "${HOSTS[@]}"; do
  IFS='|' read -r M T D <<<"$h"
  SSH="ssh -o ConnectTimeout=8 ${T% *}"; TGT=${T##* }
  [ "$T" = "$TGT" ] && SSH="ssh -o ConnectTimeout=8"
  $SSH "$TGT" true 2>/dev/null || { echo "$M: unreachable, skipped"; continue; }
  mkdir -p "$HERE/results/runs/$M" "$HERE/results/edits/$M" "$HERE/results/offline/$M"
  rsync -a -e "$SSH" --include='*.json' --exclude='*' "$TGT:$D/runs/q/" "$HERE/results/runs/$M/"
  rm -f "$HERE/results/runs/$M"/*.incomplete-*.json
  rsync -a -e "$SSH" "$TGT:$D/runs/gpu_ledger.jsonl" "$HERE/results/gpu_ledger_$M.jsonl" 2>/dev/null || true
  rsync -a -e "$SSH" --include='*.json' --exclude='*' "$TGT:$D/edits/" "$HERE/results/edits/$M/" 2>/dev/null || true
  rsync -a -e "$SSH" --include='*.json' --exclude='*' "$TGT:$D/offline/" "$HERE/results/offline/$M/" 2>/dev/null || true
  echo "$M: $(ls "$HERE/results/runs/$M" | wc -l) runs"
done
