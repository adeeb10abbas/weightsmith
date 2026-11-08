#!/bin/sh
# Track C': copy the fresh per-job records (JSON only) from ali-ms and rebuild the fresh metrics.
set -e
cd "$(dirname "$0")"
mkdir -p ../results/fresh/runs
rsync -a --include='*.json' --exclude='*' ali-ms:cross/pi05/fresh/runs/ ../results/fresh/runs/
rsync -a ali-ms:cross/pi05/fresh/ledger.jsonl ../results/fresh/ 2>/dev/null || true
rsync -a ali-ms:cross/pi05/fresh/jobs/queue.json ../results/fresh/queue.json
python3 fresh.py
