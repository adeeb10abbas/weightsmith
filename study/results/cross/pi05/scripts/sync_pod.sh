#!/bin/bash
# Copy the pod's per-job records into this repo (no videos or episode outputs: they stay on the pod), rebuild
# the metrics and plots. Usage (laptop, repo root): study/results/cross/pi05/scripts/sync_pod.sh [ssh-port host]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-17491}"; HOST="${2:-root@175.155.64.181}"
mkdir -p "$here/results/runs" "$here/results/pod"
rsync -az -e "ssh -p $PORT -o ConnectTimeout=20" --include='*.json' --exclude='*' "$HOST:/workspace/cross/pi05/runs/" "$here/results/runs/"
rsync -az -e "ssh -p $PORT -o ConnectTimeout=20" "$HOST:/workspace/cross/pi05/ledger.jsonl" "$HOST:/workspace/cross/pi05/jobs/queue.json" "$here/results/pod/" 2>/dev/null || true
ssh -p "$PORT" -o ConnectTimeout=20 "$HOST" 'cd /workspace/cross/pi05 && find videos -name "*.mp4" 2>/dev/null | sort' > "$here/results/pod/videos.txt" < /dev/null 2>/dev/null || true
python3 "$here/scripts/analyze.py"
