#!/usr/bin/env bash
# Pull run summaries and WSP build/check reports from ali-ms into the repo, then rebuild tables, metrics and plots.
# Run from anywhere on the laptop: bash study/results/cross/groot/scripts/sync_results.sh
set -eu
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
G=$ROOT/study/results/cross/groot
mkdir -p $G/results/runs $G/results/wsp $G/results/jobs
for d in fm wd ed ho se rp; do
  rsync -a --include='*.json' --exclude='*' ali-ms:cross/groot/runs/$d/ $G/results/runs/ 2>/dev/null || true
done
rsync -a --prune-empty-dirs --include='*/' --include='*.json' --exclude='cand_rep/**' --exclude='cap/**' --exclude='ref/**' \
  --exclude='chk_*/**' --exclude='*' ali-ms:cross/groot/wsp/ $G/results/wsp/ 2>/dev/null || true
rsync -a ali-ms:cross/groot/jobs/queue.txt ali-ms:cross/groot/jobs/done $G/results/jobs/ 2>/dev/null || true
rsync -a ali-ms:cross/groot/jobs/worker-g*.log $G/results/jobs/ 2>/dev/null || true
cd $ROOT && python3 $G/scripts/analyze.py > /dev/null && echo "synced: $(ls $G/results/runs | wc -l) runs"
