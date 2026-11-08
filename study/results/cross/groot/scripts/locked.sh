#!/usr/bin/env bash
# Run one command on GPU N while holding both of its flocks (the coordinator's lock dir and weightsmith's
# WSP_LOCK_DIR), then release them. The command must not leave GPU processes behind.
# Usage: locked.sh GPU LOGFILE CMD...
set -u
GPU="$1"; LOG="$2"; shift 2
IFS=: read -r L1 L2 <<< "${GROOT_LOCK_DIRS:-$HOME/.cache/weightsmith/locks:$HOME/vibes/opus-weight-surgery/state}"
mkdir -p "$L1"; exec 8>"$L1/gpu$GPU.lock"; flock 8
if [[ -n "${L2:-}" ]]; then mkdir -p "$L2"; exec 9>"$L2/gpu$GPU.lock"; flock 9; fi
echo "$(date +%T) lock gpu$GPU acquired: $*" >> "$LOG"
t0=$(date +%s)
GPU=$GPU CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@" >> "$LOG" 2>&1 8>&- 9>&-
rc=$?
echo "$(date +%T) done rc=$rc gpu_s=$(( $(date +%s) - t0 ))" >> "$LOG"
exit $rc
