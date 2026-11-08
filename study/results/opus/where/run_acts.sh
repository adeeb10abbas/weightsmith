#!/bin/bash
# Host-side: run owsp acts captures of the where-to-edit project in sequence (each takes the GPU lock itself).
#   nohup setsid bash run_acts.sh chk-base a-orig ... > logs/acts-<tag>.log 2>&1 < /dev/null &
cd ~/weightsmith || exit 1
export WSP_ROOT=$HOME/vibes/opus-weight-surgery/projects/where-to-edit
for s in "$@"; do
  echo "=== $(date '+%F %T') start $s"
  t0=$(date +%s)
  python3 weightsmith/owsp.py acts < "$WSP_ROOT/scripts/specs/$s.json"
  echo "=== $(date '+%F %T') end $s rc=$? wall=$(( $(date +%s) - t0 ))s"
done
echo "=== $(date '+%F %T') all done"
