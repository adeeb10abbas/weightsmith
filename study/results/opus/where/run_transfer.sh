#!/bin/bash
# Host-side: transfer and retention runs of the where-to-edit study (32 fresh slots per task, both GPUs per chunk).
#   setsid -f bash run_transfer.sh OUT_TAG ARM_SPECS... > logs/transfer.log 2>&1 < /dev/null
# Example ARM_SPECS: v1full=cand:v1-full p1=cand:p1-blind a1=cand:a1   (base = unedited, task's own instruction)
tag=$1; shift
P=$HOME/vibes/opus-weight-surgery/projects
S=$P/where-to-edit/scripts
args=(--arm "base=prompt:")
for a in "$@"; do args+=(--arm "$a"); done
for proj in where-glasses where-left where-cube; do
  echo "=== $(date '+%F %T') start $proj ${args[*]}"
  WSP_ROOT=$P/$proj python3 $S/where_run.py --all-lanes --out "$tag" --slot-start 94016 --n 32 --chunk 32 --pause 20 "${args[@]}"
  echo "=== $(date '+%F %T') end $proj rc=$?"
done
echo "=== $(date '+%F %T') all done"
