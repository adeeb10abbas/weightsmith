#!/bin/bash
# Host-side: transfer/retention runs on ONE GPU lane (GPU0) in 16-slot chunks, so every slot is the first episode of its
# simulator process and no lock is held while waiting for another; arms alternate order between the two chunks.
#   setsid -f bash run_transfer2.sh OUT_TAG "PROJECTS" ARM_SPECS... > logs/transfer-2.log 2>&1 < /dev/null
tag=$1; projs=$2; shift 2
P=$HOME/vibes/opus-weight-surgery/projects
S=$P/where-to-edit/scripts
args=(--arm "base=prompt:")
for a in "$@"; do args+=(--arm "$a"); done
for proj in $projs; do
  echo "=== $(date '+%F %T') start $proj ${args[*]}"
  WSP_SERVER_GPU=0 WSP_SIM_GPU=0 WSP_EVAL_EXTRA_GPUS= WSP_POOL_PORT=8600 WSP_ROOT=$P/$proj \
    python3 $S/where_run.py --out "$tag" --slot-start 94016 --n 32 --chunk 16 --pause 20 "${args[@]}"
  echo "=== $(date '+%F %T') end $proj rc=$?"
done
echo "=== $(date '+%F %T') all done"
