#!/usr/bin/env bash
# CPU-only: span-aligned backbone edit (every explicit token contributes) vs token-aligned bb, and the DiT edit,
# on (a) the 11 failed episodes used for no build and (b) 16 grasp-phase decisions. Waits for wsp_early.sh.
set -eu
G=$HOME/cross/groot; PY=$G/Isaac-GR00T/.venv/bin/python; S=$G/src; W=$G/wsp/t1; E=$W/early
cd $G
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=10
while pgrep -f wsp_early.sh > /dev/null; do sleep 10; done
ORIG=$(python3 -c "import json; print(json.dumps({'orig': json.load(open('$W/wordings.json'))['orig']}))")
echo "== $(date +%T) build bb_span"
[ -f $W/bb_span.pt ] || nice $PY $S/scripts/groot_build.py --capture $W/cap --items "$(python3 -c 'print(list(range(6)))')" --site backbone --align span --out $W/bb_span.pt --random $W/bb_span_rand.pt --report $W/bb_span_build.json > $E/build_span.log
for X in bb_span dit; do
  echo "== $(date +%T) $X replays"
  [ -f $E/chk_$X/index.json ] || nice $PY $S/adapters/groot_acts.py --items $E/items.json --wordings "$ORIG" --out $E/chk_$X --actions-only --batch 8 --device cpu --edit $W/$X.pt > $E/chk_$X.log 2>&1
  python3 $S/scripts/wsp_check.py --ref $E/ref --edited $E/chk_$X --groups $E/groups.json --out $E/check_$X.json | cut -c1-400
done
[ -f $W/var/chk_bb_span/index.json ] || nice $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings "$ORIG" --out $W/var/chk_bb_span --actions-only --batch 8 --device cpu --edit $W/bb_span.pt > $E/chk_bbspan_main.log 2>&1
echo "== bb_span on the 12/11 split"
python3 $S/scripts/wsp_check.py --ref $W/var/ref --edited $W/var/chk_bb_span --groups $W/n12/groups.json --out $W/var/check_bb_span_n12.json | cut -c1-300
echo "== $(date +%T) done"
