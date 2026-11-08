#!/usr/bin/env bash
# Offline comparison of edit constructions on CPU (no GPU lock needed): reference replays (orig, expl) and edited
# replays (orig wording) of the same build + check decisions, all on CPU so the numerics match.
# Usage: wsp_variants.sh NAME "VARIANT:BUILD_FLAGS" ...      (outputs in ~/cross/groot/wsp/NAME/var/)
set -eu
NAME="$1"; shift
G=${GROOT_WS:-$HOME/cross/groot}; PY=$G/Isaac-GR00T/.venv/bin/python; S=$G/src; W=$G/wsp/$NAME; V=$W/var
mkdir -p $V; cd $G
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
ORIG=$(python3 -c "import json; print(json.dumps({'orig': json.load(open('$W/wordings.json'))['orig']}))")
NI=$(python3 -c "import json; print(len(json.load(open('$W/build_items.json'))))")
IDX=$(python3 -c "print(list(range($NI)))")
[ -f $V/ref/index.json ] || nice $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings $W/wordings.json --out $V/ref --actions-only --batch 8 --device cpu
for spec in "$@"; do
  vn="${spec%%:*}"; flags="${spec#*:}"
  echo "== $(date +%T) $vn: $flags"
  # shellcheck disable=SC2086
  [ -f $V/$vn.pt ] || nice $PY $S/scripts/groot_build.py --capture $W/cap --items "$IDX" --out $V/$vn.pt --report $V/$vn.build.json $flags > $V/$vn.build.log
  [ -f $V/chk_$vn/index.json ] || nice $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings "$ORIG" --out $V/chk_$vn --actions-only --batch 8 --device cpu --edit $V/$vn.pt > $V/chk_$vn.log 2>&1
  python3 $S/scripts/wsp_check.py --ref $V/ref --edited $V/chk_$vn --groups $W/groups.json --out $V/check_$vn.json
done
echo "== $(date +%T) done"
