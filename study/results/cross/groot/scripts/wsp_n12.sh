#!/usr/bin/env bash
# CPU-only: does a 12-decision build generalise better than the 6-decision one? (T1, backbone)
set -eu
G=$HOME/cross/groot; PY=$G/Isaac-GR00T/.venv/bin/python; S=$G/src; W=$G/wsp/t1; V=$W/n12
mkdir -p $V; cd $G
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=10
python3 $S/scripts/wsp_select.py pick runs/fm/fm-T1.json $W/cand_rep --n-build 12 --out $V/items.json
python3 - $V <<'PY'
import json, sys
from pathlib import Path
v = Path(sys.argv[1]); it = json.loads((v / "items.json").read_text())
strip = lambda r: {"obs": r["obs"], "k": r["k"], "slot": r["slot"]}
(v / "build_items.json").write_text(json.dumps([strip(r) for r in it["build"]]))
(v / "groups.json").write_text(json.dumps({"build": [strip(r) for r in it["build"]], "check": [strip(r) for r in it["check"]]}))
PY
echo "== $(date +%T) capture"
[ -f $V/cap/index.json ] || nice $PY $S/adapters/groot_acts.py --items $V/build_items.json --wordings $W/wordings.json --out $V/cap --device cpu > $V/cap.log 2>&1
echo "== $(date +%T) build"
nice $PY $S/scripts/groot_build.py --capture $V/cap --items "$(python3 -c 'print(list(range(12)))')" --site backbone --out $V/bb12.pt --random $V/bb12_rand.pt --report $V/bb12_build.json > $V/build.log
ORIG=$(python3 -c "import json; print(json.dumps({'orig': json.load(open('$W/wordings.json'))['orig']}))")
echo "== $(date +%T) edited replay"
[ -f $V/chk_bb12/index.json ] || nice $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings "$ORIG" --out $V/chk_bb12 --actions-only --batch 8 --device cpu --edit $V/bb12.pt > $V/chk.log 2>&1
echo "== bb12 (12-decision build) on the 12/11 split"
python3 $S/scripts/wsp_check.py --ref $W/var/ref --edited $V/chk_bb12 --groups $V/groups.json --out $V/check_bb12.json | cut -c1-200
echo "== bb (6-decision build) on the same 12/11 split"
python3 $S/scripts/wsp_check.py --ref $W/var/ref --edited $W/var/chk_v1 --groups $V/groups.json --out $V/check_bb6.json | cut -c1-200
echo "== $(date +%T) done"
