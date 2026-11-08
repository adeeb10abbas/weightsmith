#!/usr/bin/env bash
# Closed-form distillation pipeline for one task, offline part (select -> capture -> build -> check). Run it under
# locked.sh (GPU0). Usage: wsp_task.sh NAME RUN_JSON ORIG_TEXT EXPL_TEXT [N_BUILD] [STRIDE]
#   NAME      e.g. t1  (outputs in ~/cross/groot/wsp/NAME)
#   RUN_JSON  the unedited practice run with recorded observations (pool_groot.py --record-obs)
set -eu
NAME="$1"; RUN="$2"; ORIG="$3"; EXPL="$4"; NB="${5:-6}"; STRIDE="${6:-2}"
G=~/cross/groot; PY=$G/Isaac-GR00T/.venv/bin/python; S=$G/src
W=$G/wsp/$NAME; mkdir -p $W; cd $G
python3 - "$ORIG" "$EXPL" > $W/wordings.json <<'EOF'
import json, sys; print(json.dumps({"orig": sys.argv[1], "expl": sys.argv[2]}))
EOF
echo "== $(date +%T) select: candidate replays (actions only, both wordings)"
python3 $S/scripts/wsp_select.py candidates $RUN --stride $STRIDE --out $W/cand.json
[ -f $W/cand_rep/index.json ] || $PY $S/adapters/groot_acts.py --items $W/cand.json --wordings $W/wordings.json --out $W/cand_rep --actions-only --batch 8
python3 $S/scripts/wsp_select.py pick $RUN $W/cand_rep --n-build $NB --out $W/items.json
python3 - $W <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1]); it = json.loads((w / "items.json").read_text())
strip = lambda r: {"obs": r["obs"], "k": r["k"], "slot": r["slot"]}
(w / "build_items.json").write_text(json.dumps([strip(r) for r in it["build"]]))
(w / "check_items.json").write_text(json.dumps([strip(r) for r in it["build"] + it["check"]]))
(w / "groups.json").write_text(json.dumps({"build": [strip(r) for r in it["build"]], "check": [strip(r) for r in it["check"]]}))
EOF
echo "== $(date +%T) capture: build decisions, full activations (backbone + action head), both wordings"
[ -f $W/cap/index.json ] || $PY $S/adapters/groot_acts.py --items $W/build_items.json --wordings $W/wordings.json --out $W/cap --head
echo "== $(date +%T) reference replays of build + check decisions (actions only)"
[ -f $W/ref/index.json ] || $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings $W/wordings.json --out $W/ref --actions-only --batch 8
NI=$(python3 -c "import json; print(len(json.load(open('$W/build_items.json'))))")
IDX=$(python3 -c "print(list(range($NI)))")
echo "== $(date +%T) build: backbone edit + matched random control; DiT edit"
CUDA_VISIBLE_DEVICES= $PY $S/scripts/groot_build.py --capture $W/cap --items "$IDX" --site backbone --out $W/bb.pt --random $W/bb_rand.pt --report $W/bb_build.json
CUDA_VISIBLE_DEVICES= $PY $S/scripts/groot_build.py --capture $W/cap --items "$IDX" --site dit --out $W/dit.pt --report $W/dit_build.json
echo "== $(date +%T) check: edited replays under the original wording"
for E in bb bb_rand dit; do
  [ -f $W/chk_$E/index.json ] || $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings "{\"orig\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$ORIG")}" --out $W/chk_$E --actions-only --batch 8 --edit $W/$E.pt
  python3 $S/scripts/wsp_check.py --ref $W/ref --edited $W/chk_$E --groups $W/groups.json --out $W/check_$E.json
done
echo "== $(date +%T) done"
