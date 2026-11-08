#!/usr/bin/env bash
# CPU-only offline check on grasp-phase decisions (T1): does an edit leave the early approach/grasp alone?
# Builds bb_tok (image tokens pinned) from the 6 build decisions, then replays 16 early decisions
# (k = 4, 8, 12, 16 of slots 30016-30019; none used to build) under orig / expl / bb / bb_tok.
set -eu
G=$HOME/cross/groot; PY=$G/Isaac-GR00T/.venv/bin/python; S=$G/src; W=$G/wsp/t1; E=$W/early
mkdir -p $E; cd $G
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=10
echo "== $(date +%T) build bb_tok"
[ -f $W/bb_tok.pt ] || nice $PY $S/scripts/groot_build.py --capture $W/cap --items "$(python3 -c 'print(list(range(6)))')" --site backbone --image-keys tokens --out $W/bb_tok.pt --random $W/bb_tok_rand.pt --report $W/bb_tok_build.json > $E/build_tok.log
python3 - $E <<'PY'
import json, sys
from pathlib import Path
e = Path(sys.argv[1])
d = json.load(open(Path.home() / "cross/groot/runs/fm/fm-T1.json"))
rows = {r["slot"]: r for r in d["rows"]}
its = [{"obs": rows[s]["obs"], "k": k, "slot": s} for s in (30016, 30017, 30018, 30019) for k in (4, 8, 12, 16)]
(e / "items.json").write_text(json.dumps(its))
(e / "groups.json").write_text(json.dumps({"early": its}))
print("success of these slots:", {s: rows[s]["success"] for s in (30016, 30017, 30018, 30019)})
PY
ORIG=$(python3 -c "import json; print(json.dumps({'orig': json.load(open('$W/wordings.json'))['orig']}))")
echo "== $(date +%T) reference replays"
[ -f $E/ref/index.json ] || nice $PY $S/adapters/groot_acts.py --items $E/items.json --wordings $W/wordings.json --out $E/ref --actions-only --batch 8 --device cpu > $E/ref.log 2>&1
for X in bb bb_tok; do
  echo "== $(date +%T) edited replay $X"
  [ -f $E/chk_$X/index.json ] || nice $PY $S/adapters/groot_acts.py --items $E/items.json --wordings "$ORIG" --out $E/chk_$X --actions-only --batch 8 --device cpu --edit $W/$X.pt > $E/chk_$X.log 2>&1
  python3 $S/scripts/wsp_check.py --ref $E/ref --edited $E/chk_$X --groups $E/groups.json --out $E/check_$X.json | cut -c1-400
done
echo "== $(date +%T) bb_tok on the build/check decisions"
[ -f $W/var/chk_bb_tok/index.json ] || nice $PY $S/adapters/groot_acts.py --items $W/check_items.json --wordings "$ORIG" --out $W/var/chk_bb_tok --actions-only --batch 8 --device cpu --edit $W/bb_tok.pt > $E/chk_bbtok_main.log 2>&1
python3 $S/scripts/wsp_check.py --ref $W/var/ref --edited $W/var/chk_bb_tok --groups $W/groups.json --out $W/var/check_bb_tok.json | cut -c1-300
echo "== $(date +%T) done"
