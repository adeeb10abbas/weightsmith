#!/usr/bin/env bash
# T8 generation-pathway (video/action tokens) distillation, the analogue of the pi0.5 image-cell edit:
# capture RF/TF gen states on RF-failure decisions of practice group 1875, build edit + control, offline check.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
LAY=${LAY:-8-20}; CAPLAY=${CAPLAY:-7-20}
# decisions (rule fixed before any edit is built): RF failures of T8-base-s30000 in slot order; build = first 6
# episodes, replans 1-5; check = the next 6 failure episodes, replans 1-5 (never used to build)
$PY scripts/make_spec.py caps/t8g_cap.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --variant O="$RF" --variant G="$TF" --capture O,G --gen-layers $CAPLAY --gen-grid 2,2
scripts/gpu1.sh t8g-cap capture $PY scripts/capture.py caps/t8g_cap.json caps/t8g_cap.npz
$PY scripts/build_edit.py caps/t8g_cap.npz edits/t8gen --path gen --layers $LAY > edits/t8gen.build.txt
BUILD_SLOTS=$($PY -c "import json; print(\",\".join(sorted({str(d[\"slot\"]) for d in json.load(open(\"caps/t8g_cap.json\"))[\"decisions\"]})))")
$PY scripts/make_spec.py caps/t8g_chk.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 --skip-slots $BUILD_SLOTS \
   --variant O="$RF" --variant G="$TF" \
   --variant "O+gen=$RF@$HOME/cross/cosmos/edits/t8gen.pt" --variant "O+gen-ctl=$RF@$HOME/cross/cosmos/edits/t8gen-ctl.pt" \
   --variant "O+und=$RF@$HOME/cross/cosmos/edits/t8und.pt" --variant "O+und-ctl=$RF@$HOME/cross/cosmos/edits/t8und-ctl.pt"
$PY - <<PYEOF
import json
a = json.load(open("caps/t8g_cap.json")); b = json.load(open("caps/t8g_chk.json"))
b["decisions"] = a["decisions"] + b["decisions"]   # build decisions first, then held-aside check decisions
json.dump(b, open("caps/t8g_chk.json", "w"), indent=1)
print(len(a["decisions"]), "build +", len(b["decisions"]) - len(a["decisions"]), "check decisions")
PYEOF
scripts/gpu1.sh t8g-chk check $PY scripts/capture.py caps/t8g_chk.json caps/t8g_chk.npz
N=$($PY -c "import json; print(len(json.load(open(\"caps/t8g_cap.json\"))[\"decisions\"]))")
M=$($PY -c "import json; print(len(json.load(open(\"caps/t8g_chk.json\"))[\"decisions\"]))")
$PY scripts/offline_check.py caps/t8g_chk.npz --teacher G --student O --variants O+gen,O+gen-ctl,O+und,O+und-ctl \
   --groups build:0-$((N-1)),check:$N-$((M-1)) --out offline/t8_offline_check.json > offline/t8_offline_check.txt
echo SESSION_DONE
