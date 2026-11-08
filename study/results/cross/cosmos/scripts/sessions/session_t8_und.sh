#!/usr/bin/env bash
# T8 text-pathway distillation: capture RF/TF text states on RF-failure decisions, build edits + controls, offline check.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
until [ -f runs/q/T8-base-s30000.json ]; do sleep 20; done
# decisions (rule fixed before any edit): RF failures of group 1875 in slot order; build = first 6 episodes, replans 1-5;
# check = the next 6 failure episodes, replans 1-5
$PY scripts/make_spec.py caps/t8_cap.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --variant O="$RF" --variant G="$TF" --capture O,G --und-layers all
scripts/gpu1.sh t8-cap capture $PY scripts/capture.py caps/t8_cap.json caps/t8_cap.npz
for AL in all ps; do
  $PY scripts/build_edit.py caps/t8_cap.npz edits/t8-und-$AL --path und --layers 0-26 --align $AL > edits/t8-und-$AL.build.txt
done
BUILD_SLOTS=$($PY -c "import json; print(\",\".join(sorted({str(d[\"slot\"]) for d in json.load(open(\"caps/t8_cap.json\"))[\"decisions\"]})))")
$PY scripts/make_spec.py caps/t8_chk.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 --skip-slots $BUILD_SLOTS \
   --variant O="$RF" --variant G="$TF" \
   --variant "O+all=$RF@$HOME/cross/cosmos/edits/t8-und-all.pt" --variant "O+all-ctl=$RF@$HOME/cross/cosmos/edits/t8-und-all-ctl.pt" \
   --variant "O+ps=$RF@$HOME/cross/cosmos/edits/t8-und-ps.pt" --variant "G+all=$TF@$HOME/cross/cosmos/edits/t8-und-all.pt"
$PY - <<PYEOF
import json
a = json.load(open("caps/t8_cap.json")); b = json.load(open("caps/t8_chk.json"))
b["decisions"] = a["decisions"] + b["decisions"]   # build decisions first, then held-aside check decisions
json.dump(b, open("caps/t8_chk.json", "w"), indent=1)
print(len(a["decisions"]), "build +", len(b["decisions"]) - len(a["decisions"]), "check decisions")
PYEOF
scripts/gpu1.sh t8-chk check $PY scripts/capture.py caps/t8_chk.json caps/t8_chk.npz
N=$($PY -c "import json; print(len(json.load(open(\"caps/t8_cap.json\"))[\"decisions\"]))")
M=$($PY -c "import json; print(len(json.load(open(\"caps/t8_chk.json\"))[\"decisions\"]))")
$PY scripts/offline_check.py caps/t8_chk.npz --teacher G --student O --variants O+all,O+all-ctl,O+ps,G+all \
   --groups build:0-$((N-1)),check:$N-$((M-1)) --out offline/t8_und_check.json > offline/t8_und_check.txt
echo SESSION_DONE
