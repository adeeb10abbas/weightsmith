#!/usr/bin/env bash
# Resume of session_t8_gen.sh after its build step (capture.py rewrites <stem>.json with the spec under "spec",
# so the decisions are read from either place). Same rule, same decisions, nothing re-chosen.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
decs() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); d=d.get(\"decisions\") or d[\"spec\"][\"decisions\"]; print($1)" "$2"; }
BUILD_SLOTS=$(decs "\",\".join(sorted({str(x[\"slot\"]) for x in d}))" caps/t8g_cap.json)
echo "build slots $BUILD_SLOTS"
$PY scripts/make_spec.py caps/t8g_chk.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 --skip-slots $BUILD_SLOTS \
   --variant O="$RF" --variant G="$TF" \
   --variant "O+gen=$RF@$HOME/cross/cosmos/edits/t8gen.pt" --variant "O+gen-ctl=$RF@$HOME/cross/cosmos/edits/t8gen-ctl.pt" \
   --variant "O+und=$RF@$HOME/cross/cosmos/edits/t8und.pt" --variant "O+und-ctl=$RF@$HOME/cross/cosmos/edits/t8und-ctl.pt"
$PY - <<PYEOF
import json
a = json.load(open("caps/t8g_cap.json")); a = a.get("decisions") or a["spec"]["decisions"]
b = json.load(open("caps/t8g_chk.json"))
n_chk = len(b["decisions"])
b["decisions"] = a + b["decisions"]   # build decisions first, then held-aside check decisions
json.dump(b, open("caps/t8g_chk.json", "w"), indent=1)
print(len(a), "build +", n_chk, "check decisions")
PYEOF
N=$(decs "len(d)" caps/t8g_cap.json)
M=$(decs "len(d)" caps/t8g_chk.json)
echo "N=$N M=$M"
scripts/gpu1.sh t8g-chk check $PY scripts/capture.py caps/t8g_chk.json caps/t8g_chk.npz
$PY scripts/offline_check.py caps/t8g_chk.npz --teacher G --student O --variants O+gen,O+gen-ctl,O+und,O+und-ctl \
   --groups build:0-$((N-1)),check:$N-$((M-1)) --out offline/t8_offline_check.json > offline/t8_offline_check.txt
echo SESSION_DONE
