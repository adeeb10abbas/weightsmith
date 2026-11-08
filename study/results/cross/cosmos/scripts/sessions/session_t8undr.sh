#!/usr/bin/env bash
# t8und-r (relation-aware preservation, coordinator spec 07:55 ET): t8und + zero-change keys from the 24 left/right
# instructions of other RoboLab tasks (tasks/preserve_relation.json), with its random control. Build on GPU0 (free first; policy numerics are bit-identical on GPU0 and GPU1), offline
# gates (T8 gap closed >= 85%; T5 and T3 deviation from unedited <= half of the t8und deviation), then closed loop if
# they pass.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
O=runs/q/T8-base-s30000.obs/g1875o0
E=$HOME/cross/cosmos/edits
TORCHDYNAMO_DISABLE=1 scripts/gpu.sh 0 t8undr-build build $PY scripts/seq_build.py --obs $O/obs_r0_e0.npz --k 1 \
  --orig "$RF" --good "$TF" --layers 0-26 --align all --lam-rel 1e-3 --preserve-file tasks/preserve_relation.json \
  --preserve-stride 1 --gpu-solve --early-stop \
  --out edits/t8undr --check $O/obs_r0_e1.npz:1 $O/obs_r0_e2.npz:2 $O/obs_r0_e3.npz:3 > edits/t8undr.build.txt 2>&1
$PY scripts/make_spec.py caps/undr_t8.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30000,30001,30002,30003,30004,30005 --variant O="$RF" --variant G="$TF" \
   --variant "O+undr=$RF@$E/t8undr.pt" --variant "O+undr-ctl=$RF@$E/t8undr-ctl.pt"
T5="Put the mustard in the left bin"
$PY scripts/make_spec.py caps/undr_t5.json --runs runs/q/T5-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O5="$T5" --variant "O5+und=$T5@$E/t8und.pt" --variant "O5+undr=$T5@$E/t8undr.pt"
T3="Put the rubiks cube to the left of the bowl"
$PY scripts/make_spec.py caps/undr_t3.json --runs runs/q/T3-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O3="$T3" --variant "O3+und=$T3@$E/t8und.pt" --variant "O3+undr=$T3@$E/t8undr.pt"
for c in t8 t5 t3; do scripts/gpu.sh 0 undr-chk-$c check $PY scripts/capture.py caps/undr_$c.json caps/undr_$c.npz; done
$PY scripts/offline_check.py caps/undr_t8.npz --teacher G --student O --variants O+undr,O+undr-ctl \
   --out offline/t8undr_t8_check.json > offline/t8undr_t8_check.txt
$PY scripts/offline_check.py caps/undr_t5.npz --teacher O5 --student O5+und --variants O5+undr \
   --out offline/t8undr_t5_check.json > offline/t8undr_t5_check.txt
$PY scripts/offline_check.py caps/undr_t3.npz --teacher O3 --student O3+und --variants O3+undr \
   --out offline/t8undr_t3_check.json > offline/t8undr_t3_check.txt
python3 - <<'PYEOF'
import json, os, sys
sys.path.insert(0, "scripts")
from tasks import job
g8 = json.load(open("offline/t8undr_t8_check.json"))["groups"]["all"]
g5 = json.load(open("offline/t8undr_t5_check.json"))["groups"]["all"]
g3 = json.load(open("offline/t8undr_t3_check.json"))["groups"]["all"]
gap = g8["O+undr"]["gap_closed_ee_pct"]
d5, u5 = g5["O5+undr"]["vs_teacher"]["ee_end_cm"], g5["student_vs_teacher"]["ee_end_cm"]
d3, u3 = g3["O3+undr"]["vs_teacher"]["ee_end_cm"], g3["student_vs_teacher"]["ee_end_cm"]
ok = gap >= 85 and d5 <= 0.5 * u5 and d3 <= 0.5 * u3
gate = {"t8_gap_closed_pct": gap, "t5_dev_cm": d5, "t5_t8und_dev_cm": u5, "t3_dev_cm": d3, "t3_t8und_dev_cm": u3,
        "rule": "T8 gap closed >= 85% and T5, T3 deviation from unedited <= 0.5 x the t8und deviation", "pass": ok}
json.dump(gate, open("offline/t8undr_gate.json", "w"), indent=1)
print("GATE", json.dumps(gate))
def front(qf, j, after=None):
    q = json.load(open(qf))
    if j["tag"] in {x["tag"] for x in q}:
        return
    tags = [x["tag"] for x in q]
    i = tags.index(after) + 1 if after in tags else 0
    q.insert(i, j)
    json.dump(q, open(qf + ".tmp", "w"), indent=1); os.replace(qf + ".tmp", qf)
if ok:
    front("jobs/queue_gpu0.json", job("T5", "T5-Et8undr-s30000", 30000, 16, edit="edits/t8undr.pt"))
    front("jobs/queue.json", job("T3", "T3-Et8undr-s30000", 30000, 16, edit="edits/t8undr.pt"))
    front("jobs/queue.json", job("T8", "T8-Et8undr-s30000", 30000, 16, edit="edits/t8undr.pt"), after="T3-Et8undr-s30000")
    print("queued T5-Et8undr (GPU0), T3-Et8undr and T8-Et8undr (GPU1)")
PYEOF
echo SESSION_DONE
