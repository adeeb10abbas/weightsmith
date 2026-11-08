#!/usr/bin/env bash
# t8und-s (coordinator approval 09:40 ET): --align ps (only the shared template prefix and suffix states are mapped; the differing instruction words are unconstrained) + zero-change keys from the 65 generic
# preservation prompts (tasks/preserve_generic.json, every 3rd position), with its random control. Build on GPU0 (free first; policy numerics are bit-identical on GPU0 and GPU1), offline
# gates (T8 gap closed >= 85%; T5 and T3 deviation from unedited below t8und-r (5.88 / 6.46 cm)), then closed loop if
# they pass.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
O=runs/q/T8-base-s30000.obs/g1875o0
E=$HOME/cross/cosmos/edits
TORCHDYNAMO_DISABLE=1 scripts/gpu.sh 0 t8unds-build build $PY scripts/seq_build.py --obs $O/obs_r0_e0.npz --k 1 \
  --orig "$RF" --good "$TF" --layers 0-26 --align ps --lam-rel 1e-3 --preserve-file tasks/preserve_generic.json \
  --preserve-stride 3 --gpu-solve --early-stop \
  --out edits/t8unds --check $O/obs_r0_e1.npz:1 $O/obs_r0_e2.npz:2 $O/obs_r0_e3.npz:3 > edits/t8unds.build.txt 2>&1
$PY scripts/make_spec.py caps/unds_t8.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30000,30001,30002,30003,30004,30005 --variant O="$RF" --variant G="$TF" \
   --variant "O+unds=$RF@$E/t8unds.pt" --variant "O+unds-ctl=$RF@$E/t8unds-ctl.pt"
T5="Put the mustard in the left bin"
$PY scripts/make_spec.py caps/unds_t5.json --runs runs/q/T5-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O5="$T5" --variant "O5+und=$T5@$E/t8und.pt" --variant "O5+unds=$T5@$E/t8unds.pt"
T3="Put the rubiks cube to the left of the bowl"
$PY scripts/make_spec.py caps/unds_t3.json --runs runs/q/T3-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O3="$T3" --variant "O3+und=$T3@$E/t8und.pt" --variant "O3+unds=$T3@$E/t8unds.pt"
# one GPU0 lock hold for the three captures (so the pool does not slip in between them)
scripts/gpu.sh 0 unds-chk check bash -c "for c in t8 t5 t3; do $PY scripts/capture.py caps/unds_\$c.json caps/unds_\$c.npz || exit 1; done"
$PY scripts/offline_check.py caps/unds_t8.npz --teacher G --student O --variants O+unds,O+unds-ctl \
   --out offline/t8unds_t8_check.json > offline/t8unds_t8_check.txt
$PY scripts/offline_check.py caps/unds_t5.npz --teacher O5 --student O5+und --variants O5+unds \
   --out offline/t8unds_t5_check.json > offline/t8unds_t5_check.txt
$PY scripts/offline_check.py caps/unds_t3.npz --teacher O3 --student O3+und --variants O3+unds \
   --out offline/t8unds_t3_check.json > offline/t8unds_t3_check.txt
python3 - <<'PYEOF'
import json, os, sys
sys.path.insert(0, "scripts")
from tasks import job
g8 = json.load(open("offline/t8unds_t8_check.json"))["groups"]["all"]
g5 = json.load(open("offline/t8unds_t5_check.json"))["groups"]["all"]
g3 = json.load(open("offline/t8unds_t3_check.json"))["groups"]["all"]
gap = g8["O+unds"]["gap_closed_ee_pct"]
d5, u5 = g5["O5+unds"]["vs_teacher"]["ee_end_cm"], g5["student_vs_teacher"]["ee_end_cm"]
d3, u3 = g3["O3+unds"]["vs_teacher"]["ee_end_cm"], g3["student_vs_teacher"]["ee_end_cm"]
ok = gap >= 85 and d5 < 5.88 and d3 < 6.46  # below t8und-r (offline/t8undr_gate.json)
gate = {"t8_gap_closed_pct": gap, "t5_dev_cm": d5, "t5_t8und_dev_cm": u5, "t3_dev_cm": d3, "t3_t8und_dev_cm": u3,
        "rule": "T8 gap closed >= 85% and T5, T3 deviation from unedited below t8und-r (5.88 / 6.46 cm)", "pass": ok}
json.dump(gate, open("offline/t8unds_gate.json", "w"), indent=1)
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
    front("jobs/queue_gpu0.json", job("T5", "T5-Et8unds-s30000", 30000, 16, edit="edits/t8unds.pt"))
    front("jobs/queue.json", job("T3", "T3-Et8unds-s30000", 30000, 16, edit="edits/t8unds.pt"))
    front("jobs/queue.json", job("T8", "T8-Et8unds-s30000", 30000, 16, edit="edits/t8unds.pt"), after="T3-Et8unds-s30000")
    print("queued T5-Et8unds (GPU0), T3-Et8unds and T8-Et8unds (GPU1)")
PYEOF
echo SESSION_DONE
