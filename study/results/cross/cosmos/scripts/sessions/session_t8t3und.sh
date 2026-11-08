#!/usr/bin/env bash
# t8t3und (stacked; rules fixed in README 13:54 ET): t8und loaded, then t3und's recipe built on the T8-edited weights
# (seq_build.py --base-edit edits/t8und.pt; teacher states from the unedited model). Served as the dense exact stack
# file edits/t8t3und-stack.pt. Build and offline gates on GPU0; closed loop only if both gates pass.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
O3="Put the rubiks cube to the left of the bowl"
G3="Pick up the rubiks cube and place it on the table to the left side of the bowl"
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
O=runs/q/T3-base-s30000.obs/g1875o0
E=$HOME/cross/cosmos/edits
scripts/gpu.sh 0 t8t3und-build build $PY scripts/seq_build.py --obs $O/obs_r0_e1.npz --k 1 \
  --orig "$O3" --good "$G3" --layers 0-26 --align auto --lam-rel 1e-3 --base-edit edits/t8und.pt --no-ctl \
  --out edits/t8t3und --check $O/obs_r0_e2.npz:1 $O/obs_r0_e4.npz:2 $O/obs_r0_e6.npz:3 > edits/t8t3und.build.txt 2>&1
$PY scripts/make_spec.py caps/t8t3und_t3.json --runs runs/q/T3-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30001,30002,30004,30006,30007,30008 --variant G="$G3" --variant "O8=$O3@$E/t8und.pt" \
   --variant "O+st=$O3@$E/t8t3und-stack.pt"
$PY scripts/make_spec.py caps/t8t3und_t8.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30000,30001,30002,30003,30004,30005 --variant O="$RF" --variant G="$TF" \
   --variant "O+und=$RF@$E/t8und.pt" --variant "O+st=$RF@$E/t8t3und-stack.pt"
scripts/gpu.sh 0 t8t3und-chk check bash -c "for c in t3 t8; do $PY scripts/capture.py caps/t8t3und_\$c.json caps/t8t3und_\$c.npz || exit 1; done"
$PY scripts/offline_check.py caps/t8t3und_t3.npz --teacher G --student O8 --variants O+st \
   --out offline/t8t3und_t3_check.json > offline/t8t3und_t3_check.txt
$PY scripts/offline_check.py caps/t8t3und_t8.npz --teacher G --student O --variants O+und,O+st \
   --out offline/t8t3und_t8_check.json > offline/t8t3und_t8_check.txt
python3 - <<'PYEOF'
import json
g3 = json.load(open("offline/t8t3und_t3_check.json"))["groups"]["all"]
g8 = json.load(open("offline/t8t3und_t8_check.json"))["groups"]["all"]
gap3, gap8 = g3["O+st"]["gap_closed_ee_pct"], g8["O+st"]["gap_closed_ee_pct"]
ok = gap3 >= 85 and gap8 >= 85
gate = {"t3_n": g3["O+st"]["n"], "t3_t8und_to_specific_cm": g3["student_vs_teacher"]["ee_end_cm"],
        "t3_stack_to_specific_cm": g3["O+st"]["vs_teacher"]["ee_end_cm"], "t3_gap_closed_pct": gap3,
        "t8_n": g8["O+st"]["n"], "t8_gap_closed_pct": gap8, "t8_t8und_gap_closed_pct": g8["O+und"]["gap_closed_ee_pct"],
        "rule": "T3 gap (T8-edited default -> unedited specific) closed >= 85% and T8 RF -> TF gap closed >= 85%",
        "pass": ok}
json.dump(gate, open("offline/t8t3und_gate.json", "w"), indent=1)
print("GATE", json.dumps(gate), flush=True)
PYEOF
if python3 -c "import json,sys; sys.exit(0 if json.load(open('offline/t8t3und_gate.json'))['pass'] else 1)"; then
  python3 scripts/sessions/queue_front.py jobs/queue.json --after T8-Et3und-s30000 \
     T8:T8-Et8t3und-s30000:30000:edits/t8t3und-stack.pt T3:T3-Et8t3und-s30000:30000:edits/t8t3und-stack.pt
  python3 scripts/sessions/queue_front.py jobs/queue_gpu0.json --after T5-Et3und-s30000 \
     T5:T5-Et8t3und-s30000:30000:edits/t8t3und-stack.pt
  echo queued
fi
echo SESSION_DONE
