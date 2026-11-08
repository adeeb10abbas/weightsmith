#!/usr/bin/env bash
# t3und (rules fixed in README 13:54 ET, before the build): T3's RoboLab "specific" wording distilled into T3's default
# wording with t8und's recipe (seq_build.py, text layers 0-26, lam_rel 1e-3, unconditional keys, random control);
# --align auto (unequal token counts -> ps). Build and offline gate on GPU1; closed loop only if the gate passes.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
O3="Put the rubiks cube to the left of the bowl"
G3="Pick up the rubiks cube and place it on the table to the left side of the bowl"
O=runs/q/T3-base-s30000.obs/g1875o0
E=$HOME/cross/cosmos/edits
# build decision: first T3 practice failure (30001, replan 1); checks 30002:1, 30004:2, 30006:3
scripts/gpu.sh 1 t3und-build build $PY scripts/seq_build.py --obs $O/obs_r0_e1.npz --k 1 \
  --orig "$O3" --good "$G3" --layers 0-26 --align auto --lam-rel 1e-3 \
  --out edits/t3und --check $O/obs_r0_e2.npz:1 $O/obs_r0_e4.npz:2 $O/obs_r0_e6.npz:3 > edits/t3und.build.txt 2>&1
# offline gate set: skip the first 6 failure slots, next 6 failure episodes (30009-30011, 30013-30015), replans 1-5
$PY scripts/make_spec.py caps/t3und_t3.json --runs runs/q/T3-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30001,30002,30004,30006,30007,30008 --variant O="$O3" --variant G="$G3" \
   --variant "O+t3und=$O3@$E/t3und.pt" --variant "O+t3und-ctl=$O3@$E/t3und-ctl.pt"
scripts/gpu.sh 1 t3und-chk check $PY scripts/capture.py caps/t3und_t3.json caps/t3und_t3.npz
$PY scripts/offline_check.py caps/t3und_t3.npz --teacher G --student O --variants O+t3und,O+t3und-ctl \
   --out offline/t3und_t3_check.json > offline/t3und_t3_check.txt
python3 - <<'PYEOF'
import json
g = json.load(open("offline/t3und_t3_check.json"))["groups"]["all"]
gap, ctl = g["O+t3und"]["gap_closed_ee_pct"], g["O+t3und-ctl"]["gap_closed_ee_pct"]
ok = gap >= 85 and abs(ctl) <= 15
gate = {"n": g["O+t3und"]["n"], "t3_unedited_to_specific_cm": g["student_vs_teacher"]["ee_end_cm"],
        "t3_edited_to_specific_cm": g["O+t3und"]["vs_teacher"]["ee_end_cm"], "t3_gap_closed_pct": gap,
        "control_gap_closed_pct": ctl, "rule": "T3 gap closed >= 85% and |control gap closed| <= 15%", "pass": ok}
json.dump(gate, open("offline/t3und_gate.json", "w"), indent=1)
print("GATE", json.dumps(gate), flush=True)
PYEOF
if python3 -c "import json,sys; sys.exit(0 if json.load(open('offline/t3und_gate.json'))['pass'] else 1)"; then
  python3 scripts/sessions/queue_front.py jobs/queue.json T3:T3-Et3und-s30000:30000:edits/t3und.pt \
     T3:T3-Ct3und-s30000:30000:edits/t3und-ctl.pt T8:T8-Et3und-s30000:30000:edits/t3und.pt
  python3 scripts/sessions/queue_front.py jobs/queue_gpu0.json T5:T5-Et3und-s30000:30000:edits/t3und.pt
  echo queued
fi
echo SESSION_DONE
