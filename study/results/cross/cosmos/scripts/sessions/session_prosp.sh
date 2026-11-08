#!/usr/bin/env bash
# Prospective token-count test (pre-registered in README 14:59 ET). t8und's recipe, unchanged, on CODE's RF -> TF pair.
# usage: session_prosp.sh CODE NAME ALIGN "RF" "TF" [closed]   (ALIGN all for the equal-length pair, auto for S1F;
# "closed" queues practice edit + control on GPU1 if the offline gate passes). One GPU1 lock hold for build + capture.
set -euo pipefail
cd ~/cross/cosmos
export CODE=$1 NAME=$2 ALIGN=$3 RF=$4 TF=$5 MODE=${6:-offline}
export PY=cosmos-framework/.venv/bin/python E=$HOME/cross/cosmos/edits
eval "$(python3 scripts/sessions/prosp_decisions.py runs/q/$CODE-rf-s30000.json)"
export B C1 C2 C3 SKIP HELD
echo "build $B checks $C1 $C2 $C3; held-aside failure slots $HELD"
scripts/gpu.sh 1 $NAME build bash -c '
set -e
$PY scripts/seq_build.py --obs $B --k 1 --orig "$RF" --good "$TF" --layers 0-26 --align $ALIGN --lam-rel 1e-3 \
  --out edits/$NAME --check $C1 $C2 $C3 > edits/$NAME.build.txt 2>&1
$PY scripts/make_spec.py caps/${NAME}_chk.json --runs runs/q/$CODE-rf-s30000.json --k 1-5 --episodes 6 \
  --skip-slots $SKIP --variant O="$RF" --variant G="$TF" --variant "O+e=$RF@$E/$NAME.pt" \
  --variant "O+e-ctl=$RF@$E/$NAME-ctl.pt"
$PY scripts/capture.py caps/${NAME}_chk.json caps/${NAME}_chk.npz
'
$PY scripts/offline_check.py caps/${NAME}_chk.npz --teacher G --student O --variants O+e,O+e-ctl \
  --out offline/${NAME}_check.json > offline/${NAME}_check.txt
python3 - <<'PYEOF'
import json, os
name, code = os.environ["NAME"], os.environ["CODE"]
b = json.load(open(f"edits/{name}.json"))
g = json.load(open(f"offline/{name}_check.json"))["groups"]["all"]
gap, ctl = g["O+e"]["gap_closed_ee_pct"], g["O+e-ctl"]["gap_closed_ee_pct"]
ok = gap >= 85 and abs(ctl) <= 15
gate = {"code": code, "edit": name, "align": b["align"], "n_tokens_rf": b["n_tokens_orig"], "n_tokens_tf": b["n_tokens_good"],
        "matched": b["matched"], "n": g["O+e"]["n"], "held_aside_slots": os.environ["HELD"],
        "rf_to_tf_cm": g["student_vs_teacher"]["ee_end_cm"], "edited_to_tf_cm": g["O+e"]["vs_teacher"]["ee_end_cm"],
        "gap_closed_pct": gap, "closer_to_tf": g["O+e"]["closer_to_teacher_ee"], "control_gap_closed_pct": ctl,
        "rule": "gap closed >= 85% and |control gap closed| <= 15%", "pass": ok}
json.dump(gate, open(f"offline/{name}_gate.json", "w"), indent=1)
print("GATE", json.dumps(gate), flush=True)
if ok and os.environ["MODE"] == "closed":
    q = json.load(open("jobs/queue.json"))
    task = json.load(open(f"runs/q/{code}-rf-s30000.json"))["task"]
    new = [{"tag": f"{code}-E{name}-s30000", "task": task, "slot_start": 30000, "n": 16, "prompt": os.environ["RF"],
            "edit": f"edits/{name}.pt"},
           {"tag": f"{code}-C{name}-s30000", "task": task, "slot_start": 30000, "n": 16, "prompt": os.environ["RF"],
            "edit": f"edits/{name}-ctl.pt"}]
    tags = {j["tag"] for j in q}
    q = [j for j in new if j["tag"] not in tags] + q
    json.dump(q, open("jobs/queue.json.tmp", "w"), indent=1); os.replace("jobs/queue.json.tmp", "jobs/queue.json")
    print("queued", [j["tag"] for j in new], flush=True)
PYEOF
echo SESSION_DONE
