#!/usr/bin/env bash
# t8und-p: t8und plus zero-change preservation keys from 65 generic RoboLab instructions (tasks/preserve_generic.json),
# built on GPU0 between side-effect jobs; then closed loop on T5 (GPU0 queue) and T8 practice (GPU1 queue); then offline
# checks (T8 held-aside decisions; T5 decisions where the unedited policy succeeded).
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
O=runs/q/T8-base-s30000.obs/g1875o0
TORCHDYNAMO_DISABLE=1 scripts/gpu.sh 0 t8undp-build build $PY scripts/seq_build.py --obs $O/obs_r0_e0.npz --k 1 \
  --orig "$RF" --good "$TF" --layers 0-26 --align all --lam-rel 1e-3 --preserve-file tasks/preserve_generic.json \
  --out edits/t8undp --check $O/obs_r0_e1.npz:1 $O/obs_r0_e2.npz:2 $O/obs_r0_e3.npz:3 > edits/t8undp.build.txt 2>&1
python3 - <<PYEOF
import json, sys
sys.path.insert(0, "scripts")
from tasks import job
def front(qf, j):
    q = json.load(open(qf))
    if j["tag"] in {x["tag"] for x in q}:
        return
    json.dump([j] + q, open(qf + ".tmp", "w"), indent=1)
    import os; os.replace(qf + ".tmp", qf)
front("jobs/queue_gpu0.json", job("T5", "T5-Et8undp-s30000", 30000, 16, edit="edits/t8undp.pt"))
front("jobs/queue.json", job("T8", "T8-Et8undp-s30000", 30000, 16, edit="edits/t8undp.pt"))
print("queued T5-Et8undp (GPU0) and T8-Et8undp (GPU1)")
PYEOF
# offline: T8 held-aside decisions (slots 30006-30011) and T5 decisions the unedited policy solved
$PY scripts/make_spec.py caps/undp_t8.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30000,30001,30002,30003,30004,30005 --variant O="$RF" --variant G="$TF" \
   --variant "O+und=$RF@$HOME/cross/cosmos/edits/t8und.pt" --variant "O+undp=$RF@$HOME/cross/cosmos/edits/t8undp.pt"
T5="Put the mustard in the left bin"
$PY scripts/make_spec.py caps/undp_t5.json --runs runs/q/T5-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O5="$T5" --variant "O5+und=$T5@$HOME/cross/cosmos/edits/t8und.pt" \
   --variant "O5+undp=$T5@$HOME/cross/cosmos/edits/t8undp.pt"
scripts/gpu.sh 0 undp-chk check $PY scripts/capture.py caps/undp_t8.json caps/undp_t8.npz
scripts/gpu.sh 0 undp-chk5 check $PY scripts/capture.py caps/undp_t5.json caps/undp_t5.npz
$PY scripts/offline_check.py caps/undp_t8.npz --teacher G --student O --variants O+und,O+undp \
   --out offline/t8undp_t8_check.json > offline/t8undp_t8_check.txt
$PY scripts/offline_check.py caps/undp_t5.npz --teacher O5 --student O5+und --variants O5+undp \
   --out offline/t8undp_t5_check.json > offline/t8undp_t5_check.txt
echo SESSION_DONE
