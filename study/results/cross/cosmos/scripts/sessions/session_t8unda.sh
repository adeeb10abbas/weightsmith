#!/usr/bin/env bash
# t8und-a (TARGETED, chosen after the T5/T3 side effect was seen): t8und plus zero-change preservation keys from every
# RoboLab instruction except the shared-set tasks (tasks/preserve_all_other.json, 108 prompts, incl. the mirror
# "Put the mustard in the right bin"). Built on GPU1 with early-stopped text passes; then offline checks; then closed
# loop on T5 (GPU0), T8 and T3 (GPU1) practice.
set -euo pipefail
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF="Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
O=runs/q/T8-base-s30000.obs/g1875o0
TORCHDYNAMO_DISABLE=1 scripts/gpu.sh 1 t8unda-build build $PY scripts/seq_build.py --obs $O/obs_r0_e0.npz --k 1 \
  --orig "$RF" --good "$TF" --layers 0-26 --align all --lam-rel 1e-3 --preserve-file tasks/preserve_all_other.json \
  --preserve-stride 3 --no-ctl --gpu-solve --early-stop \
  --out edits/t8unda --check $O/obs_r0_e1.npz:1 $O/obs_r0_e2.npz:2 $O/obs_r0_e3.npz:3 > edits/t8unda.build.txt 2>&1
$PY scripts/make_spec.py caps/unda_t8.json --runs runs/q/T8-base-s30000.json --k 1-5 --episodes 6 \
   --skip-slots 30000,30001,30002,30003,30004,30005 --variant O="$RF" --variant G="$TF" \
   --variant "O+unda=$RF@$HOME/cross/cosmos/edits/t8unda.pt"
T5="Put the mustard in the left bin"
$PY scripts/make_spec.py caps/unda_t5.json --runs runs/q/T5-base-s30000.json --k 1-5 --episodes 8 --outcome success \
   --variant O5="$T5" --variant "O5+und=$T5@$HOME/cross/cosmos/edits/t8und.pt" \
   --variant "O5+unda=$T5@$HOME/cross/cosmos/edits/t8unda.pt"
scripts/gpu.sh 1 unda-chk check $PY scripts/capture.py caps/unda_t8.json caps/unda_t8.npz
scripts/gpu.sh 1 unda-chk5 check $PY scripts/capture.py caps/unda_t5.json caps/unda_t5.npz
$PY scripts/offline_check.py caps/unda_t8.npz --teacher G --student O --variants O+unda \
   --out offline/t8unda_t8_check.json > offline/t8unda_t8_check.txt
$PY scripts/offline_check.py caps/unda_t5.npz --teacher O5 --student O5+und --variants O5+unda \
   --out offline/t8unda_t5_check.json > offline/t8unda_t5_check.txt
echo OFFLINE_DONE
