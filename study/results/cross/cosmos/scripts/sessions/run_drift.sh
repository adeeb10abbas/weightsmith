#!/usr/bin/env bash
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
O=runs/q/T8-base-s30000.obs/g1875o0/obs_r0_e0.npz:1
scripts/gpu1.sh drift-t8und check $PY scripts/prompt_drift.py edits/t8und.pt $O offline/drift_t8und.json \
  "T8=Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle." \
  "T8-TF=Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box." \
  "T1=Put the cube then the banana in the bowl" "T2=Pick up the eye glasses" \
  "T3=Put the rubiks cube to the left of the bowl" "T4=Stack the yellow block on the red block" \
  "T5=Put the mustard in the left bin" "T6=Stack the rubiks cubes in a tower" \
  "T7=Pick up the green vegetable block" "R1=Put the cube in the bowl" \
  "BOWL-RF=Move the left bowl so that the right bowl is underneath and supporting the left bowl." \
  "T1-alt=Put the cube in the bowl then put the banana in the bowl" \
  "T3-alt=Pick up the rubiks cube and place it on the table to the left side of the bowl"
