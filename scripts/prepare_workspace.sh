#!/usr/bin/env bash
# One-time (GPU host): export every editable pi0.5 tensor as layer-split bf16 .npy files.
# Produces $BASE_WORKSPACE/{workspace.json,base/,edits/,scratch/}: 730 tensors, 2,823,447,440
# editable values out of 3,353,433,872 (VLM blocks, action expert, action projections, time MLP).
# Needs ~11 GB of disk and the OpenPI venv. Candidates symlink to $BASE_WORKSPACE/base.
# The reference export used --max-changed-fraction 1.0 (owsp sets 1.0 per candidate anyway).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(cd "$repo" && python3 -c 'from weightsmith import config as C; import shlex
print("OPENPI_PY=" + shlex.quote(C.OPENPI_PY)); print("CKPT=" + shlex.quote(str(C.CHECKPOINT))); print("WS=" + shlex.quote(str(C.BASE_WS)))')"
cd "$repo"
# == weightsmith.lab.openpi_vlm_full.prepare_workspace(CHECKPOINT_DIR, BASE_WORKSPACE, 1.0, include_action=True)
"$OPENPI_PY" -m weightsmith.lab.openpi_vlm_full prepare \
  --checkpoint "$CKPT" --workspace "$WS" --include-action --max-changed-fraction 1.0
