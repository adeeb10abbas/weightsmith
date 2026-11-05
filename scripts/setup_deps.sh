#!/usr/bin/env bash
# Reproduce the exact policy + simulator stack used for the weightsmith results.
#
#   scripts/setup_deps.sh            # clone + patch + install everything into third_party/
#   scripts/setup_deps.sh --no-sync  # clone + patch only (skip uv sync / downloads)
#
# Result (all under third_party/, git-ignored):
#   openpi/   xuningy/openpi @ aa64205 + 1 patch (request-scoped sampling seeds)
#   RoboLab/  NVlabs/RoboLab @ 0aef241 (v0.2.1) + 6 patches (seeded clients, batched infer, env seeds, task discovery)
#   envs/robolab-sim/.venv   Isaac Sim 5.0 / Isaac Lab 2.2 + RoboLab, pinned by envs/robolab-sim/uv.lock
#   openpi/.venv             OpenPI policy env (OpenPI's own uv.lock)
# and prints the wsp.env lines to use.
#
# Requirements: git, git-lfs, uv, ffmpeg, NVIDIA driver >= 535 (the sim env uses CUDA 11.8 torch wheels so it
# also works on 535-series drivers), ~60 GB disk (RoboLab LFS assets ~10 GB, checkpoint 12 GB, workspace 11 GB).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$repo/third_party"
SYNC=1; [ "${1:-}" = "--no-sync" ] && SYNC=0

OPENPI_URL=https://github.com/xuningy/openpi.git
OPENPI_BASE=aa6420561529593114160d05e5ad155792b272f3
ROBOLAB_URL=https://github.com/NVlabs/RoboLab.git
ROBOLAB_BASE=0aef241fb088ca21bb4ebd24448940ed56620d17
GIT_ID=(-c user.name=weightsmith -c user.email=weightsmith@users.noreply.github.com)

apply() {  # apply() DIR BASE PATCHDIR
  local dir="$1" base="$2" patches="$3"
  git -C "$dir" checkout -q "$base"
  git -C "$dir" checkout -q -B weightsmith
  git "${GIT_ID[@]}" -C "$dir" am -q --committer-date-is-author-date "$patches"/*.patch
  echo "  $(basename "$dir"): $(git -C "$dir" log --oneline -1)"
}

mkdir -p "$TP"
echo "[1/5] OpenPI"
[ -d "$TP/openpi/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --recurse-submodules "$OPENPI_URL" "$TP/openpi"
if [ "$(git -C "$TP/openpi" rev-parse HEAD~1 2>/dev/null)" != "$OPENPI_BASE" ]; then
  apply "$TP/openpi" "$OPENPI_BASE" "$repo/third_party/patches/openpi"
  git -C "$TP/openpi" submodule update -q --init --recursive
fi

echo "[2/5] RoboLab (git-lfs assets, ~10 GB)"
command -v git-lfs >/dev/null || { echo "git-lfs is required (apt install git-lfs; git lfs install)"; exit 1; }
[ -d "$TP/RoboLab/.git" ] || git clone -q "$ROBOLAB_URL" "$TP/RoboLab"
if [ "$(git -C "$TP/RoboLab" rev-parse HEAD~6 2>/dev/null)" != "$ROBOLAB_BASE" ]; then
  apply "$TP/RoboLab" "$ROBOLAB_BASE" "$repo/third_party/patches/robolab"
fi
git -C "$TP/RoboLab" lfs pull

if [ "$SYNC" = 1 ]; then
  echo "[3/5] OpenPI env (uv sync)"
  (cd "$TP/openpi" && GIT_LFS_SKIP_SMUDGE=1 uv sync)
  echo "[4/5] Isaac Sim / RoboLab env (uv sync, Python 3.11; large download)"
  (cd "$repo/envs/robolab-sim" && OMNI_KIT_ACCEPT_EULA=Y uv sync --python 3.11)
  echo "[5/5] pi0.5 DROID joint-position checkpoint (gs://openpi-assets-simeval/pi05_droid_jointpos, 12 GB)"
  export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
  ckpt=$(cd "$TP/openpi" && uv run python -c \
    "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets-simeval/pi05_droid_jointpos'))" | tail -1)
  echo "  checkpoint: $ckpt"
  if [ -f "$repo/third_party/pi05_droid_jointpos.sha256" ]; then
    (cd "$ckpt" && sha256sum --quiet -c "$repo/third_party/pi05_droid_jointpos.sha256") \
      && echo "  checkpoint sha256: OK (identical to the reference run)"
  fi
fi

cat <<EOF

Done. Put these lines in $repo/wsp.env (see .env.example for the rest):
  OPENPI_DIR=$TP/openpi
  OPENPI_PYTHON=$TP/openpi/.venv/bin/python
  ROBOLAB_DIR=$TP/RoboLab
  SIM_PYTHON=$repo/envs/robolab-sim/.venv/bin/python
  CHECKPOINT_DIR=${ckpt:-<OPENPI_DATA_HOME>/openpi-assets-simeval/pi05_droid_jointpos}
  BASE_WORKSPACE=<a path with ~11 GB free>/pi05_droid_jointpos-full
Then: scripts/prepare_workspace.sh   (exports the editable layer-split weights)
      scripts/smoke_test.sh          (end-to-end check on 16 simulated episodes)
EOF
