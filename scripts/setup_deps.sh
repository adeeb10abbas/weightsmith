#!/usr/bin/env bash
# Reproduce the exact policy + simulator stack used for the weightsmith results.
#
#   scripts/setup_deps.sh              # clone + patch + install everything into third_party/
#   scripts/setup_deps.sh --no-sync    # clone + patch only (skip uv sync / downloads)
#   scripts/setup_deps.sh --check      # only check the system requirements
#   scripts/setup_deps.sh --print-env  # only print the wsp.env lines for this layout:
#                                      #   scripts/setup_deps.sh --print-env >> wsp.env
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
MODE="${1:-}"
case "$MODE" in ''|--no-sync|--check|--print-env) ;; *) echo "usage: $0 [--no-sync|--check|--print-env]"; exit 2 ;; esac
SYNC=1; [ "$MODE" = "--no-sync" ] && SYNC=0
CKPT_DEFAULT="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/openpi-assets-simeval/pi05_droid_jointpos"

print_env() {  # the wsp.env lines for the layout this script creates (checkpoint path: $1)
  printf '%s\n' "OPENPI_DIR=$TP/openpi" "OPENPI_PYTHON=$TP/openpi/.venv/bin/python" "ROBOLAB_DIR=$TP/RoboLab" \
    "SIM_PYTHON=$repo/envs/robolab-sim/.venv/bin/python" "CHECKPOINT_DIR=$1" \
    "BASE_WORKSPACE=$HOME/weightsmith-workspace/pi05_droid_jointpos-full"
}
if [ "$MODE" = "--print-env" ]; then print_env "$CKPT_DEFAULT"; exit 0; fi

check_requirements() {  # report every missing requirement at once; return 1 if any is missing
  local missing=0 c drv
  for c in git git-lfs uv ffmpeg bwrap; do
    if ! command -v "$c" >/dev/null; then echo "  missing: $c"; missing=1; fi
  done
  if ! /usr/bin/python3 -c 'import numpy' 2>/dev/null; then
    echo "  missing: numpy for /usr/bin/python3 (the owsp py sandbox runs it)"
    missing=1
  fi
  if command -v nvidia-smi >/dev/null; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$drv" ] && [ "${drv%%.*}" -lt 535 ]; then echo "  NVIDIA driver $drv is older than 535"; missing=1; fi
  else
    echo "  missing: nvidia-smi (the GPU host needs an NVIDIA driver >= 535)"; missing=1
  fi
  if [ "$missing" = 1 ]; then
    echo "  Ubuntu: sudo apt install git git-lfs ffmpeg bubblewrap python3-numpy && git lfs install"
    echo "  uv: https://docs.astral.sh/uv/getting-started/installation/"
    return 1
  fi
  echo "  requirements: OK"
}
echo "[0/5] system requirements"
if ! check_requirements; then exit 1; fi
if [ "$MODE" = "--check" ]; then exit 0; fi

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
[ -d "$TP/RoboLab/.git" ] || GIT_LFS_SKIP_SMUDGE="${SKIP_LFS:-0}" git clone -q "$ROBOLAB_URL" "$TP/RoboLab"
if [ "$(git -C "$TP/RoboLab" rev-parse HEAD~6 2>/dev/null)" != "$ROBOLAB_BASE" ]; then
  apply "$TP/RoboLab" "$ROBOLAB_BASE" "$repo/third_party/patches/robolab"
fi
# SKIP_LFS=1: leave the ~10 GB of assets for the caller (scripts/fast_setup.sh pulls them in parallel)
[ "${SKIP_LFS:-0}" = 1 ] || git -C "$TP/RoboLab" lfs pull

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

echo
echo "Done. Next, at the repo root:"
echo "  cp .env.example wsp.env                       # if you have no wsp.env yet"
echo "  scripts/setup_deps.sh --print-env >> wsp.env  # appends these lines:"
print_env "${ckpt:-$CKPT_DEFAULT}" | sed 's/^/    /'
echo "  (edit BASE_WORKSPACE if the ~11 GB export should live elsewhere)"
echo "Then: scripts/prepare_workspace.sh   (exports the editable layer-split weights, ~11 GB)"
echo "      scripts/smoke_test.sh          (end-to-end check on 16 simulated episodes)"
