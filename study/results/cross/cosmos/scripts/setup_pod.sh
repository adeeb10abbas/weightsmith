#!/usr/bin/env bash
# Cosmos3 Edge serving env on a fresh GPU pod, next to a weightsmith checkout set up by scripts/fast_setup.sh
# (which provides RoboLab, its LFS assets and the Isaac Sim env). Run on the pod from WS (= this script's parent dir):
#   [SIM_ENV=robolab-sim-cu128] [LOCK_DIR=/workspace/locks] bash scripts/setup_pod.sh MACHINE_LABEL /workspace/weightsmith
# Installs cosmos-framework @ cf5d68c (the commit the "Same Goal, Different Words" study pinned for Edge) with
# `uv sync --all-extras --group=cu128-train --group=policy-server` (torch 2.10 + cu128), pre-downloads the
# checkpoint (nvidia/Cosmos3-Edge-Policy-DROID, not gated) and writes WS/host.json for pool.py and gpu.sh.
set -euo pipefail
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MACHINE=${1:?machine label}; WSP=${2:-/workspace/weightsmith}
cd "$WS"; mkdir -p logs runs edits caps offline jobs videos
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d cosmos-framework/.git ] || git clone -q https://github.com/NVIDIA/cosmos-framework.git cosmos-framework
git -C cosmos-framework checkout -q cf5d68c
(cd cosmos-framework && UV_CONCURRENT_DOWNLOADS=64 UV_HTTP_TIMEOUT=300 \
   uv sync --all-extras --group=cu128-train --group=policy-server) > logs/uv_sync.log 2>&1
cosmos-framework/.venv/bin/python -c "import torch, openpi_client; print('torch', torch.__version__, torch.cuda.is_available())"
ICD=$(ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null | head -1 || true)
python3 - "$MACHINE" "$WSP" "${ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}" "${SIM_ENV:-robolab-sim}" \
    "${LOCK_DIR:-$HOME/.cache/weightsmith/locks}" <<'PY'
import json, sys, pathlib
m, wsp, icd, simenv, lockdir = sys.argv[1:6]
d = {"machine": m, "robolab": f"{wsp}/third_party/RoboLab", "sim_py": f"{wsp}/envs/{simenv}/.venv/bin/python",
     "lock_dirs": [lockdir], "vk_icd": icd}
pathlib.Path("host.json").write_text(json.dumps(d, indent=1))
print(json.dumps(d))
PY
echo SETUP_POD_DONE
