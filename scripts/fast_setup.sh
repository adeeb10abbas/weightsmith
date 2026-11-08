#!/usr/bin/env bash
# Fresh GPU host -> ready weightsmith host, with every large download running at the same time.
# Same result as scripts/setup_deps.sh + prepare_workspace.sh, in roughly a third of the time.
#
#   scripts/fast_setup.sh             # system packages (if root), network check, clone + patch,
#                                     # parallel downloads, wsp.env, weight export
#   FORCE=1 scripts/fast_setup.sh     # continue even if the network check finds a slow link
#   SKIP_WORKSPACE=1 ...              # stop before the ~11 GB weight export
#
# What runs at once: RoboLab LFS assets (~10 GB), the OpenPI env, the Isaac Sim env (~15 GB of wheels)
# and the 12 GB checkpoint (ranged, parallel GCS fetch). The weight export starts as soon as the OpenPI env
# and the checkpoint are in. Logs: logs/fast_setup/<step>.log. On a host with ~1 Gbps down this takes about
# 25-35 min; most of the time is installing wheels, not downloading them.
#
# The network check matters: a slow host (vast.ai lists "Inet Down") turns 30 minutes into 2 hours, and
# nothing in this script can fix that. Rent hosts with >= 1000 Mbps down.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo"
LOG="$repo/logs/fast_setup"; mkdir -p "$LOG"
t0=$(date +%s)
stamp() { printf '[%4ds] %s\n' "$(( $(date +%s) - t0 ))" "$*"; }
die() { stamp "FAILED: $*"; exit 1; }

# 0. system packages and uv
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  stamp "apt: git git-lfs ffmpeg bubblewrap python3-numpy zstd"
  (export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq \
     git git-lfs ffmpeg bubblewrap python3-numpy zstd curl ca-certificates) >"$LOG/apt.log" 2>&1 \
    || die "apt-get (see $LOG/apt.log)"
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >"$LOG/uv.log" 2>&1 || die "uv install"; }
git lfs install --skip-repo >/dev/null

# 1. network check: 8 ranged reads of a checkpoint object for 8 s
probe=https://storage.googleapis.com/openpi-assets-simeval/pi05_droid_jointpos/params/ocdbt.process_0/d/769c49cccda8deef27abb289e728e90d
mbps=$( { for i in 0 1 2 3 4 5 6 7; do
            curl -s -o /dev/null --max-time 8 -r $((i * 268435456))-$((i * 268435456 + 268435455)) \
                 -w '%{size_download}\n' "$probe" &
          done; wait; } | awk '{s += $1} END {printf "%d", s / 8 / 1e6}')
stamp "network: ~${mbps} MB/s down (8 parallel streams)"
if [ "${mbps:-0}" -lt 40 ] && [ "${FORCE:-0}" != 1 ]; then
  echo "  This host downloads at ~${mbps} MB/s. Setup fetches ~40 GB: about $(( 40000 / (mbps > 0 ? mbps : 1) / 60 )) min of download alone."
  echo "  Rent a host with >= 1000 Mbps 'Inet Down' instead, or rerun with FORCE=1."
  exit 3
fi

# 2. clone + patch (seconds; the LFS assets come in step 3)
SKIP_LFS=1 scripts/setup_deps.sh --no-sync >"$LOG/clone.log" 2>&1 || die "clone/patch (see $LOG/clone.log)"
stamp "clone + patch done"

# 3. every download at once
export UV_CONCURRENT_DOWNLOADS=64 UV_HTTP_TIMEOUT=300 OMNI_KIT_ACCEPT_EULA=Y
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
CKPT="$OPENPI_DATA_HOME/openpi-assets-simeval/pi05_droid_jointpos"
step() {  # step NAME CMD...: run in the background, log to $LOG/NAME.log, record the outcome
  local name=$1; shift
  ( if "$@" >"$LOG/$name.log" 2>&1; then stamp "$name done"; else stamp "$name FAILED (see $LOG/$name.log)"; touch "$LOG/$name.failed"; fi ) &
}
rm -f "$LOG"/*.failed
step lfs        git -C third_party/RoboLab -c lfs.concurrenttransfers=32 lfs pull
step openpi-env bash -c 'cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync'; p_openpi=$!
step sim-env    bash -c 'cd envs/robolab-sim && uv sync --python 3.11'
step checkpoint python3 scripts/gcs_fetch.py gs://openpi-assets-simeval/pi05_droid_jointpos "$OPENPI_DATA_HOME"; p_ckpt=$!
stamp "downloading: lfs, openpi-env, sim-env, checkpoint (tail -f $LOG/*.log)"

# 4. wsp.env, then the weight export as soon as its two inputs (OpenPI env, checkpoint) are in
[ -f wsp.env ] || cp .env.example wsp.env
grep -q '^CHECKPOINT_DIR=' wsp.env || scripts/setup_deps.sh --print-env | sed "s#^CHECKPOINT_DIR=.*#CHECKPOINT_DIR=$CKPT#" >> wsp.env
wait "$p_openpi" "$p_ckpt"
if [ -e "$LOG/checkpoint.failed" ] || [ -e "$LOG/openpi-env.failed" ]; then
  wait; die "openpi-env or checkpoint (see $LOG/)"
fi
(cd "$CKPT" && sha256sum --quiet -c "$repo/third_party/pi05_droid_jointpos.sha256") \
  && stamp "checkpoint sha256: OK" || die "checkpoint sha256 mismatch"
if [ "${SKIP_WORKSPACE:-0}" != 1 ]; then
  step workspace scripts/prepare_workspace.sh
fi
wait
ls "$LOG"/*.failed >/dev/null 2>&1 && die "$(basename -s .failed "$LOG"/*.failed | tr '\n' ' ')(logs in $LOG/)"
stamp "ready. Next: scripts/smoke_test.sh (16 episodes; the first run also compiles shaders)"
