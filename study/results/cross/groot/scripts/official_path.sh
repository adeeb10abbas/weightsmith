#!/usr/bin/env bash
# Validation run of the unmodified official path: Isaac-GR00T's run_gr00t_server.py (--use-sim-policy-wrapper)
# + RoboLab's policies/gr00t/run.py (serial per-env client, open-loop horizon 8), default rendering preset.
# Usage: official_path.sh TASK NUM_ENVS OUTNAME [extra RoboLab flags...]
set -u
TASK="$1"; NE="$2"; NAME="$3"; shift 3
G=~/cross/groot
CKPT=~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-DROID/snapshots/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5
R=~/vibes/weight-space-policy-lab/robolab_checkout
SIM=~/vibes/weight-space-policy-lab/robolab_env/.venv/bin/python
if [[ "${OFFICIAL_LOCKED:-0}" != 1 ]]; then   # set OFFICIAL_LOCKED=1 when run under locked.sh (it holds the flocks)
  exec 8>~/.cache/weightsmith/locks/gpu0.lock; flock 8
  exec 9>~/vibes/opus-weight-surgery/state/gpu0.lock; flock 9
fi
cd $G/Isaac-GR00T
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path $CKPT \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --device cuda --host 127.0.0.1 --port 8601 \
  --use-sim-policy-wrapper > $G/logs/$NAME.server.log 2>&1 8>&- 9>&- &
SP=$!
until grep -q "listening on" $G/logs/$NAME.server.log; do sleep 2; kill -0 $SP || exit 1; done
cd $R
OMNI_KIT_ACCEPT_EULA=Y VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json $SIM policies/gr00t/run.py \
  --headless --remote-host 127.0.0.1 --remote-port 8601 --task "$TASK" --num-envs "$NE" --num-runs 1 \
  --open-loop-horizon 8 --instruction-type default --video-mode none --output-folder-name "$NAME" \
  --device cuda:0 --/rtx/verifyDriverVersion/enabled=false --/renderer/multiGpu/enabled=false \
  --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0 "$@" > $G/logs/$NAME.sim.log 2>&1
kill -INT $SP; sleep 5; kill -9 $SP 2>/dev/null
python3 -c "
import json; rs=[json.loads(l) for l in open('$R/output/$NAME/episode_results.jsonl') if l.strip()]
print('$NAME', sum(r['success'] for r in rs), '/', len(rs))"
