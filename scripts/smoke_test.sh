#!/usr/bin/env bash
# End-to-end check of a fresh install: sandbox, 16 closed-loop Isaac episodes of the unedited policy
# (StackYellowOnRedTask), and one activation capture on a recorded episode. ~10-15 min on one RTX 3090.
# Uses its own scratch project (projects/smoke) so it never touches real experiments.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo"
export WSP_ROOT="${WSP_ROOT:-$repo/projects/smoke}"
mkdir -p "$WSP_ROOT"
[ -f "$WSP_ROOT/config.json" ] || cp configs/stack-yellow-red.json "$WSP_ROOT/config.json"
owsp() { python3 weightsmith/owsp.py "$@"; }
echo "== status";   owsp status | head -20
echo "== sandbox";  printf 'W = base("action.blocks.05.mlp_1.linear"); print("sandbox ok", W.shape, W.dtype)\n' > /tmp/wsp_smoke.py
owsp py base /tmp/wsp_smoke.py
echo "== 16 closed-loop episodes (unedited pi0.5)"
owsp eval base --n 16
eid=$(python3 -c "import json;print(json.loads(open('$WSP_ROOT/data/episodes/index.jsonl').readline())['eid'])")
echo "== activation capture on $eid"
printf '{"id":"smoke-%s","cand":"base","items":[{"episode":"%s","k":2}],"action_layers":[16],"vlm_layers":[12],"denoise_steps":[0],"image_tokens":true,"action_attention":true}\n' "$(date +%s)" "$eid" > /tmp/wsp_smoke_acts.json
owsp acts < /tmp/wsp_smoke_acts.json
echo "SMOKE TEST PASSED (parity_err above ~0.01 or less is expected bf16 noise)"
