#!/usr/bin/env bash
# End-to-end check of a fresh install, in three steps: the weight sandbox (`owsp py`), 16 closed-loop Isaac
# episodes of the unedited policy (StackYellowOnRedTask), and one activation capture (`owsp acts`) on a
# recorded episode. About 10-15 min on one RTX 3090; the first run also compiles and caches the policy, and
# every GPU step first waits for the GPU lock if another job holds it.
# Uses its own scratch project (projects/smoke) so it never touches real experiments.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo"
export WSP_ROOT="${WSP_ROOT:-$repo/projects/smoke}"
mkdir -p "$WSP_ROOT"
[ -f "$WSP_ROOT/config.json" ] || cp configs/stack-yellow-red.json "$WSP_ROOT/config.json"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
owsp() { python3 weightsmith/owsp.py "$@"; }  # owsp.py itself reads `py` scripts and `acts` specs from stdin
fail() { echo "SMOKE TEST FAILED: $*"; exit 1; }

echo "== status"
owsp status | head -20

echo "== sandbox (owsp py)"
printf 'W = base("action.blocks.05.mlp_1.linear"); print("sandbox ok", W.shape, W.dtype)\n' > "$tmp/smoke.py"
out="$(owsp py base < "$tmp/smoke.py")"
echo "$out"
grep -q "^sandbox ok" <<<"$out" || fail "the weight sandbox did not run the script (needs bwrap and numpy for /usr/bin/python3)"

echo "== 16 closed-loop episodes of the unedited policy (owsp eval)"
owsp eval base --n 16 | tee "$tmp/eval.out"
grep -Eq "^\[eval base fp=base n=[1-9][0-9]* success=" "$tmp/eval.out" \
  || fail "no episode finished; see the newest pool-* folder under logs/ (sim-*.log, server-*.log)"

eid="$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['eid'])" "$WSP_ROOT/data/episodes/index.jsonl")"
echo "== activation capture on $eid (owsp acts)"
printf '{"id":"smoke-%s","cand":"base","items":[{"episode":"%s","k":2}],"action_layers":[16],"vlm_layers":[12],"denoise_steps":[0],"image_tokens":true,"action_attention":true}\n' \
  "$(date +%s)" "$eid" > "$tmp/acts.json"
owsp acts < "$tmp/acts.json"

echo "SMOKE TEST PASSED. A parity_err of about 0.01 or less above is normal bf16 noise; expect roughly 3-10/16"
echo "successes (the reference box stacks the yellow block in about 40% of episodes)."
