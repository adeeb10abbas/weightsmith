#!/bin/bash
# Local: run the analyses on the GPU host and copy every result JSON into results/.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
H=ali-ms
P='~/vibes/opus-weight-surgery/projects'
S="$P/where-to-edit/scripts"
ho="$P/where-to-edit/heldout/where-94016-64"
r() { ssh -o BatchMode=yes $H "$1"; }
src=$(r "test -f $ho/results.json && echo results.json || echo partial.json")
r "python3 $S/where_analyze.py $ho/$src" > "$here/results/heldout_analysis.json"
r "python3 $S/where_analyze.py $ho/$src --slots 94016-94031,94048-94063" > "$here/results/heldout_analysis_run0.json"
r "python3 $S/where_stages.py $ho" > "$here/results/heldout_stages.json"
r "python3 $S/where_behaviour.py $ho" > "$here/results/heldout_behaviour.json"
r "cat $ho/plan.json" > "$here/results/heldout_plan.json"
r "cat $ho/$src" > "$here/results/heldout_$src"
for t in glasses:where-glasses/heldout/where-tx-94016-32 left:where-left/heldout/where-tx16-94016-32 cube:where-cube/heldout/where-tx16-94016-32; do
  k=${t%%:*}; d="$P/${t#*:}"
  if r "test -f $d/plan.json"; then
    s=$(r "test -f $d/results.json && echo results.json || (test -f $d/partial.json && echo partial.json || echo none)")
    if [ "$s" != none ]; then
      r "python3 $S/where_analyze.py $d/$s" > "$here/results/transfer_${k}_analysis.json" || true
      r "python3 $S/where_behaviour.py $d" > "$here/results/transfer_${k}_behaviour.json" || true
      r "cat $d/plan.json" > "$here/results/transfer_${k}_plan.json"
      r "cat $d/$s" > "$here/results/transfer_${k}_$s"
    fi
  fi
done
r "python3 $S/gpu_time.py" > "$here/results/gpu_time.json"
echo collected; ls "$here/results"
