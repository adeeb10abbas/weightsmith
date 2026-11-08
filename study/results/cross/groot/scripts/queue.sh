#!/usr/bin/env bash
# Run a queue of GR00T evaluations one after another; each line of the queue file is
#   TAG|TASK|SLOT_START|N|PROMPT(or -)|EDIT(or -)|EXTRA_FLAGS
# Each evaluation takes and releases GPU0's flocks itself (pool_groot.py), so other jobs can run in the gaps.
# Lines already done (results JSON present) are skipped, so the queue can be re-launched safely.
set -u
Q="$1"; OUT="$2"
cd ~/cross/groot
while IFS='|' read -r tag task s0 n prompt edit extra; do
  [[ -z "${tag// }" || "$tag" == \#* ]] && continue
  if [[ -f "$OUT/$tag.json" ]]; then echo "skip $tag"; continue; fi
  args=(--task "$task" --tag "$tag" --slot-start "$s0" --n "$n" --out "$OUT")
  [[ "$prompt" != "-" ]] && args+=(--prompt "$prompt")
  [[ "$edit" != "-" ]] && args+=(--edit "$edit")
  # shellcheck disable=SC2206
  [[ -n "${extra// }" ]] && args+=($extra)
  echo "$(date +%T) start $tag"
  python3 src/scripts/pool_groot.py "${args[@]}" >> "$OUT/$tag.out" 2>&1
  echo "$(date +%T) done  $tag $(tail -1 "$OUT/$tag.out" | cut -c1-300)"
done < "$Q"
echo "$(date +%T) queue finished"
