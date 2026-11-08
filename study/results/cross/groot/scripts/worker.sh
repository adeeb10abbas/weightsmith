#!/usr/bin/env bash
# One GPU lane: runs jobs from a shared, editable queue file in order, one at a time. Usage: worker.sh GPU PORT
# Queue lines (re-read before every job, so jobs can be added or reordered while the worker runs):
#   TAG|eval|OUTSUB|TASK|SLOT_START|N|PROMPT(or -)|EDIT(or -)|EXTRA_FLAGS     -> pool_groot.py (takes the GPU flocks itself)
#   TAG|cmd|SHELL COMMAND                                                    -> locked.sh (GPU flocks held for the command)
#   TAG|after:DEP1,DEP2|...  (same as above, run only once every DEP is done with rc 0; a gate can be opened by hand:
#                            echo 0 > jobs/done/GATE)  e.g.  ed-x|after:wsp-t1,ok-t1|eval|...
# A job is claimed atomically (mkdir jobs/claims/TAG) so several lanes can share one queue; jobs/done/TAG gets the rc.
# Stop a lane after its current job: touch jobs/STOP.gGPU. Launch detached:
#   setsid nohup bash src/scripts/worker.sh 0 8600 > jobs/worker-g0.log 2>&1 < /dev/null &
set -u
GPU="$1"; PORT="$2"
G=~/cross/groot; Q=$G/jobs/queue.txt; S=$G/src/scripts
mkdir -p $G/jobs/claims $G/jobs/done $G/logs
cd $G
echo "$(date '+%F %T') worker gpu=$GPU port=$PORT queue=$Q"
idle=0
while [[ ! -f $G/jobs/STOP.g$GPU ]]; do
  picked=""
  while IFS= read -r line; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    tag="${line%%|*}"; rest="${line#*|}"
    [[ -e $G/jobs/claims/$tag ]] && continue
    if [[ "$rest" == after:* ]]; then
      deps="${rest%%|*}"; deps="${deps#after:}"; rest="${rest#*|}"; ok=1
      for d in ${deps//,/ }; do [[ -f $G/jobs/done/$d && "$(cut -d' ' -f1 $G/jobs/done/$d)" == 0 ]] || ok=0; done
      [[ $ok == 1 ]] || continue
    fi
    if mkdir $G/jobs/claims/$tag 2>/dev/null; then picked="$tag"; job="$rest"; break; fi
  done < "$Q"
  if [[ -z "$picked" ]]; then
    (( idle++ % 20 == 0 )) && echo "$(date '+%F %T') idle"
    sleep 30; continue
  fi
  idle=0
  kind="${job%%|*}"; body="${job#*|}"
  echo "$(date '+%F %T') start $picked ($kind) on gpu$GPU"
  t0=$(date +%s)
  if [[ "$kind" == eval ]]; then
    IFS='|' read -r sub task s0 n prompt edit extra <<< "$body"
    out=$G/runs/$sub; mkdir -p $out
    if [[ -f $out/$picked.json ]]; then rc=0; echo "  exists: $out/$picked.json"
    else
      args=(--task "$task" --tag "$picked" --slot-start "$s0" --n "$n" --out "$out" --gpu "$GPU" --port "$PORT")
      [[ "$prompt" != "-" ]] && args+=(--prompt "$prompt")
      [[ "$edit" != "-" ]] && args+=(--edit "$edit")
      # shellcheck disable=SC2206
      [[ -n "${extra// }" ]] && args+=($extra)
      python3 $S/pool_groot.py "${args[@]}" >> $out/$picked.out 2>&1; rc=$?
      echo "  $(tail -1 $out/$picked.out | cut -c1-400)"
    fi
  else
    bash $S/locked.sh "$GPU" $G/logs/$picked.log bash -c "$body"; rc=$?
  fi
  echo "$rc $(( $(date +%s) - t0 ))s gpu$GPU $(date '+%F %T')" > $G/jobs/done/$picked
  echo "$(date '+%F %T') done  $picked rc=$rc $(( $(date +%s) - t0 ))s"
done
echo "$(date '+%F %T') stop file found; exiting"
