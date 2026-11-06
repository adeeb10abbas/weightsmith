#!/bin/bash
# Operator only: score one editor's submission on 64 held-out slots (default 91008-91071) with heldout.py, detached.
# usage: score_ctrl.sh KEY CAND [SLOT_START]   e.g. score_ctrl.sh ctb p2   (SLOT_START must be a multiple of 16; default 91008)
set -e
k="$1"; c="$2"; s0="${3:-91008}"
[ -n "$k" ] && [ -n "$c" ] || { echo "usage: score_ctrl.sh KEY CAND"; exit 2; }
cd ~/weightsmith
set -a; . ./wsp.env; set +a
export WSP_ROOT=$HOME/vibes/opus-weight-surgery/projects/ctrl-$k
[ -d "$WSP_ROOT/cands/$c" ] || { echo "no candidate $c in ctrl-$k"; exit 3; }
nohup setsid python3 weightsmith/heldout.py --cand "$c" --slot-start "$s0" --n 64 > "$HOME/weightsmith/logs/ctl/ho-$k.out" 2>&1 < /dev/null &
echo "launched held-out scoring: ctrl-$k / $c (pid $!)"
