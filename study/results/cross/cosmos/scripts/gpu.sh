#!/usr/bin/env bash
# Run a command on one GPU while holding that GPU's flocks (the dirs in WS/host.json "lock_dirs", in order; on ali-ms
# the coordinator's ~/.cache/weightsmith/locks, then weightsmith's WSP_LOCK_DIR); logs held time to runs/gpu_ledger.jsonl.
# usage: gpu.sh GPU TAG KIND cmd args...
set -u
G=$1; TAG=$2; KIND=$3; shift 3
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mapfile -t LD < <(python3 - "$WS" <<'PY'
import json, sys, pathlib
ws = pathlib.Path(sys.argv[1]); h = pathlib.Path.home()
d = {"machine": "ali-ms", "lock_dirs": [str(h / ".cache/weightsmith/locks"), str(h / "vibes/opus-weight-surgery/state")]}
if (ws / "host.json").exists():
    d.update(json.loads((ws / "host.json").read_text()))
print(d["machine"])
for x in d["lock_dirs"]:
    print(x)
PY
)
MACHINE=${LD[0]}
LEDGER=$WS/runs/gpu_ledger.jsonl; mkdir -p "$WS/runs"
FD=7
for d in "${LD[@]:1}"; do mkdir -p "$d"; eval "exec $FD>>\"$d/gpu$G.lock\""; flock -x $FD; FD=$((FD+1)); done
T0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$G TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1} PYTHONUNBUFFERED=1 "$@"
RC=$?
T1=$(date +%s.%N)
python3 -c "import json; t0,t1=float('$T0'),float('$T1'); print(json.dumps({'tag':'$TAG','machine':'$MACHINE','gpu':$G,'start':t0,'end':t1,'gpu_s':round(t1-t0,1),'kind':'$KIND','rc':$RC}))" >> "$LEDGER"
exit $RC
