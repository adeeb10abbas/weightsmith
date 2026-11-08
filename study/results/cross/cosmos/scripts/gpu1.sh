#!/usr/bin/env bash
# Run a command on GPU1 while holding the GPU1 flocks (the coordinator's lock dir first, then weightsmith's
# WSP_LOCK_DIR, the order every user of these locks takes them); logs the held time to runs/gpu_ledger.jsonl.
# usage: gpu1.sh TAG KIND cmd args...
set -u
TAG=$1; KIND=$2; shift 2
LEDGER=$HOME/cross/cosmos/runs/gpu_ledger.jsonl
exec 8>>"$HOME/.cache/weightsmith/locks/gpu1.lock"; flock -x 8
exec 9>>"$HOME/vibes/opus-weight-surgery/state/gpu1.lock"; flock -x 9
T0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=1 TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1} PYTHONUNBUFFERED=1 "$@"
RC=$?
T1=$(date +%s.%N)
python3 -c "import json,sys; t0,t1=float('$T0'),float('$T1'); print(json.dumps({'tag':'$TAG','gpu':1,'start':t0,'end':t1,'gpu_s':round(t1-t0,1),'kind':'$KIND','rc':$RC}))" >> "$LEDGER"
flock -u 9; flock -u 8
exit $RC
