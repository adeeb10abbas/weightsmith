#!/usr/bin/env bash
# Second Cosmos lane on ali-ms GPU0 (port 8502, GPU0 flocks), started once Track B has released GPU0.
# 1) determinism check: replay 5 requests recorded on GPU1 through the compiled server path on GPU0;
# 2) take the GPU0 jobs out of the GPU1 queue (they sit at its end as a fallback);
# 3) run the GPU0 pool. Every paired comparison (base vs edit of one task and slot group) runs on one GPU.
cd ~/cross/cosmos
PY=cosmos-framework/.venv/bin/python
RF="Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
O=runs/q/T8-base-s30000.obs/g1875o0
TORCHDYNAMO_DISABLE=0 scripts/gpu.sh 0 replay-gpu0 check $PY scripts/replay_check.py "$RF" offline/replay_gpu0_vs_gpu1.json \
  $O/obs_r0_e0.npz:0 $O/obs_r0_e0.npz:5 $O/obs_r0_e1.npz:3 $O/obs_r0_e2.npz:10 $O/obs_r0_e3.npz:15
python3 - <<PYEOF
import json
g0 = {j["tag"] for j in json.load(open("jobs/queue_gpu0.json"))}
q = json.load(open("jobs/queue.json"))
json.dump([j for j in q if j["tag"] not in g0], open("jobs/queue.tmp.json", "w"), indent=1)
PYEOF
mv jobs/queue.tmp.json jobs/queue.json
echo "GPU0 jobs removed from the GPU1 queue; starting the GPU0 pool"
exec python3 scripts/pool.py --jobs $HOME/cross/cosmos/jobs/queue_gpu0.json --out $HOME/cross/cosmos/runs/q --envs 8 \
  --render default --gpu 0 --port 8502
