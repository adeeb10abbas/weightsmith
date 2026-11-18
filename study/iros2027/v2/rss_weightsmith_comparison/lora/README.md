# LoRA on the successful temporal edit's six source decisions

This additive baseline uses exactly the original `c2-multi` observations and its
clarified-instruction action chunks. All six decisions train the adapter. None is
relabeled as validation. Candidate selection belongs to separate fresh development
rollouts; the fitting diagnostics never select an adapter or checkpoint.

The immutable source pairs are `base.base.s20001/k5`, `s20007/k5`, `s20009/k4`,
`s20022/k4`, `s20023/k4`, and `s20040/k4`. Capture rows 0–5 use the original request;
rows 12–17 use the clarification, with matching observation identity, step, state,
and sampling seed. Their files and the preparation code are hash-sealed first.

Unlike the newer comparison's preparation path, the historical archive does not
provide modern exact-live-batch certificates. The recorded instrumented/live
maximum action discrepancy is 0.07994968093243243. We preserve that limitation and
measure current native original/clarified replay against the archived captures and
original live actions. These diagnostics are not an exact live replay claim.

Targets are the archived teacher's **physical** 15×8 action chunks, not newly
sampled substitutes. The first seven absolute joint targets are converted to
deltas, normalized using the frozen checkpoint, and padded to 32 coordinates.
The remaining 24 normalized coordinates are zero padding, not unavailable latent
teacher outputs. Native output transformation must reconstruct every physical
target within **1e-5**; this tolerance cannot be changed in response to a failure.

The optimization matches the full successful edit's output-projection support
and algebraic rank bound: VLM layers 0–16, rank 72, 22,560,768 factor parameters.
Three rates are declared in advance: 0.00003, 0.0001, and 0.001. Each uses the same
seed 17, 200 AdamW steps, batch size one, zero weight decay, global norm clipping
at one, random A with standard deviation .01, and zero B. Native flow-matching
loss uses fresh noise/time draws and no image augmentation. All other weights
remain frozen. The final merged BF16 checkpoint is exported without an inference
adapter. This matches source observations and editable module/rank scope, not
historical search effort or a universal fine-tuning optimum.

A one-step rank-72 gradient/memory smoke must complete on GPU 0 before the three
fits. Every GPU command uses the shared `GpuLock`. Preparation, smoke, and fitting
outputs must be new directories; failures remain in place and are not retried
automatically. No held-out rollouts are launched by this code.

```bash
JAX_PLATFORMS=cpu /home/ali/openpi-robolab/.venv/bin/python -m unittest discover \
  -s study/iros2027/v2/rss_weightsmith_comparison/lora -p 'test_*.py' -v
python3 study/iros2027/v2/rss_weightsmith_comparison/lora/historical_lora.py seal \
  --output projects/rss-weightsmith-comparison-20261118/lora-source.json
/home/ali/openpi-robolab/.venv/bin/python study/iros2027/v2/rss_weightsmith_comparison/lora/historical_lora.py prepare \
  --source-seal projects/rss-weightsmith-comparison-20261118/lora-source.json \
  --output projects/rss-weightsmith-comparison-20261118/lora-preparation --gpu 0
```

Use `smoke --dataset ... --output NEW_DIR --gpu 0` for the one-step check, then
`train --dataset ... --output NEW_DIR --gpu 0 --lr RATE` for each declared fit.
Each fit writes settings, progress, training diagnostics, factors, a merged
workspace, and an artifact seal. The comparison controller receives a manifest
only after all three fits complete, then evaluates 16 independent development
slots per candidate. Its frozen rule selects by original-request success and
breaks ties using the smaller learning rate.
