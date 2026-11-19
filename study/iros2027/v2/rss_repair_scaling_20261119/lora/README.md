# Fresh full-trajectory self-demonstration LoRA

This baseline addresses whether fine-tuning on successful executions is enough to
internalize the language correction. Each training episode is a successful native
clarified-policy rollout from a selected development pair. Every recorded action
chunk is included; student inputs use the original instruction. There are no human
action labels. The source count is **1, 3, 6, or 12 distinct episode pairs**, all
for the same temporal repair goal. The overall collector is capped at 48 attempted
pairs. All attempted collection episodes must be counted in adaptation cost.

The full-trajectory arm receives more observations and actions than the
closed-form edit's one selected failure decision per pair. It therefore directly
tests the successful-trajectory alternative, rather than claiming identical
supervision information. The optimizer acts on the same VLM MLP output
projections (layers 0–16, rank 72). This is a module-matched LoRA comparison, not a
test of unrestricted fine-tuning or action-expert adaptation.

Targets are the current native policy's recorded physical action predictions
along successful executions. They are not measured joint trajectories or archived
instrumented samples. Source observations and actions are checked against a fresh
replay of the exact fixed-16 server and actual batch members, with maximum physical
action discrepancy at most **1e-6**. No tolerance is relaxed after a failure.
Preparation uses the serving process's compilation cache, memory allocation
settings, and warmup. A preserved first-block diagnostic found a 0.00289 maximum
discrepancy under a fresh compilation setup; matching the serving setup recovered
bit-identical actions for all 16 requests in that diagnostic. The full source
replay certificate remains mandatory for every accepted training label.
Absolute joint targets are made relative to the recorded joint state and passed
through the checkpoint's frozen input normalization. The gripper coordinate is
preserved and 24 padding coordinates remain zero. The native output transform
must reconstruct the recorded physical actions within **1e-5**.

Both learning rates, **1e-4 and 1e-3**, use batch size one, seed 17, AdamW with zero
weight decay, gradient clipping at one, and native flow-matching loss without
image augmentation. The number of steps is `max(200, 2 * action_chunk_count)`.
Shuffled complete passes ensure every training chunk is used at least twice.
Adapters start with random A and zero B; the starting BF16 weights must equal the
base exactly. All other weights are frozen. Final weights are merged and exported
with no additional inference module. Candidate selection uses 16 independent
development rollout slots per rate and budget; ties select the lower rate.
Fitting losses never select a checkpoint or serve as behavioral evidence.

Run only after the source collector has released the chosen GPU:

```bash
python3 study/iros2027/v2/rss_repair_scaling_20261119/lora/run.py \
  --source-manifest projects/rss-repair-scaling-20261119/sources/block-01/source_manifest.json \
  --output projects/rss-repair-scaling-20261119/lora-adaptation --gpu 0
```

Use the actual exported source manifest path. The runner executes available
budgets serially and writes `candidates.json` with source identity, available and
completed budgets, and each candidate's workspace and artifact checksums. An
optional `--budget N` runs one available budget. Completed stages may be resumed
under the exact same identity; incomplete stages remain preserved and are not
automatically retried. The script acquires the shared GPU lock for every model
operation and writes explicit running, failed, or complete status.

For the closed-form edit's independent replay certificate, `fresh_lora.py prepare
--arm matched_decision` also accepts a `correction.json` spec. This optional
preparation uses fresh singleton-native teacher actions at the selected failure
states to match the activation-capture regime, while checking source actions with
the fixed-16 server. It does not train an additional LoRA arm unless separately
requested. Singleton-versus-live drift and instrumented-versus-native parity are
different checks; they must not be conflated.

CPU validation:

```bash
python3 -m unittest discover -s study/iros2027/v2/rss_repair_scaling_20261119/lora -p 'test_*.py' -v
```

The tests check action normalization, complete successful-trajectory coverage,
development-only provenance boundaries, correction-spec compatibility, and full
training-data coverage. GPU preparation, smoke training, and closed-loop success
remain separate completion gates.
