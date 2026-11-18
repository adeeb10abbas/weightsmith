# Action-supervised LoRA comparison

This is a genuine gradient-trained LoRA baseline on the frozen π0.5 policy, using
the same development observations and teacher instructions as the closed-form
repair. It learns to reproduce the clearer-instruction policy's action chunks
under the original instruction. Preservation records use identical original and
teacher prompts. There are no human demonstrations, but there **are** collected
development observations and policy-generated action targets; this is not
data-free fine-tuning.

The primary registered budget is rank 32 on language MLP down projections in
layers 8–16, 200 AdamW steps, learning rate 0.001, zero weight decay, gradient norm
clipping at 1, batch size 1, seed 17. The 5,308,416 trainable factor parameters
start with random A (standard deviation 0.01) and zero B. The update is A B, with
implicit LoRA scaling α/r = 1. Base weights, vision encoder, remaining language
weights, action expert, and action head remain frozen. The objective is OpenPI's
native flow-matching loss on the normalized teacher action chunk. Training uses
fresh flow-matching noise/time draws and no image augmentation. This is a
module-matched LoRA baseline, not an exhaustive search over adaptation modules
or optimization budgets.

The final 200-step state is exported; development validation is diagnostic and
does not select a checkpoint. Changing the budget or selecting among multiple
runs must be recorded explicitly by the study controller before final testing.

## Shared development specification

Use the study's `prepare_shared.py` selector to produce the specification from
completed development receipts. The following excerpt illustrates the instruction
and train/validation fields; it is not sufficient by itself to run preparation,
because the selector also supplies the mandatory live-batch certificate described
below.

```json
{
  "cohort_id": "rss-fresh-temporal",
  "split": "development",
  "original_prompt": "the instruction to repair",
  "teacher_prompt": "the clearer instruction",
  "items": [
    {"item_id": "D01-repair-0", "obs_path": "/absolute/development/episode-D01/obs.npz", "decision_index": 0, "split": "train", "role": "repair", "recorded_prompt": "the clearer instruction"},
    {"item_id": "D04-repair-0", "obs_path": "/absolute/development/episode-D04/obs.npz", "decision_index": 0, "split": "validation", "role": "repair", "recorded_prompt": "the clearer instruction"}
  ]
}
```

Each item may override `original_prompt`, `teacher_prompt`, and
`recorded_prompt`. An optional `phase` or other provenance fields are retained.
`role: preservation` requires equal original and teacher instructions. Distinct
instructions may reuse a state; duplicate state/instruction pairs are refused.
Training and validation must use separate episode paths. Held-out paths and
non-development specifications are refused. The study controller owns the
stronger layout/seed/phrasing separation: path validation alone cannot establish
that two renamed files are independent episodes.

Each `obs.npz` needs `base_img`, `wrist_img`, `joint`, `gripper`, `step`, `seed`,
and `actions`, indexed by `decision_index`. Targets use the recorded nonnegative
sampling seed. `recorded_prompt` identifies the instruction used by the source
episode and is checked against its sealed receipt.

The selector supplies `live_batch_size`, `live_batch_bucket`,
`live_batch_members`, `source_env_id`, `source_run_id`, `source_step`,
`live_context`, and `live_context_sha256`. Members are dictionaries containing
`env_id`, `obs_path`, `decision_index`, and `obs_sha256`. The context identifies
the source receipt/episode files and their hashes, every source-batch observation
hash, and the members. Its checksum is SHA-256 of canonical JSON with sorted keys
and compact separators. Preparation independently reconstructs membership from
the decision step in every sealed source-batch episode; incorrect sizes, buckets,
members, prompts, source cases, or hashes are refused before model loading.

Two separate numerical checks apply. Exact live replay uses the existing
`BatchedServer.infer_batch` implementation, repeating the selected source request
at its recorded active batch size and therefore its original padding bucket.
After conversion to the recorded float32 representation, the maximum absolute
physical-action error must be at most **1e-6**. The shared checkpoint is loaded
once for both native teacher sampling and this replay. Singleton native replay
versus live-batch drift is recorded separately and cannot substitute for the
exact replay certificate. The closed-form pipeline independently checks native
versus instrumented singleton teacher actions at the unchanged **0.01** gate
before solving/fitting completes. Both checks must pass in the study controller.

This separation follows a development-only diagnostic: both tested live bucket-16
requests reproduced recorded float32 actions bit-for-bit, while identical inputs
under singleton kernels exhibited BF16 numerical drift. The original stopped
attempt is preserved. No fitting budget or held-out result informed the change.
Every source episode, specification, prepared item, and replay context is hashed.

## Commands

Run from `/home/ali/weightsmith-iros2027` with the installed OpenPI interpreter.
`--gpu` names the physical GPU. Commands acquire the shared `GpuLock` and set
`CUDA_VISIBLE_DEVICES`; an explicitly conflicting CUDA setting is rejected.
Coordinate allocation with the experiment controller before launching.

```bash
/home/ali/openpi-robolab/.venv/bin/python study/iros2027/v2/rss_repair_comparison/lora/train_lora.py prepare --spec /absolute/shared-spec.json --output projects/rss-repair-comparison/CASE/lora-dataset --gpu 1
/home/ali/openpi-robolab/.venv/bin/python study/iros2027/v2/rss_repair_comparison/lora/train_lora.py train --dataset projects/rss-repair-comparison/CASE/lora-dataset --output projects/rss-repair-comparison/CASE/lora-r32-s200 --gpu 1
```

Both output directories must be new. Preparation writes per-item arrays with
`teacher_actions` (normalized, 15×32) and `teacher_actions_physical` (15×8), plus
the transformed original-instruction observation. It also saves
`live_replay_actions_physical`, `recorded_actions_physical`, and
`singleton_recorded_actions_physical` so the replay certificate can be recomputed
without GPU inference. `dataset.json` includes `live_replay_gate`, each exact and
singleton error, context/input hashes, and the number/time of replay forward calls
and repeated/padded examples. Training refuses absent or invalid exact-live
certificates and checks that saved action arrays agree with those certificates.
Replay context is validation only and does not add training observations.
The aggregate
`teacher-actions.npz` provides `actions` (physical), `normalized_actions`,
`item_ids`, and `policy_seeds`, ordered exactly as the input specification. The
closed-form capture pipeline can independently compare its teacher actions
against these arrays, reporting both exact equality and maximum absolute error.
The production `adapt_case.py` controller runs preparation, capture, cross-method
parity, and closed-form solving before invoking LoRA training; the two commands
above are individual stages, not a substitute for that complete study workflow.

Training writes factors, settings, progress, and initial/final validation
flow-matching and action errors. Action comparisons use the fixed recorded seed
and include per-item, per-role, and overall errors. Physical MSE averages mixed
joint/gripper coordinates; it is an action discrepancy, not task success. These
development diagnostics must remain separate from final held-out rollouts.

The `workspace/` output contains only the selected merged down projections and a
symlink to the immutable original base. Export verifies the base matches the
checkpoint exactly and serializes the same BF16 tensors evaluated in JAX. It
passes the existing workspace validator and can be served by the unchanged
policy pool. The adapter has no additional inference cost after merging. BF16
rounding can increase the algebraic rank of the effective dense difference;
rank 32 refers to the trained factors.

## Verification and current evidence

```bash
JAX_PLATFORMS=cpu /home/ali/openpi-robolab/.venv/bin/python -m unittest discover -s study/iros2027/v2/rss_repair_comparison/lora -p 'test_*.py' -v
```

The twelve CPU checks exercise data-boundary rejection, episode separation,
prompt identity, file-integrity checks, reconstructed live-batch membership,
wrong-bucket/source rejection, missing replay certificates, float32 storage
precision, zero-start equivalence, immutable base and frozen nonselected tensors,
nonzero factor gradients, and exact merged-workspace export.

A single synthetic gradient compatibility run on 2026-11-17 used the real frozen
checkpoint, rank 32, and layers 8–16 on physical GPU 1. It completed in 27.2
seconds including loading/compilation, with finite loss 1.27646, gradient norm
0.04585, and peak live JAX allocation 14.00 GB. The process exited and released
the GPU lock. This establishes feasibility on one 24 GB RTX 3090; it is **not**
a trained repair, a policy rollout, or evidence of improved task success.
Its artifact is `projects/rss-repair-comparison/lora-smoke-20261117/smoke.json`.
