# Fresh closed-form comparison

This additive pipeline uses the same frozen development observation manifest as
`../lora/train_lora.py`. It has not produced new policy results merely by passing
its CPU tests. Only `capture` launches a GPU worker; coordinate it with the study
orchestrator. Existing source recordings and workspaces are read-only.

## Interface

From the repository root:

```sh
python3 study/iros2027/v2/rss_repair_comparison/edit/pipeline.py prepare \
  --spec /absolute/path/shared-development.json --output /absolute/path/prepared
python3 study/iros2027/v2/rss_repair_comparison/edit/pipeline.py capture \
  --prepared /absolute/path/prepared --lora-dataset /absolute/path/lora-dataset --gpu 0
OPENBLAS_NUM_THREADS=8 python3 study/iros2027/v2/rss_repair_comparison/edit/pipeline.py solve \
  --prepared /absolute/path/prepared --lora-dataset /absolute/path/lora-dataset \
  --output /absolute/path/closed-form
```

The pool receives `/absolute/path/closed-form/workspace`. The sparse workspace
contains nine merged, exactly bf16-representable MLP output tensors and a link to
the unchanged base. `solve.json` records source/capture hashes, numerical fit,
preservation shift, rank, merged changes, and replay/teacher parity.

The shared JSON has `cohort_id`, `split: "development"`, optional global
`original_prompt`, `teacher_prompt`, and `recorded_prompt`, and `items` containing:

```json
{
  "item_id": "temporal-train-000",
  "obs_path": "/absolute/path/development/episode/obs.npz",
  "decision_index": 4,
  "split": "train",
  "role": "repair",
  "original_prompt": "original instruction",
  "teacher_prompt": "clearer instruction",
  "recorded_prompt": "clearer instruction"
}
```

Every item is `train` or `validation`; these splits must use disjoint source
episodes. Preservation items require `role: "preservation"` and identical
original/teacher prompts. Each recorded seed must be nonnegative. The preparation
records exact raw sensor/state, seed, prompt, and source-file hashes. Missing
recorded-prompt attribution cannot establish replay parity.

## Fixed primary method

The primary comparison pools training repair decisions globally into the same
32 image-cell keys (two cameras, sixteen cells each) in VLM layers 8–16. Capturing
layer 7 supplies the first residual-increment target. Targets are clearer-minus-
original residual increments, as implemented by `weightsmith.distill`.
Every training preservation decision contributes its 32 unpooled keys with zero
targets. Preservation weight is 1 and these keys are scaled by
`1 / sqrt(number_of_preservation_decisions)`: the preservation squared-loss
penalty averages over decisions, instead of growing with the number of examples.
Individual keys are retained to prevent cancellation across different states.
The absolute ridge is fixed at `1e-3 * sum(K_repair**2) / len(K_repair)`, using
only the 32 repair keys, as in the prior SGDW preservation solve. Adding or
duplicating preservation decisions therefore does not change regularization.
Only train decisions enter the fit. The receipt records this
normalization and both raw and per-decision-RMS preservation output shifts.

The solve retains numerical rank up to 32 before bf16 merging; the primary
single-prompt, global-pooling fit already has rank at most 32 because preservation
targets are zero. The rounding can increase algebraic rank, as for merged LoRA.
`--pooling phase` is available for later, explicitly separate ablations: it requires
phase labels, stacks groups, and caps total rank at 32 while reporting discarded
energy. It is **not** the primary protocol.

`prepare_shared.py` reconstructs each selected request's live batch membership
from all completed environment recordings at the same simulation step. It stores
the active count, power-of-two server bucket, member identities, and hashes of the
full source batch and its receipts. These additional development observations
provide numerical replay context; they do not become training examples.

LoRA preparation certifies exact live-pool replay using the original inference
bucket, seed, instruction, and input, with a fixed `1e-6` maximum action error
after casting to the recorded action dtype. Before any capture, this pipeline
checks that certificate against the same input/context hashes and independently
recomputes its errors from saved replay actions and the original recording.
`solve.json.replay_parity` refers exclusively to this exact-pool gate.

`capture` uses the existing `weightsmith.acts_worker`, with identical singleton
inputs and seeds for original and teacher wording. BF16 batch/graph kernels can
make singleton replay differ from live-bucket actions; these differences are
reported in `singleton-drift.json` and never mislabeled as exact replay. Native
singleton LoRA teacher versus singleton instrumented teacher remains a separate,
fixed `1e-2` gate in physical action units. No acceptance tolerance was inflated.
Per-item exactness and maximum errors are recorded for both comparisons. `parity`
can rerun the CPU-only cross-method check:

```sh
python3 study/iros2027/v2/rss_repair_comparison/edit/pipeline.py parity \
  --prepared /absolute/path/prepared --lora-dataset /absolute/path/lora-dataset
```

Both `capture` and `solve` require `--lora-dataset`. A solve cannot proceed without
both the exact live-pool certificate and the cross-method singleton teacher gate.

CPU contract checks:

```sh
OPENBLAS_NUM_THREADS=2 python3 -m unittest discover \
  -s study/iros2027/v2/rss_repair_comparison/edit -p 'test_*.py' -v
```
