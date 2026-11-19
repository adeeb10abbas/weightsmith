# Modest correction scaling and full-trajectory fine-tuning

This additive study addresses three objections to the Weightsmith paper: how
many correction examples construct a weight update, whether it transfers to
changed object arrangements, and whether fine-tuning on the successful policy
trajectory provides the same benefit. It retains the previously successful
temporal construction and tests **1, 3, 6, and 12 distinct correction episodes**.
It does not run a 100-example experiment or sequential weight composition.

The task is placing the cube and then the banana in the bowl. The original
request is “Put the cube then the banana in the bowl”; the clarification is “Put
the cube in the bowl then put the banana in the bowl.” This is **one repair goal
and one instruction pair**, regardless of the number of source episodes.

## What counts as one correction

A correction is one fresh nominal scene/noise episode pair in which the original
request fails the strict temporal predicate and the clarification succeeds.
The failed original must also contain an early-release decision under the
pre-existing deterministic gripper rule. We take its first such decision and use
one observation from that original episode. This heuristic identifies the
construction state; gripper state alone is not proof of a particular object
contact or of the physical cause of failure.

Pairs are ordered by their predeclared slot. Budgets use nested prefixes of the
first eligible pairs, without selecting a subset from edited-policy outcomes.
The collection stops after the first complete block of 16 pairs that supplies
12 eligible corrections, or after **48 pairs / 96 policy episodes**. Every
attempt and rejection reason is retained. If the cap yields fewer than 12
eligible corrections, the unavailable budget remains unavailable; the rule is
not relaxed and the source set is not replenished indefinitely.

The human supplies one clarification string. The experiment varies the number
of distinct observed episode pairs, not the number of human wording changes.
Nominal scene and policy-noise pairing does not make separately rendered
trajectories pixel-identical.

## Methods and supervision

Weightsmith applies the historical pooled construction: average the original
MLP keys and clarified-minus-original residual-increment targets over the
chosen source observations; solve one 72-row ridge system at each VLM MLP
down-projection in layers 0–16; merge with multiplier one and BF16 rounding.
The 72 rows comprise 40 aligned template/state positions and 32 image cells.
Relative ridge is fixed at 0.001. Neither update strength nor source count is
selected using held-out rollouts. The action expert and action heads remain
unchanged. This is gradient-free fitting using observations and a clarification,
not data-free adaptation.

The primary comparator is rank-72 LoRA on the same VLM projections. For each
selected pair it uses **every predicted native physical action chunk from the
successful clarified rollout**, while presenting the original request to the
student. These are policy self-demonstrations along a successful trajectory,
not measured joint trajectories or human demonstrations. LoRA therefore sees
more state/action supervision than the one-decision closed-form construction;
the shared budget is source episode pairs, and actual action-chunk counts are
reported separately.

The two fixed learning rates are 0.0001 and 0.001. Each fit takes at least 200
steps and two complete shuffled passes over all available action chunks. Each
rate receives 16 separate development episodes. The rate with more strict
successes is selected, with ties resolved toward the lower rate. Final BF16
weights are exported; there is no inference-time adapter. All available budgets
must finish before the held-out schedule is frozen.

The `matched_decision` data-preparation mode supplies an independent native
teacher and replay certificate for closed-form capture. It does not imply that
a matched-decision LoRA policy was trained or evaluated in this main study.

## Numerical and physical checks

Source collection uses the existing temporal RoboLab scoring/runtime and a
fixed 16-row inference kernel. The active set can shrink, but padding keeps the
kernel batch at 16. Replay reconstructs the actual live members and their exact
recorded observations and seeds. Source outputs must reproduce within 1e-6;
physical-action normalization must round-trip within 1e-5. Instrumented
singleton teacher capture must agree with the independent native singleton
teacher within 0.01. Singleton-versus-live differences are reported separately
and cannot be excused by increasing a failed threshold.

Every simulator reset validates written and settled poses, stability, and an
initially false target predicate. Changed layouts additionally require a
separate policy-free reset preflight before the held-out policy comparison.
Each policy batch has one attempt; partial batches and numerical failures stop
the corresponding stage and preserve their artifacts.

## Held-out questions

Sixteen independently generated changed layouts are fixed before policy
evaluation. In half, cube and banana swap their horizontal positions. Every
layout also shifts the bowl horizontally by 4–6 cm, plus the declared small
position jitter. These are changed arrangements in the same scene, not a new
scene, robot, or task population. Development rate selection uses a separate
set of 16 small-perturbation layouts.

Each Weightsmith and selected LoRA candidate receives:

- 32 original-request episodes: 16 changed layouts and two policy-noise seeds.
- 16 episodes under one unseen phrasing: “Place the cube in the bowl, followed
  by the banana.”
- 16 reverse-order competing-goal episodes: banana before cube.

The unchanged policy receives the same 64 trials. The unchanged policy with
the clarification receives 32 original-task trials to measure whether those
changed arrangements remain feasible for the known skill. The clarification
is a diagnostic control; it does not modify the policy.

At all four available source budgets this is **608 held-out episodes**, plus
up to **128 LoRA-selection episodes** and **96 source-collection episodes**.
Exact executed counts, including failed acquisition, are reported from
receipts. Repair, transfer to the single unseen phrasing, and reverse-goal
retention are reported separately. Nominal paired differences use layout-
cluster intervals; they do not establish generalization across independent
repair goals. No sequential-edit or broad language-generalization claim is
part of this design.

## Files and stage interfaces

Code lives in this directory. Large observations, captures, workspaces, and
receipts live under ignored `projects/rss-repair-scaling-20261119/`.

- `protocol.json` fixes the scientific choices and budgets.
- `collect.py prepare` seals the collection runtime and writes
  `design/{collection,development,changed}-layouts.json`.
- `collect.py run` runs paired source blocks on the two GPUs. Each block owns
  separate `collection/block-NN/{original,teacher}/` plans and receipts.
- `prepare_sources.py export` verifies completed source blocks and creates
  `sources/block-NN/source_manifest.json`. Its `specs[str(N)]` maps to
  `nN/correction.json` and `nN/full_trajectory.json`. Each spec binds the parent
  source-manifest hash and exact live-batch context; every source item is
  training data. There is no per-frame validation split.
- `build_edits.py` prepares fresh native replay certificates, captures both
  prompts, fits all available nested budgets, and writes
  `edits/candidates.json` only after all candidates finish.
- `lora/run.py` fits the declared full-trajectory adapters and writes
  `lora-adaptation/candidates.json` as budgets complete. Selection rejects an
  intermediate registry until every available source budget is complete.
- `evaluate.py` seals development selection and then the complete held-out
  schedule, including candidate, runtime, input, and scoring identities.
- `analyze.py` verifies receipts before reporting policy outcomes.

The first source block is intentionally bounded so a fresh replay check can
complete before collection is extended. The orchestration layer owns GPU
allocation; do not start independent training while both collection workers
are running.

```bash
python3 study/iros2027/v2/rss_repair_scaling_20261119/collect.py prepare
python3 study/iros2027/v2/rss_repair_scaling_20261119/collect.py run --gpus 0 1 --max-blocks 1
```

After the native replay gate passes, a supervisor can extend the same frozen
collection to at most three blocks, stopping when the target count is reached.
Use the final exported source manifest for both construction methods:

```bash
OPENBLAS_NUM_THREADS=4 python3 study/iros2027/v2/rss_repair_scaling_20261119/build_edits.py \
  --source-manifest /absolute/path/to/final/source_manifest.json --gpu 0
python3 study/iros2027/v2/rss_repair_scaling_20261119/lora/run.py \
  --source-manifest /absolute/path/to/final/source_manifest.json --gpu 1 \
  --output /home/ali/weightsmith-iros2027/projects/rss-repair-scaling-20261119/lora-adaptation
```

## Preserved startup failure and current status

The first launch on November 19 failed during startup because the new wrapper
imported configuration before setting separate GPU-worker ports. Both servers
attempted to bind port 8370. No episode results, observation recordings, or
completed batch receipts were produced. The full 434-file attempt was retained
under `projects/rss-repair-scaling-20261119-startup-failed/`; the new root's
`startup-attempts.json` records the inventory hashes and repair. The wrapper now
sets ports before importing the runtime, and a regression test checks this
ordering. A fresh collection seal binds the repair. No policy outcome was
discarded or retried in that startup correction.

The first source block completed 32 development policy episodes: original
success 1/16, clarified success 11/16, and seven eligible episode corrections.
These are qualification results, not edited-policy outcomes. The two initial
native replay checks then passed bit-for-bit (20 successful-trajectory action
chunks and the selected failure decision), and all 32 evaluation/development
layouts passed two policy-free reset cycles. Collection has continued under
the same fixed rule toward 12 eligible pairs.

The first replay preparation differed from the serving process by up to
0.002890 in physical action coordinates. A saved comparison reproduced the
same 16 recorded requests bit-for-bit after matching the complete serving
setup (compilation cache, allocation, loading, and warmup). Every preparation
entry point now uses that setup. The failed attempt and diagnostic variants
are preserved; labels and the 1e-6 threshold are unchanged. This identifies a
configuration bundle that restores replay, not the individual causal effect
of cache or warmup.

The durable `weightsmith-scaling-study-20261119.service` sequences acquisition,
adaptation, development selection, frozen evaluation, and the verified report.
Its stages stop on failures, preserve partial artifacts, and have no automatic
policy retry. The launch audit is in `LAUNCH_AUDIT.json`; current state is in
`projects/rss-repair-scaling-20261119/supervisor-status.json`.
**No new edited-policy result is claimed by this launch audit.**
