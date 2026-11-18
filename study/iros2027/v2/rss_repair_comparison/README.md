# Frozen clarification-memory versus weight-adaptation comparison

This additive study compares one temporal repair (cube then banana into a bowl)
and one spatial repair (mustard right of a raisin box), both on π0.5 DROID.
Historical results, checkpoints, and original simulator files are unchanged.
`protocol.json` declares the input selection, adaptation settings, exact test
wordings, conditions, layouts, seeds, and interpretation limits.

The primary comparison has four conditions: base policy, saved clarification
with a conservative text-only grammar router, closed-form image-cell editing,
and teacher-action LoRA. The router is hand-engineered and covers the declared
relations; this is not a blind or learned language-generalization benchmark.
Its exact-cache counterpart is derived only from episodes with identical
effective policy inputs and is explicitly labeled reused evidence.

Both weight methods receive the same development observations, instructions,
sampling seeds, and identity-preservation examples. The edit receives internal
activation targets; LoRA receives sampled teacher actions. Both modify the same
nine MLP output projections, layers 8–16, with nominal rank 32. This matches
module footprint and factor capacity, not information or optimization budgets.
BF16 rounding can increase the algebraic rank of the merged difference.

## Evidence boundaries

- New run artifacts live under `projects/rss-repair-comparison-20261117/`, which
  is ignored by Git. Completed batch receipts, raw observations, videos, metric
  traces, source snapshots, hashes, and failed attempts are retained there.
- Development uses four layouts per case. D01–03 supply training states; D04
  supplies development validation. State selection is fixed by layout, replica,
  and trajectory fraction, without selecting successful episodes.
- Evaluation uses eight additional layouts and two policy-noise streams per
  layout. Five cells per case and four conditions produce 640 policy episodes.
  The original repair request is seen; two target paraphrases are withheld
  from weight adaptation. Primary preservation reuses known competent
  development wording (reverse order and stacking). The additional cube-only
  and left-placement stress tests use unseen wording of development-seen intents.
- Only complete receipts count as policy evidence. Unit tests and a real-model
  LoRA gradient compatibility check are engineering evidence, not robot success.
- Identical initial poses, robot states, and noise seeds do not guarantee
  pixel-identical rendering or trajectory replay. Repeat controls measure this
  nuisance; no observations are dropped for an unfavorable outcome.
- Sixteen episodes per cell cannot establish a tight preservation margin.
  Competence-qualified alternatives support retention comparisons. A baseline
  floor supports only a stress-test interpretation.
- The full-horizon cube-only cell measures sustained compliance for 60 seconds,
  including after the cube first reaches the bowl. This policy has no explicit
  stop action. The earlier success-terminated cube-only development diagnostic
  is retained but replaced for shared adaptation data and final evaluation.
- Initial development comprises 256 episodes. A bounded LoRA tuning amendment,
  recorded before any heldout access, preserves the initial rate of 1e-3 and
  adds 3e-5 and 1e-4, each for 200 steps on the same data. Balanced validation
  action MSE nominates one new rate. That nominee and the original fit receive
  16 original-request and 16 primary-preservation development trials each.
  Maximum total strict success selects the candidate; ties use action MSE,
  then lower learning rate. This adds 128 development episodes. All fits,
  feedback, and costs remain in the record; the methods have different budgets.
- LoRA selection and heldout comparison use a fixed sixteen-row inference
  batch, including after other episodes finish. Earlier development used
  dynamic power-of-two batching and is reported separately. Exact replay
  checks reproduce the original live batch; singleton drift is diagnostic.

## Components

- `runtime/`: separately registered tasks, deterministic layout specifications,
  reset qualification, strict and stock-compatible scoring, raw telemetry.
- `worker.py`: bounded batch execution with receipts and source/candidate seals.
  New outputs are routed locally because the historical output symlink targets
  an unavailable external disk. The old symlink is untouched.
- `rewrites.py`: explicit canonicalization and abstention; never receives an
  evaluator goal identifier.
- `prepare_shared.py`: produces one shared adaptation specification from
  complete development records, with an explicit full-horizon replacement.
- `adapt_case.py`: runs teacher generation, activation capture, cross-method
  teacher parity, closed-form fitting, and LoRA training; it performs no test
  rollouts and refuses to overwrite an interrupted adaptation.
- `edit/`, `lora/`: method implementations, tests, and usage details.
- `lora/sweep.py`, `lora/select_candidate.py`: bounded tuning and receipt-checked
  development selection; they preserve initial and rejected fitted candidates.
- `make_eval_plan.py`: requires both fitted cases and passing parity before
  freezing all four candidate artifacts and generating the test plans.
- `analysis/`: receipt-checked paired summaries, layout-level exploratory
  uncertainty, competing-goal outcomes, and explicitly reused exact-cache rows.

## Execution

Run from the repository root. The machine paths are read from `wsp.env`; do not
publish credentials or raw environment configuration. A typical adaptation is:

```bash
python3 study/iros2027/v2/rss_repair_comparison/adapt_case.py \
  --run-root projects/rss-repair-comparison-20261117 --case spatial --gpu 1
```

After both adaptations, the declared LoRA sweep and development selection,
and qualification checks finish:

```bash
python3 study/iros2027/v2/rss_repair_comparison/make_eval_plan.py \
  --run-root projects/rss-repair-comparison-20261117 \
  --output projects/rss-repair-comparison-20261117/heldout-frozen-v1
```

Pass each resulting plan to `worker.py` with a new output root and its allocated
GPU. All GPU commands use the shared machine lock. Do not edit frozen runtime,
plan, or candidate files during a cohort. A partial batch stops the cohort for
inspection; it is not silently erased or converted into a policy failure.

The manuscript is maintained separately at `/home/ali/weightsmith-rss2027-paper`.
Historical results there are not automatically replaced by unfinished jobs.
