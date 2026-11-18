# Frozen historical-artifact recovery diagnostic

This additive development diagnostic compares six already-existing policies on
the original temporal request. It makes no new edit and selects no candidate.
Its 96 episodes are eight new small-perturbation layouts with two policy-noise
streams each, shared across six sequential arms on GPU 0. GPU 1 is not used.

The historical full (`v2mi-f`) and image-only (`abl-imgl8`) weights are reused
exactly, alongside the recent closed-form edit and development-selected LoRA.
The unedited policy is run with the original and clarified instructions.
These candidates had different construction data and selection budgets. This
is a recovery diagnostic, not an equal-budget comparison or an independent
replication of edit construction. It does not replace any prior result.

The unchanged comparison runtime uses fixed-16 inference, the 60-second temporal
task, and its strict stable-placement/order predicate as the primary outcome.
Its existing stock predicate is recorded as a secondary diagnostic on the same
trajectories; it is not a new stock-runtime replication. All simulator attempts
are retained; one attempt per arm is allowed and an incomplete batch stops work.

The original successful recipe used failure-local observations; the recent
recipe changed observation selection and added preservation simultaneously.
This comparison cannot attribute differences to either design choice alone.

Preparation (CPU only; refuses an existing artifact directory):

```sh
python3 study/iros2027/v2/rss_historical_recovery/prepare.py
```

The worker reuses the original comparison worker with the previously validated
single-attempt pool. It freezes all runtime sources, candidates, touched base
tensors and checkpoint files. Base tensor equality is also enforced by the
existing policy loader. No historical workspace is written.

```sh
python3 study/iros2027/v2/rss_historical_recovery/worker.py --plan projects/rss-historical-recovery-20261118/freeze/temporal-plan.json --root projects/rss-historical-recovery-20261118/evaluation --gpu 0
```

Run only after the prepared plan has been reviewed. Never retry an incomplete
arm automatically or switch predicates in response to its outcomes.
