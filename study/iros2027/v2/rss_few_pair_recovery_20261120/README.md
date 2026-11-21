# Continued few-pair validation

The user authorized resuming the stopped study, testing the new edits, and
continuing the investigation. This directory adds recovery orchestration while
retaining the original source data, runtime files, frozen schedules, and failed
attempts. It does not change success predicates or select favorable source groups.

## Recovery decisions

1. **Independent preparation.** Qualified temporal and cube-behind goals can fit
   their edits and perform development selection while mustard source setup is
   diagnosed. The complete held-out matrix still waits for all registered goal
   dispositions and all required development selections.
2. **Batch-matched reset checks.** The original mustard preflight reused one
   simulator across three layout groups. Actual policy batches each use a fresh
   simulator. `scene/isolated_preflight.py` checks every original pose twice in
   the corresponding fresh-process lifecycle. The failed preflight is retained;
   no geometry, scoring, or tolerance changes are allowed by this recovery.
3. **Punctuation-safe baseline.** The old instruction boundary parser rejected
   prompts ending in a period because the tokenizer merges the period and
   separator comma. The additive `baselines_v2` runtime locates the stable
   State-colon anchor and excludes its preceding separator. Host and compiled
   masks are checked for all 12 registered prompts. Existing temporal selection
   remains valid only after exact native action equivalence is demonstrated for
   all 12 original baseline candidates. No original runtime file is changed.
4. **Additional development check.** A separate fixed study evaluates the base
   and all three new edits on the existing development layouts for each of the
   two initially qualified goals: 128 additional policy episodes maximum.
   These results are exploratory, are reported separately, and do not select
   or tune the edits or change the confirmatory baseline grid.

The original study cap remains 2,544 policy episodes. Including the separate
development check, the combined cap is 2,672. Recorded-state forward passes
and policy-free reset checks are not counted as robot policy episodes.

## Entry points

- `prepare_goal.py`: the already-running original temporal development path.
- `prepare_goal_v2.py`: source qualification, fitting, baseline construction,
  reset checks, and development selection for recovered goals.
- `collect_mustard.py`: acquire original mustard source pairs after the exact
  original layouts pass the batch-matched reset certificate.
- `diagnose_edits.py`: the separately frozen development comparison of all
  three source groups and the unchanged reference.
- `evaluate_recovered.py` and `worker_recovered.py`: route new evaluation to
  the certified baseline runtime; retain the original experimental design.
- `continue.py`: verify readiness, freeze all held-out schedules, execute,
  and analyze the complete comparison.
- `watch_recovery.py`: read-only monitoring and messages to the existing
  conversation, reflecting the user's authorization to continue the work.

Outputs and immutable recovery decisions live under
`projects/rss-few-pair-recovery-20261120/`. Original study results remain under
`projects/rss-few-pair-validation-20261120/`. The old top-level stop statuses
are intentionally preserved; current orchestration status is the recovery
directory's `supervisor-status.json` and per-goal `prepare-status.json`.

Every failed attempt is preserved. Technical recoveries are recorded before
continuation. No held-out outcome may be used to tune or reselect a candidate.
ICWM remains paused.
