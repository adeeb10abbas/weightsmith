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

The original study cap remains 2,544 policy episodes. The initial separate
development check allowed 128 additional episodes. Its first cube reference
batch completed 16 raw trials but failed the post-run identity guard. Every
pinned file subsequently matched, and no offending path was recorded, so those
trials remain quarantined rather than certified. The explicit recovery-v3
amendment repeats the unchanged four-condition cube matrix with an audited
worker. Counting the quarantined attempt, the auxiliary cap is now 144 and the
combined cap is 2,688. Recorded-state forward passes and policy-free reset
checks are not counted as robot policy episodes.

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
- `diagnostics/update_agreement.py`: compare the deployed weight changes from
  every fixed source triplet, without loading a policy or using a GPU.
- `diagnostics/local_activation_fit.py`: measure the same deployed updates on
  their own three construction states and the other six recorded source states.
  This is local activation fit with frozen keys, not edited-network behavior.
- `diagnostics/recover_cube_check.py` and `diagnostics/worker_audited.py`:
  preserve the failed auxiliary attempt and run the unchanged matrix with
  complete integrity-check diagnostics.

Outputs and immutable recovery decisions live under
`projects/rss-few-pair-recovery-20261120/`. Original study results remain under
`projects/rss-few-pair-validation-20261120/`. The old top-level stop statuses
are intentionally preserved; current orchestration status is the recovery
directory's `supervisor-status.json` and per-goal `prepare-status.json`.

Every failed attempt is preserved. Technical recoveries are recorded before
continuation. No held-out outcome may be used to tune or reselect a candidate.
ICWM remains paused.

## Reading the scientific results

The central test is whether three selected source pairs produce a persistent
original-request repair that works on changed arrangements. Report all three
triplets separately for every registered goal. A large gain from only one
triplet would establish source sensitivity, not repeatability. The shared base
cohort and shared layouts are not independent observations repeated three times.

Three pairs is the construction budget, not the acquisition budget. Temporal
qualification attempted 32 pairs (64 episodes), yielding 20 eligible pairs;
cube qualification attempted 16 pairs (32 episodes), yielding 13. Each goal
contributes its first nine eligible pairs to three fixed, disjoint triplets.
Every edit uses one fixed clarification; these are not three different language
corrections. Each goal also has its own edited checkpoint, so cross-goal success
would support portability of the recipe, not composition of multiple repairs.

Competing-goal results must include both net success and fixed/broken pairs.
Equal aggregate success does not demonstrate preservation. In the historical
N3 reverse-order result, 9/16 before and after contained five fixed and five
broken nominally paired trials. Initial images differed, and the paired
layout interval on the net change was -37.5 to +37.5 percentage points.

Compare every edit against the shared embedding baseline and its source-matched
residual baseline. Keep unfamiliar wording separate. The additional development
checks and weight-agreement diagnostics must not be pooled into held-out evidence
or used to choose the best source group. After the complete frozen comparison,
use its outcome to define the next bounded study on fresh test arrangements;
retain unsuccessful goals and all technical attempts in the accounting.

The completed local-fit diagnostic provides a specific hypothesis for follow-up.
On source states excluded from each edit, temporal corrections reduce target
squared error by 31.1%, 17.0%, and 14.4%; cube corrections reduce it by 3.0% and
6.7%, while the third increases it by 8.0%. These are activation errors, not
success rates. Strong fit to the pooled mean (relative error about 0.05--0.06)
therefore does not establish strong fit to individual states. If the complete
robot comparison also shows source sensitivity, a subsequent separately frozen
study can test whether averaging the construction states is the bottleneck.
This observation must not change or select any candidate in the ongoing study.
