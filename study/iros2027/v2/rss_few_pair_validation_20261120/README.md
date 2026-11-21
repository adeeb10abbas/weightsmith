# Weightsmith: independent few-pair validation

This additive study tests whether the earlier three-pair repair repeats across
source choices and repair goals, and whether simpler inference corrections
explain its benefit. Completed studies remain unchanged. ICWM stays paused.

## Registered design

The three historically informed goals are:

1. Put the cube in the bowl, then the banana.
2. Place the cube behind the bowl.
3. Place the mustard bottle to the right of the raisin box.

For each goal, acquire fresh original/clarified episode pairs in blocks of 16.
A pair qualifies when the original instruction fails, its clarification succeeds,
and the predetermined state selector identifies a construction state. Stop at
the first complete block containing nine eligible pairs, or at 48 attempted
pairs. Retain failed goal qualification; do not substitute another goal.

Partition the first nine eligible pairs into three disjoint, predetermined
triplets. Each Weightsmith candidate uses three source states and the same
historical closed-form editing recipe. These are three source selections for
one fixed instruction pair, not nine different linguistic corrections.

All acquisition cost is reported. A three-pair edit does not imply only three
trials were required to find those sources.

## Comparisons

- Unchanged policy, executing the original request.
- Unchanged policy, executing the clarification, as an informative control.
- Three independent Weightsmith edits, one per source triplet.
- One instruction-embedding offset per goal. Its construction is text-only;
  duplicating it across source groups would create false replication.
- Three constant residual fields, using the same source triplets as the edits.
  These intervene after VLM block 16 over the same 32 image cells.

The simpler baselines use a fixed strength grid of 0.5, 1, and 2. Separate
development layouts select strength, with the unchanged policy as a shared
zero-strength fallback. Ties select the smaller strength. All development
candidates count toward cost. Weightsmith has one fixed recipe and no rollout
selection among its three source groups.

The baseline server must pass exact fixed16 native parity, zero-intervention
parity, a clarified-embedding replacement positive control, nonzero-action
smoke checks, and unchanged-checkpoint checks before evaluation. These checks
establish implementation correctness, not behavioral efficacy.

## Held-out tests

Each qualified goal uses 16 fresh arrangements. Each method receives 32
original-request episodes (two noise seeds per arrangement), 16 unfamiliar-
wording episodes, and 16 competing-goal episodes. The clarification control
has only the 32 original-request episodes. Total: 544 held-out episodes per
qualified goal, plus 208 development-selection episodes.

All goals' selected candidates and complete test schedules are sealed before
the first held-out episode. Report each source group and goal separately.
Paired uncertainty uses layouts as clusters. Shared baselines and test layouts
are not three independent test cohorts. Changed arrangements retain the same
objects and scene; this study does not establish new-object or hardware transfer.

Maximum new policy episodes: 288 source acquisition + 624 development selection
+ 1,632 held-out = **2,544**. Policy-free scene checks and recorded-state
forward-pass checks are counted separately.

## Execution and evidence

All output lives under `projects/rss-few-pair-validation-20261120/`.

- `collect.py prepare`: create immutable layout and collection manifests.
- `collect.py run`: execute source qualification on the two shared GPU locks.
- `supervise.py`: wait for collection, fit all qualified goals, select the
  simpler baselines, freeze every test schedule, execute, and verify reports.
- `analyze.py --goal GOAL`: verify a completed goal's comparison.
- `analyze.py --aggregate`: preserve all three goal outcomes and cost accounting.

Partial batches and failed technical checks stop the pipeline. They are retained
for inspection and are never silently retried. A recovery must be separately
recorded. The simulator and native policy runtimes are reused without changing
historical cohorts or global inference code.

The main frozen protocol is `protocol.json`. The adaptation directory records
an explicit recipe clarification: the historical 40 prompt rows consist of
three opening template positions plus a fixed 37-position State window, which
can include matching padding. This corrects the initial prose description
while preserving the demonstrated numerical recipe and 72 total key rows.
The archived three-pair edit must be reproduced exactly before fitting new
edits.

Status files distinguish scene checks, source episodes, fitting, development
selection, held-out execution, and verified completion. An active service alone
is not a completed experiment.
