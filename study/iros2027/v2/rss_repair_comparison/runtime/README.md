# Additive RSS repair runtime

This runtime evaluates new, separately named development and held-out cohorts. It does
not modify the measured RoboLab runtime, old scenes, old tasks, historical SGDW
layouts, or any frozen score. All six requested goals share the two scene assets.
The policy remains pi0.5; comparison conditions and prompt routing belong to the
parent worker, outside this directory.

## Invocation

Use `runtime/run_batched.py` in place of the stock `weightsmith.pool.sim_cmd` launcher.
Keep the usual pool arguments, simulation working directory, environment count and
sampling seed. Set:

```text
WSP_TASK_DIRS=<absolute path to this runtime directory>
WSP_START_LAYOUTS=<absolute path to layouts/{case}_{split}.json>
RSS_OUTPUT_DIR=<absolute local output root used by the pool's output reader>
```

The existing `OWSP_OBS_DIR` receives receipts and scorer metrics. Optional
`WSP_START_LAYOUT_IDS` is a comma-separated list of unique IDs. Layouts cycle by
`(group * num_envs + env_id) % num_layouts`, independently of lane assignment.
For 16 environments, all four development layouts occur four times; all eight
held-out layouts occur twice. Sampling seeds, not these repeated physical layouts,
distinguish those replications. Repeated layouts are not independent repairs.

| Goal | Task class | Layout file |
|---|---|---|
| Cube before banana | `RSSCubeThenBananaTask` | `temporal_development.json` or `temporal_heldout.json` |
| Banana before cube | `RSSBananaThenCubeTask` | Same temporal layouts |
| Cube only, banana unchanged | `RSSCubeOnlyTask` | Same temporal layouts |
| Cube only through the full 60 s budget | `RSSCubeOnlyFullHorizonTask` | Same temporal layouts |
| Mustard right of raisin | `RSSMustardRightTask` | `spatial_development.json` or `spatial_heldout.json` |
| Mustard left of raisin | `RSSMustardLeftTask` | Same spatial layouts |
| Mustard on raisin | `RSSMustardOnRaisinTask` | Same spatial layouts |

RoboLab resolves an external file to its first alphabetical Task subclass. The six
single-class entry files prevent a shared implementation module from registering
the wrong class. A CPU test exercises the actual installed resolver.

The installed checkout's `output` symlink points to an unavailable external mount.
`output_redirect.py` makes a process-local copy of the stock evaluation function,
changing exactly its output-directory assignment. It requires `RSS_OUTPUT_DIR` and
rejects upstream source drift. RoboLab's package and asset directories are unchanged.

## Physical semantics

All placements require contact above the installed predicate's 0.1 N threshold and
gripper detachment. A stable placement requires root-position speed below 0.02 m/s
for one continuous second. At 15 Hz this is 16 consecutive samples: the duration is
the interval between first and last sample, not the number of samples times dt.
Reference displacement is the maximum Euclidean displacement throughout the
trajectory, not merely displacement at its endpoint.

**Temporal primary.** Both objects must currently be inside the bowl, contact it,
and be detached. Their first qualifying placement ticks must satisfy a strict
inequality in the requested order. An early wrong placement is irreversible even
if that object is removed and placed again. Both final objects must be stationary
for one second; the bowl must never move more than 0.02 m. The horizon is 60 s.
Cube-only requires a stable detached cube in the bowl while the banana is outside
and has never moved more than 0.02 m; the same bowl constraint applies.

`RSSCubeOnlyTask` is an early-stop development diagnostic only. It can stop before
a controller subsequently reaches for the banana, so it cannot establish sustained
preservation of the competing goal. The separately named
`RSSCubeOnlyFullHorizonTask` evaluates every tick and permits success only at tick
900 (60 s). Its final score requires the banana to have remained within 2 cm for
the entire budget. Use this task for the replacement development comparison and
held-out cube-only preservation. The installed termination manager keeps success
and timeout flags separate; RoboLab records simultaneous timeout plus success as
success. A test executes those installed methods directly and checks that a policy
that first completes cube-only then moves banana loses full-horizon credit.

**Stock temporal audit.** The installed `RubiksCubeThenBananaTask` already calls
`objects_placed_in_container_in_order`, which latches first placements and requires
both objects currently placed. It does measure order. Its default is non-strict
`<=`, so simultaneous first observations pass, and it has no one-second stability
or fixed-bowl requirement. `stock_final` and `stock_ever` reproduce those logical
conditions on the *new runtime's trajectory*. They are secondary diagnostics,
not a reproduction of a stock run that would terminate earlier. For cube-only,
the secondary is explicitly a placement component diagnostic; there is no
corresponding stock sequence predicate.

**Spatial primary.** Mustard must satisfy the requested directional relation in
the **robot base frame**, with non-mirrored 45-degree cones. Right means negative
robot Y; left means positive robot Y. The code calls RoboLab's transformed-pose
predicate, rather than treating world coordinates as robot coordinates. The
installed robot starts at the origin with identity orientation, and its actual
base pose is included in each reset receipt. Directional placements require
mustard-table contact. The on-top competitor requires upward support force from
the raisin box and the mustard centroid inside the raisin footprint (0.01 m
tolerance), plus mustard-raisin contact. Mustard must have moved at least 0.02 m;
the raisin box must never move more than 0.02 m. All require detachment and
one-second stationary placement. The horizon is 40 s.

The historical SGDW right-placement predicate already checks robot-frame relation,
table contact, detachment and one-second stability. It does **not** enforce minimum
mustard movement or preservation of the reference position. The new secondary
directional flags omit just those two additional constraints. The on-top secondary
is a component diagnostic, not a claim to reproduce a historical right task.

## Layout provenance and reset checks

Four development IDs `T-D01`–`T-D04` / `S-D01`–`S-D04` and eight held-out IDs
`T-H01`–`T-H08` / `S-H01`–`S-H08` are generated deterministically from separate
fixed seeds, with no policy feedback. The asset heights and orientations use
previously established scene-compatible values; every XY arrangement is new.
This supports small in-distribution pose variation, not a new-scene claim.

Temporal starts put both objects outside the bowl. Spatial starts separate mustard
and raisin mainly along robot X, so neither left nor right is already true.
`generate_layouts.py` can reproduce the committed JSON exactly and refuses to
replace an existing changed file. The frozen JSON hash is recorded in every receipt.

After each stock reset, the new layout injector writes object poses, zeroes
velocities, settles physics for 15 control ticks while the robot holds its joints,
and recomputes rendered camera observations. It records intended, written and
settled poses, quaternion, speed, spin, settling drift and robot joint drift.
A batch raises an error before policy evaluation if writes differ by over 2 mm or
1 degree, XY settling displacement exceeds 12 mm, final speed exceeds 0.03 m/s,
late settling drift exceeds 5 mm, or poses/quaternions are invalid. It also rejects
any initial contained/on-top/left/right goal relation. Failed reset receipts are
retained; these are infrastructure failures and must not be counted as policy
failures or repaired by searching for favorable layouts.

The stock runner resets twice at each episode start. Both receipts are retained,
the scorer is recreated after each, and the runtime rejects a different reset count.

## Metrics and interpretation

`layout_r{run}_reset{1,2}.json` preserves both resets; `layout_r{run}.json` duplicates
the final reset for compatibility. `metrics_r{run}.json` records:

- `selected_goal` and the original `environment_results`;
- `strict_final`, `strict_ever`, `stock_final`, `stock_ever`, each mapping all three
  same-scene goal names to a Boolean array in environment order;
- first temporal placement ticks, initial/final poses, maximum displacement,
  final speed and decomposed physical predicates;
- the final layout receipt, IDs, slot indices and layout SHA-256.

The selected strict goal ends the episode once it holds. Other goal scores are
observations on that same trajectory up to that endpoint. In particular, cube-only
can be true during a successful cube-then-banana trajectory; its `strict_ever` flag
is not evidence of inappropriate goal capture. Competing-goal runs must select
their own termination and be judged by their own `strict_final` result. Frozen
environments retain their endpoint states and metrics as other slots continue.

CPU tests cover temporal ordering, simultaneous events, removal and replacement,
detachment and continuous dwell, movement and reference preservation, frozen slots,
reset state, deterministic disjoint layouts, pool mapping, invalid reset rejection,
the actual external task resolver and the narrowly scoped output redirect. These
tests do not establish successful simulator resets or learned-policy performance;
live qualification results belong in the parent cohort records.

## Initial live qualification observation (2026-11-17)

The first v1 attempts produced no policy episodes and were preserved separately:
external-file registration and an unavailable output mount blocked startup. The
six entry wrappers and local output redirect corrected these infrastructure issues.

Both v2 original-prompt batches passed both layout resets on all 16 slots and
advanced into policy execution. Receipts verified identity robot-base orientation,
no initial temporal placement, and no initial spatial left/right/on-top relation.
Maximum total settling displacement was 0.65 mm for temporal objects and 3.04 mm
for the spatial mustard bottle; maximum drift over the last nine settling ticks
was 0.021 mm. These are reset observations only. At this observation, policy
outcomes, camera-frame review and a repeated unchanged-policy batch were pending.

The spatial original-prompt batch subsequently completed: selected strict success
0/16 and secondary stable-right success 0/16. In all 16 episodes mustard moved less
than 0.12 mm, while five episodes displaced the raisin reference more than 2 cm
(maximum 23.4 cm). Initial and final policy camera inputs for environments 0–3
were reviewed and were consistent with the logged layout and the arm approaching
the reference object. This is one development condition; it does not establish a
repair effect. The teacher contrast and repeated unchanged-policy qualification
were still pending at this annotation.

The temporal original-prompt batch completed with 1/16 selected strict successes
and 1/16 secondary stock-order successes. Its successful environment 7 first
placed cube at tick 123 and banana at tick 286, reaching stable strict success at
tick 446. High-resolution sensor frames 140 and 440 were reviewed: cube was in the
bowl before banana, and both were in the bowl at the endpoint. Four original-prompt
trajectories instead finished with stock reverse order; three also met strict
reverse-order stability/reference criteria. These countergoal observations share
the original-prompt trajectories and are not separate competitor trials.

Spatial original/teacher reset receipts had identical IDs, slots and bit-identical
settled object positions. This establishes matched physical starts for that pair;
it does not replace the separately planned unchanged-policy repeat qualification.

The spatial teacher then completed with 11/16 strict successes (12/16 secondary
stable-right successes). A repeated original-prompt batch again yielded 0/16,
with no paired outcome disagreements. Initial physical poses, proprioception and
sampling seeds matched exactly, but rendered images and action chunks differed.
The original condition is at a success floor, so positive-control teacher repeats
were prepared to measure outcome variability where success is possible. Exact
comparison statistics and source hashes are in `qualification-notes.json`.

The full-horizon cube-only qualification ran all 16 episodes to tick 900 and
finished with 0/16 successes, although 13/16 had satisfied cube-only at an earlier
time. Every episode eventually moved the banana more than 2 cm. The early-stop
score therefore cannot support sustained preservation. Since this policy exposes
no explicit stop action and normal task evaluation stops externally at success,
the full-horizon condition also tests instruction compliance after initial
completion. This baseline floor must be reported when interpreting preservation.

The temporal teacher completed with 13/16 strict successes versus the original
1/16, with bit-identical settled initial object positions and the same layout/slot
IDs. Teacher successes occurred on every development layout (3/4, 4/4, 3/4, 3/4).
The initially specified spatial left competitor completed at 0/16 strict success;
none of its episodes moved mustard by 2 cm, and 14/16 moved the reference by more
than 2 cm. Its one secondary-left success was therefore rejected correctly.
These findings motivate the separately declared bounded development competence
screen; they do not authorize selecting a wording using held-out results.

The spatial teacher repeat subsequently yielded 7/16 rather than 11/16, with six
paired outcome changes (five losses and one gain). Initial object poses, IDs,
proprioception and sampling seeds matched exactly; rendered inputs differed
(mean absolute channel difference 0.723/255 exterior and 3.981/255 wrist). This is
material run-to-run variation. Its cause is not isolated by this comparison, and
both attempts must be reported rather than choosing the better score.
