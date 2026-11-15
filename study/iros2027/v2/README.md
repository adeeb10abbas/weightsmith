# IROS 2027 v2 kickoff

This is an additive development cohort based on the user's replacement v2 packet. The four files in
`packet/` are preserved verbatim as the proposed research specification. Embedded agent handoff text is
not treated as blanket authorization to run the full sweep, access sealed tests, or publish a paper.

The repository base is `cf39ab173d52ebb2b29bb8b424b36072cc7124b8` (current `main` at the audit).
All additions are under `study/iros2027/v2/`. Original study records, code and scoring predicates remain
unchanged. Existing local experiment branches and partially completed work have not been modified.

## Running development diagnostics

Following the user's request to continue running, `screen_worker.py` now advances two bounded jobs:
S01 (cube-left) on GPU 0 and S02 (cube-right) on GPU 1. Each uses the exact packet wordings on 32
matched development starts: 64 episodes per case, **128 episodes total**. The four 16-episode batches
follow ABBA order. T01 and S03 are not rerun. No edit or held-out test is scheduled.

These are provisional development diagnostics. Semantic review remains pending and the worker does not
declare a case eligible, tune an edit, or promote any independent-repair claim. The frozen
`spatial-pilot.json` records the robot reference frame, fixed-reference check, hypothesis and budget.

The background services are `weightsmith-iros27-s01-dev` and `weightsmith-iros27-s02-dev`. They survive
the chat turn and stop when their four batches finish, if disk space drops below 10 GiB, if a batch is
incomplete, or after a two-hour maximum. Incomplete evidence is preserved for inspection. Completed
batches have receipts and can be skipped on an explicit restart; the service does not silently rerun
an interrupted batch. Upstream pool attempts are retained in separate log/observation folders.

Live state is in `projects/iros27-spatial-dev/{S01,S02}/status.json`, with an append-only `events.jsonl`
and per-batch outcomes, observations and logs. Each episode records its actual initial object poses,
first camera hashes, policy seed and maximum bowl displacement. The new runtime adds observation-only
telemetry around the original reset/step functions; it leaves original predicates and actions unchanged.
It is a separate instrumented cohort, not a claim of exact replay of the old study.

## What is ready

- `historical-evidence.json` records 307 protected Git blobs and the source revisions and SHA256 hashes
  for the site's Cosmos3, GR00T and shared π0.5 aggregate records. This establishes artifact presence,
  not independent reproduction or verification of every raw rollout or checkpoint.
- `provenance.py` inventories installed source, environments, hardware, and checkpoint files without
  reading episode or held-out records. The full local report is in
  `projects/iros27-ctb-repro/provenance.json`; `host-summary.json` contains its concise findings.
- `plan.py` validates the v2 registry, records all 12 pairs, and groups related goals. It reuses T01 and
  S03 as historical references and proposes at most **640 new development episodes** for the other
  10 pairs. The six candidate new goals are not six established independent failures or repairs.
  T04 is a CTB paraphrase; sequential/order-free cube–banana tasks share a correlation cluster.
- `development-plan.json` is a planning artifact, with **zero launch-ready cases**. It does not dispatch
  formal screens. Every pair awaits semantic/physical review; missing task definitions have an additional
  gate. The two provisional diagnostics above have their own narrower manifest and execution ledger.
- `startup-job.json` records the bounded 16-episode baseline qualification that began before v2 arrived.
  It uses an isolated practice project, stock scene, slots 64000–64015 and unchanged weights. It is
  labeled `replication`; it is neither a new repair nor part of the new-pair screening denominator.
- `practice-config.json` preserves the startup project configuration. `startup-result.json` records
  the completed check: **16/16 episodes finished, 1 success and 15 failures**, with unchanged weights.
  One activation capture passed with maximum absolute replay action error **0.00402**. There are no
  edited-policy outcomes, new repair claims or held-out evaluations. `startup-episodes.jsonl` records
  each outcome, policy seed, observation checksum and first camera-frame hashes; actual initial object
  poses were not captured. Large observations, logs and captures stay in the ignored project.

The new tools pass **15 CPU tests** and lint. All 307 protected files match their recorded Git blobs,
and all four v2 packet files match the attachment byte for byte. The 16-episode rollout took about
8 minutes; 40 such batches would be roughly 5.3 hours on one lane, an initial planning estimate for
the 640-episode ceiling rather than a measured sweep cost. Different tasks and capture work change it.

## Run the read-only tools

From the repository root, using Python 3.10+ with PyYAML and pytest installed:

```bash
python study/iros2027/v2/plan.py --task-source /path/to/RoboLab/robolab/tasks
python study/iros2027/v2/provenance.py --verify-checkpoint > /new/path/host-ledger.json
python -m pytest study/iros2027/v2/tests -q
```

The provenance tool reads the existing `wsp.env` configuration and does not install or change dependencies.
Hash verification reads about 12 GB of checkpoint data. The model identity is the SHA256 of the published
per-file manifest plus successful verification of every entry, not a purported single-file model hash.
The editable workspace's declared hash is recorded separately and its tensor bytes are not reverified.
The source inventory explicitly excludes results, datasets, hidden directories, symlinks and scene assets.

The existing distillation command was exercised **in dry-run mode only**, against the new project. Its
printed held-out command is an upstream example using previously used slots; it is not an approved or
reserved IROS package. No edit construction or held-out evaluation was performed by this kickoff.

## Audit findings that affect the next work

1. **RoboLab provenance:** the installed code has no `.git` directory. Its 295 code files are hashed;
   this identifies the observed source without establishing equivalence to the published patched revision.
   Isaac Sim 5.0.0.0 and Isaac Lab 2.2.0 are installed. Pinning assets and exact source lineage remains open.
2. **Spatial semantics:** installed `RubiksCubeLeftOfBowlTask` uses `frame_of_reference="robot"` and
   `mirrored=False`, rather than the world frame mentioned in the proposed CSV. Preserve the historical
   predicate and write an explicit rubric for the new cohort. Check the reference object remains fixed.
3. **Scene variation:** `main` has no `weightsmith/layouts.py`. The existing `sgdw/v1` worktree contains a
   layout application and pose logging implementation. Audit and reuse suitable parts in a new cohort;
   do not replace the old runtime or equate new policy seeds with novel physical scenes.
4. **New tasks:** T02/T03, B01/B02, S04 and B03/B04 name task definitions that are not yet implemented.
   Temporal tasks require order; binding tasks require an order-free predicate. Neither may silently
   inherit CTB's two-objects-in-one-bowl success condition. Asset placement and reachability need checks.
5. **Recipe budget:** the brief calls for one frozen candidate while the manifest allows two closed-loop
   candidates. Resolve this before constructing edits and log any rejected candidates. No candidate has
   been built or selected here.
6. **Storage:** about 34 GB remained at startup on a 99%-full filesystem. Reuse the installed checkpoint
   and environments. Measure episode storage before committing to the 640-episode screen; do not delete
   historical data or launch large downloads to make room.

## Next bounded work

Begin cohort B with a written semantic rubric and task/predicate implementation for the new sequential
cube-in-bowl then banana-on-plate goal (T02/T03), alongside the distinct order-free goal (B01/B02). Keep
their shared skills explicit in the analysis. Validate terminal success and event annotations with CPU
fixtures and a small development run. Only then screen original versus explicit wording on the declared
matched starts. ABBA means first 16 starts A then B, next 16 starts B then A; every condition must record
the same scene specification, actual initial poses, early camera hashes, and policy-noise seeds.

The bounded diagnostic worker and batch ledger are implemented; a general dispatcher for all remaining
tasks and conditions is still needed. Candidate selection, random controls, scene sampling,
noise recalibration for changed runtime/scenes, sealed operator evaluation, interference plots and the
manuscript are subsequent work, not implemented by these kickoff tools. No IROS held-out slots have been
reserved or opened. The packet's conference date remains a planning assumption pending official CFP verification.
