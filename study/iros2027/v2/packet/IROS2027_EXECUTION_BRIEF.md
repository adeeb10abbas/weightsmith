# Weightsmith → IROS 2027: Simulation-first execution brief

**Target:** IROS 2027, paper submission **March 1, 2027** (IEEE RAS event listing; confirm author-specific time zone/format when CFP is published).
**Base repo:** https://github.com/adeeb10abbas/weightsmith (main)
**Reference:** *Distilling Instructions into Weights* (arXiv:2611.11894v1). **Important:** the project website also documents later Cosmos3 Edge and GR00T N1.7 experiments not included in arXiv v1. Audit the current worktree and existing results before duplicating them.

## Integration rule: extension, not replacement (v2 audit)

**Preserve the original evidence as Cohort A.** The experiments in `paper/main.tex`, `BENCHMARK.md`, `study/results/opus/comparisons.json`, `study/results/opus/ctb/`, `study/results/ctrl/README.md`, and `study/RESULTS.md` are historical, frozen results. Keep their task wording, fixed scene, seed protocol, original check-point, original runner, failure reports, and analysis unchanged. NEVER overwrite or retroactively rescore those results. Do not pool the earlier Astra and Claude studies: their runtimes/baselines differ.

**Add IROS experiments as separate cohorts, each answering a new question:**

| Cohort | Existing evidence to retain and cite | New contribution | Do not rerun merely for N |
|---|---|---|---|
| A: published fixed-scene evaluation | `v2mi-f`: 27/128 → 106/128; minimal-image-cell edit: 19/128 → 102/128; random control 13/64 → 9/64; explicit wording 13/64 → 57/64; transfer to eyeglasses and cube-left | A fixed anchor result; optional exact reproduction on newly reserved seeds clearly labeled *replication*, not new independent evidence | Old 128 CTB episodes and old same-scene transfer/control trials |
| B: independent diagnosed repairs | CTB is the one established clear repair; unsuccessful perception/contact/missing-skill cases remain limitations | Distinct *semantic task goals / failure diagnoses* with independently built edits, known-good prompts, frozen held-out tests, and a disclosed screening denominator | CTB paraphrases miscounted as new repairs |
| C: scene-level generalization | Original RoboLab tasks hold environment seed 0 and primarily vary policy noise | Apply the original frozen `v2mi-f` **and** any new frozen edits to held-out actual scene layouts using identical predicates and documented validity checks | Interpreting new noise seeds as new scene layouts |
| D: systematic edit interference | Original CTB edit already tested on many of the 16 tasks, with transfer, neutral outcomes, and some regressions | Evaluate each *new* edit on the same benchmark suite, then add opposite-relation, same-relation/new-object, and scene-shift checks; for original edit, fill uncertainty/specificity gaps only | Rerunning identical CTB-vs-base retention cells under the same protocol |
| E: model differences | Reuse any verifiable Cosmos3 Edge / GR00T N1.7 study artifacts, if present in current repo or project site | A common metric schema and **new** architecture-specific gaps only; never claim weight coefficients or edits transfer across models | Repeating published cross-model experiments because a new plot requires them |

**Do not silently alter the original protocol.** Scene-randomized tasks and revised failure selectors are new evaluation cohorts with their own versioned configs. Main fixed-scene baseline stays valid and separately comparable; if implementation changes affect episode success, use a contemporaneous baseline for the new cohort and disclose the difference. All new files should live under a versioned `study/iros2027/` namespace. Record source Git SHA, edit SHA, checkpoint SHA, simulator SHA, and scene sampler revision.

**Important counting correction:** The 12 registry rows are *12 wording pairs for screening*, not 12 independent trials of the method. T01 is the published CTB case. T04 is another wording of the same CTB goal; T02/T03 share a task; B01/B02 share a task; B03/B04 share a task. S03 is flagged as an existing-site result requiring independent provenance verification. S01/S02 are related converse-relation tests and must be reported as correlated. Count as an *independent repair* only an edit independently constructed on a distinct diagnosed failure/semantic goal with fresh held-out evidence. Report pair-level results separately; cluster by task/goal family for generality claims. See `IROS2027_DELTA_AND_REUSE.md`.

**Resource guardrail:** Each job must carry a `reuse_or_new` label and a one-line claim it supports. If an existing result supports the identical claim with the same checkpoint, edit, task, scene, and scoring protocol, skip the run unless pre-registered as reproducibility/precision confirmation. Never replace a negative result with an eligibility-filtered subset without retaining the full screening denominator.

## Mission / falsifiable claim

**Hypothesis:** When a VLA or world-action policy has a latent manipulation skill but fails because of instruction grounding, a frozen, closed-form edit that distills an equivalent successful wording can recover the skill *across independent failures and changed scenes*, with bounded effects on unrelated instructions. The method will fail predictably when there is no known-good wording, insufficient word/token alignment, or a missing physical skill.

**What we must NOT claim:** generic correction of all VLA failures; general spatial-language competence; hardware generalization before hardware tests; OOD scene generalization from policy-noise seeds alone; or zero collateral damage from nonsignificant tests.

## Priorities (in order)

### P0 — Reproducibility & experimental hygiene (1–3 days)
- [ ] Pin repo commit, π0.5 checkpoint SHA, OpenPI/RoboLab/Isaac Sim versions, hardware, driver/CUDA. Preserve site/paper results provenance (especially Cosmos3/GR00T).
- [ ] Reproduce one published π0.5 repair through `weightsmith distill run` and independently reproduce **base vs base** and **edited vs edited** to measure run-to-run discordance. Keep development and held-out slots disjoint; previously used held-out slots are **not fresh**.
- [ ] Define dataset keys: `architecture, checkpoint_sha, task_id, wording_id, semantic_goal_id, scene_seed, policy_seed, eval_replica, candidate_sha, condition, job_id`.
- [ ] Ensure every matched condition uses *the same scene specification* and policy-noise stream, with run order counterbalanced (ABBA or alternating blocks). Verify early camera-frame hashes and initial object poses. If the simulator is nondeterministic, report it; do not imply a perfect replay.
- [ ] Run each task against an operator-only sealed held-out set. Agents can read practice/dev data only. Failures and unpromising screens are logged, not discarded.

**Deliverable:** one reproducibility ledger + comparison plot base/base, base/edited, base/random, base/explicit. No new claims until the noise-control result is in.

### P1 — New *independent* language-repair cases (highest scientific value)

Do **not** just scale the original cube→banana task. Screen **12 registry wording pairs** across three families (4 per family), of which **two are reference/existing** (T01, S03) and several share task goals. Report the total screened denominator separately from genuinely new, independently diagnosed failures. Prefer task primitives the existing π0.5 DROID model reliably completes when explicitly prompted. Proposed families:

1. **Temporal scope / clause attachment:** original cube-then-banana ambiguity, plus new multi-step objectives built from the *already competent* cube-in-bowl and banana-on-plate primitives. Example: `Before putting the banana on the plate, put the cube in the bowl.` vs `Put the cube in the bowl first. Then put the banana on the plate.` Verify the same action-order and goal predicates. Canonical task predicates stay identical.
2. **Spatial viewpoint / converse relation:** e.g. `Move the cube to the left of the bowl` vs `Place the bowl to the right of the cube` are *not necessarily equivalent actions* when the bowl is fixed; use *goal-state* formulations that name which object is moved, or take the already observed Cosmos3 mustard/raisin converse pair as the first case. Include true goal-equivalent left/right and right/left pairs. Freeze frame-of-reference convention (world vs camera vs agent) per task.
3. **Object–destination binding / distractors:** two movable objects, two receptacles (e.g., cube→bowl; banana→plate). Original sentence combines clauses; explicit sentence binds each destination. Ensure task is physically feasible with this checkpoint before treating an error as language-binding; otherwise report *not eligible* rather than force an edit.

**A case qualifies for the repair study only if** (i) both prompts mean the same physical goal under a written rubric; (ii) unedited policy performs the target skill reasonably under the explicit wording; (iii) there is a meaningful wording-linked error when replaying the same decisions; (iv) edit construction uses only approved practice captures. Suggested *screening* rule (not a significance claim): on 32 matched development starts, original ≤ 40% success, explicit ≥ 60%, gap ≥ 25 percentage points, and identifiable action-level change. Log **all 12** attempted pairs, including ineligible/failed, so that method coverage is measurable.

For each eligible case:
- [ ] Build **one candidate** using a frozen distillation recipe (the existing image-cell layers 8–16 recipe first), with a fixed small edit/capture budget. Record all candidate attempts and rejected offline checks. Do not modify scoring conditions after observing held-out results.
- [ ] Compare: **A** base/original; **B** base/explicit; **C** distilled edit/original; **D** norm/rank/spectrum-matched random edit/original; **E** distilled edit/explicit (preservation of known-good behavior). Add an *optional* fine-tuning/LoRA comparator only where a fair, explicit supervision and compute budget can be defined. Do **not** call fine-tuning gradient-free.
- [ ] Report full-task success *and* precise failure stage: wrong object, wrong destination, premature release, failed grasp, contact slip, wrong order, time-out; labels audited for event-oracle mistakes.

**Primary paper outcome:** `# distinct independently repaired goal/failure units / # eligible distinct goal/failure units`, and conditional success-rate improvement on a **fresh, frozen** held-out set. Explicitly separate this from `# eligible / # screened` to prevent selection bias.

### P2 — Actual scene generalization (requires adding scene randomization)

The current benchmark's **slot primarily controls policy-noise stream**, not a broad distribution of new scene layouts. A held-out slot is NOT a held-out physical scene. Extend RoboLab with deterministic, logged scene samplers:
- Small within-reach 2D object-pose perturbations (first, easiest); counterbalanced left/right positions.
- New distractor identities/positions; lighting, texture, and camera jitters as separate stress axes.
- A hard split of scene configurations: `dev_scene_seed`, `validation_scene_seed`, `test_scene_seed`, plus independent `policy_seed` for each. Keep the success predicate and reachability constraints fixed. Verify scene perturbations are valid and symmetric.
- Evaluate **in-distribution unseen scenes** and **shifted/OOD scenes** separately. Add a competence oracle (same baseline, explicit wording) so failures due to unreachable grasps are not misclassified as language failures.

**Implementation deliverable:** a small extension to the RoboLab task factory that accepts a sampled scene spec and writes that spec to each episode's metadata. No visual-only random seed claim without checking actual object positions.

### P3 — Edit specificity / interference / composition (major differentiator)

The project site already observes that one Cosmos3 Edge spatial edit hurts other tasks involving `left`. Make this systematic:
- Pre-register **16–24 retention instructions** stratified into: unrelated task; same object/different relation; same relation/different object; opposite direction; paraphrase of the repaired instruction; original known-good wording.
- Each frozen edit is evaluated unchanged on the entire retention suite. Include all model architectures where that edit exists; do not confuse *same edit coefficients* with *same algorithm* across architectures.
- Quantify `gain on target`, `mean retention delta`, `worst regression`, `fraction of tasks regressed`, and 95% uncertainty intervals. Predeclare practical non-inferiority margin (e.g. no worse than −10 percentage points in pooled retention); CI crossing the margin means **inconclusive**, not 'no harm'.
- Mechanistic controls: edit location (text/VLM vs action expert/head), edit size/sparsity, aligned vs unaligned prompt tokens, random-direction control. Existing findings can be cited; extend only the necessary gaps.
- If attempting two simultaneous edits: report order dependence (A→B vs B→A), target retention after second edit, and unrelated-task interference. **Exploratory unless time remains.**

### P4 — Architecture cross-check and baselines
- [ ] π0.5 is the primary controlled testbed (zero-shot), not the only evidence.
- [ ] Reanalyze already documented Cosmos3 Edge and GR00T N1.7 experiments using the *same metric definitions*. Do not rerun completed cases without a defined replication purpose.
- [ ] Pre-register model/token-alignment hypothesis: equal/aligned prompt tokens vs inserted clauses. Predict failures *before* running. This helps explain Cosmos3/GR00T results without overclaiming.
- [ ] Add a practical comparator: `explicit wording at inference` is an upper bound, not a competing solution when the runtime prompt cannot change. For editing-vs-adaptation, a small LoRA baseline is worthwhile **only if** train data, wall time and compute budget are disclosed fairly.

## Sample sizes and run plan (planning numbers, not fixed statistical guarantees)

| Stage | Unit | Proposed trials | Purpose |
|---|---|---:|---|
| Discovery | 12 planned wording pairs | 32 shared starts × 2 phrasings = **768 episodes** | Find eligible failures, publish denominator |
| Build/dev | each eligible case | 64 practice starts per condition; fixed budget | Construct/check; exclude all dev from test |
| Held-out headline | each **frozen** independent repair | **128 paired starts per condition** for A/B/C/D/E = up to **640 episodes/case** | Estimate causal effect and controls |
| Noise control | per architecture/task family | **128 matched pairs** of identical weights | Test reliability of nominally paired rollout |
| Retention | 16–24 tasks × 64 starts × base/edit per task | **2,048–3,072 episodes per edit** | Detect broad collateral effects |
| OOD scenes | selected 2–3 repairs | 128 shared *scene* starts × A/B/C | Generalization beyond fixed initial scene |

These are a staged budget: do **not** automatically run the full factorial of every candidate, edit, model and scene shift. First screen; then promote predeclared cases; then run fixed held-out packages. Because many comparisons will be tested, declare primary contrasts in advance and report multiplicity-adjusted results (e.g., Holm across primary repaired cases). Estimate power using observed within-pair discordance, not naive independent-binomial assumptions. Don't inflate the impression of independence by treating multiple rollout-noise samples from one physical scene as independent scene generalization.

## Machine-readable result contract

Write one row per full episode (plus an optional per-step event table):

```csv
run_id,git_sha,model_id,checkpoint_sha,task_id,semantic_goal_id,wording_id,condition,edit_sha,scene_seed,policy_seed,eval_replica,order_block,success,fail_stage,episodes_to_success,sim_seconds,wall_seconds
```

Save the frozen `edit.safetensors`, build spec, notebook hypothesis, simulation config, tokenization output, images/rollout video for key figures, and exact pre-/post-edit model hashes. A dashboard produces:
1. **Repair funnel**: screened → wording-sensitive → built → offline-passed → heldout-passed.
2. **Forest plot**: paired held-out delta + confidence interval and fixed/broke counts, by independent repair.
3. **Interference heatmap**: edit × retention instruction, including opposite relations.
4. **Scene-shift robustness**: in-distribution/OOD performance, original vs explicit vs edited.
5. **Mechanism**: selected action-stage traces + VLM/text vs head ablation, plus token alignment.

Primary inferential unit = scene/start pair (and potentially hierarchical task/scene modeling when replicated); use exact McNemar for fixed-vs-broke success on paired starts, paired bootstrap or suitable hierarchical bootstrap for CI, and baseline-vs-baseline to calibrate null effects. For multi-edit claims, report spread across cases and not just aggregate episodes.

## Acceptance gates / release plan

- **Gate A (by Dec 7):** existing paper's pipeline reproduced; matched controls, operator-only test split, scene randomization minimum viable, no leakage.
- **Gate B (by Jan 4):** at least 12 prespecified pairs screened, 3 independent eligible language failures if they exist, and results for *all* screens documented. If fewer than 3 are eligible, explain coverage truthfully and adjust the paper's claim.
- **Gate C (by Feb 1):** ≥2 independently improved language repairs on held-out starts is a *research aspiration*, not an acceptance requirement; complete matched controls, OOD and retention study even if findings are negative. Freeze all results and analysis code.
- **Gate D (by Feb 15):** full IROS manuscript draft, figure pack, tables, limitations. Hardware if available, but no critical-path dependence.
- **Gate E (by Feb 23):** internal critique/overlap check, reproducibility bundle, submission-ready manuscript. Verify final CFP/template/time zone before uploading March 1.

### Scope control
**Must ship:** independent repairs + screened denominator, frozen paired eval + noise controls, specificity/retention, scene randomization, manuscript. **Optional:** new architecture or hardware, LoRA, edit composition. More rollouts of cube→banana only if uncertainty or reproducibility require them.

## Quick start in the *existing* repo (GPU host; not run here)

The following commands are already documented by weightsmith; they are starter checks, not a new automated orchestration framework. Do not test final held-out before freezing a candidate. `$WSP_ROOT` is a separate project so no published data is overwritten.

```bash
# existing GPU host with repo, pinned dependencies and BASE_WORKSPACE set
scripts/setup_deps.sh --check
scripts/smoke_test.sh
export WSP_ROOT="$PWD/projects/iros27-ctb-repro"
mkdir -p "$WSP_ROOT"
cp configs/cube-then-banana.json "$WSP_ROOT/config.json"
scripts/base_practice.sh 64
weightsmith distill run examples/cube_then_banana/distill.json --dry-run
# After verifying this is the intended new practice project:
weightsmith distill run examples/cube_then_banana/distill.json
# Follow the *printed* practice command, then freeze the candidate and only
# then have an independent operator run heldout.py on a NEW, unexamined slot block.
```

`weightsmith distill` supports existing `select` rules only for `early_release` / `grasp`; a new *spatial/object binding* diagnosis will need **additional read-only event selectors** and tests. **Do not** simply put a new prompt into the cube-then-banana recipe and assume it performs the new diagnosis.

## Agent handoff (copy into a coding agent)

> In `adeeb10abbas/weightsmith`, implement the IROS 2027 simulation evaluation program according to `IROS2027_EXECUTION_BRIEF.md` and `IROS2027_SWEEP_MANIFEST.yaml`. Preserve all published results, checkpoint hashes, success predicates, and operator-only held-out isolation. First audit site results for Cosmos3 Edge and GR00T N1.7; do not duplicate completed experiments. Implement a deterministic task/scene-spec sampler, semantic-pair registry with human-review flags, generalized failure-event selectors, and a job queue for five conditions (base/orig, base/explicit, edit/orig, random/orig, edit/explicit). Build read-only scripts for sealed held-out scoring and forest/retention plots. Add CPU tests using fake simulator/model outputs and a GPU smoke path. Run development-only tests until I explicitly authorize a frozen held-out package; never tune after seeing held-out scores. Log every screened pair and all negative results. Start with the existing π0.5 cube-then-banana reproduction; return an implementation PR or patch, reproducibility ledger, and an estimate of GPU jobs before launching the expensive sweep.

**Access status:** The GitHub connector available while drafting this had `pull=true`, `push=false`; this packet has **not** been committed to the repository or run on a GPU host.
