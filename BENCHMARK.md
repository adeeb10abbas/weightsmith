# weightsmith benchmark

**Can a frontier model fix a robot policy by editing its weights directly?**

1. An LLM editor reads and edits the weights of π0.5 through one tool, `owsp`, to raise its success on one RoboLab task.
2. No gradients, training or black-box search are allowed. Every edit is closed-form, with its hypothesis written down before it is tested.
3. The editor practises on paired simulator seeds, then submits one frozen candidate, or `base`.
4. An operator scores that candidate against the unedited policy on at least 64 held-out paired seeds the editor never saw.
5. The candidate is accepted if it wins with an exact McNemar p < 0.05. Current entries are on the [leaderboard](LEADERBOARD.md).

| At a glance | |
|---|---|
| **Policy** | OpenPI π0.5 `pi05_droid_jointpos`: 3.35B parameters, 730 editable bf16 tensors. Zero-shot on every task. |
| **Simulator** | NVIDIA RoboLab on Isaac Sim, 16 parallel environments, fixed scene per task |
| **Tasks** | 16 RoboLab tasks; unedited success ranges from 0/16 to 16/16 ([§2](#2-tasks)) |
| **Editor** | Any LLM agent. Drivers are provided for Claude Code and the Codex CLI (GPT-6 Astra). |
| **Practice** | Slots 20000–20255, paired with base. Reference budget: 1,200 episodes and 6,000 activation items. |
| **Held-out** | Slots ≥ 80000, operator only, scored with `weightsmith/heldout.py` |
| **Gate** | At least 64 held-out pairs, more successes than base, and exact two-sided McNemar p < 0.05 |
| **Noise floor** | About 30% of pairs flip with unchanged weights. Detectable change: about 19 points at 64 pairs, 14 at 128. |
| **Report** | Edit size, budget used, every candidate including rejected ones, and the failure level targeted |

Setup is in [docs/SETUP.md](docs/SETUP.md). The measured results and their interpretation are in the [report](study/blog/failure-levels.md).

---

## 1. Policy and simulator

| Component | Details |
|---|---|
| Policy | A PaliGemma VLM (18 layers) plus a Gemma action expert (18 layers) that emits 15-step action chunks by flow matching (10 Euler steps). It has 3,353,433,872 parameters. 730 tensors holding 2,823,447,440 values are exported as editable bf16 arrays by `scripts/prepare_workspace.sh`. |
| Training | Trained on DROID real-robot data, with **no training on any RoboLab task** |
| Observation and action | Two 224×224 cameras plus joint and gripper state in. The policy is queried every 15 sim steps, and each 15×8 chunk runs open-loop: 7 joint targets and 1 gripper command binarised at 0.5. |
| Episode | A 450- or 900-step horizon, depending on the task. Success is RoboLab's boolean terminal flag; partial scores do not count. |
| Rendering | Isaac Sim `performance` preset, environment seed 0, identical for every candidate |
| Pins | OpenPI fork `xuningy/openpi@aa64205` + 1 patch, RoboLab `v0.2.1` + 6 patches, Isaac Sim 5.0 / Isaac Lab 2.2 (`envs/robolab-sim/uv.lock`), and checkpoint sha256 values in `third_party/pi05_droid_jointpos.sha256` |

## 2. Tasks

Each task was surveyed with the unedited policy on 16 episodes (slots 60000–60015, `weightsmith/survey.py`). Instructions are the strings RoboLab reports, and `owsp eval` refuses to run if a project's config disagrees. Source: report [Appendix A](study/blog/failure-levels.md#appendix-a-unedited-policy-survey-16-episodes-per-task).

| # | Task | RoboLab task | Instruction | Base | Dominant failure classes |
|---:|---|---|---|---:|---|
| 1 | Cube into bowl | `RubiksCubeTask` | "Put the cube in the bowl" | 16/16 | — |
| 2 | Banana onto plate | `BananaOnPlateTask` | "Pick up the banana and put it on the plate" | 16/16 | — |
| 3 | Butter box onto raisin box | `ButterAboveRaisinTask` | "Pick up the butter box and place it on top of the raisin box" | 14/16 | dropped ×1, missed ×1 |
| 4 | Stack three cubes | `Stack3RubiksCubeTask` | "Stack the rubiks cubes in a tower" | 11/16 | wrong object first ×5 |
| 5 | Block tower, any order | `BlockStackingOrderAgnosticTask` | "Stack the blocks into a tower" | 11/16 | wrong object first ×5 |
| 6 | Stack left bowl on right bowl | `BowlStackingLeftOnRightTask` | "Stack the left bowl on the right bowl" | 10/16 | wrong object first ×5 |
| 7 | Stack yellow on red | `StackYellowOnRedTask` | "Stack the yellow block on the red block" | 6/16 | wrong object first ×9 |
| 8 | Pick up eyeglasses | `PickGlassesTask` | "Pick up the eye glasses" | 5/16 | wrong object first ×6, grabbed but not lifted ×5 |
| 9 | Mustard into left bin | `MustardInLeftBinTask` | "Put the mustard in the left bin" | 3/16 | grabbed then dropped ×13 |
| 10 | Cube then banana into bowl | `RubiksCubeThenBananaTask` | "Put the cube then the banana in the bowl" | 2/16 | wrong object first ×8, placed but not success ×4 |
| 11 | Cube left of bowl | `RubiksCubeLeftOfBowlTask` | "Put the rubiks cube to the left of the bowl" | 1/16 | wrong object first ×14 |
| 12 | Marker in mug | `MarkerInMugTask` | "Put the whiteboard marker in the mug" | 1/16 | closed but missed ×6, mixed |
| 13 | Yogurt into bowl | `YogurtInBowlTask` | "Put the small red yogurt in the red bowl" | 1/16 | wrong object first ×7, mixed |
| 14 | Cube right of bowl | `RubiksCubeRightOfBowlTask` | "Put the rubiks cube to the right of the bowl" | 0/16 | wrong object first ×16 |
| 15 | Grab a bagel | `GrabABagelTask` | "Grab a bagel" | 0/16 | wrong object first ×7, mixed |
| 16 | Pick up flat green box | `PickUpGreenObjectTask` | "Pick up the green vegetable block" | 0/16 | grabbed but not lifted ×10 |

- "Wrong object first" relies on a RoboLab event that the [Astra audit](study/RESULTS.md) found unreliable, so treat those counts as upper bounds.
- Base rates measured on more episodes are 9/64 for cube then banana, 28/64 for stack yellow on red, 16/96 for mustard and 0/64 for the flat green box (report §5–6).
- Ready-made configs and briefs exist for `stack-yellow-red`, `cube-then-banana` and `banana-then-cube` (`BananaThenRubiksCubeTask`, 99/128 base) in `configs/` and `briefs/`. For any other task, copy a config and set `task` and `instruction` from the table above.

## 3. Failure levels

Levels were assigned from diagnosis (kinematics, camera frames and activation captures), not by an independent rater ([report §4](study/blog/failure-levels.md#4-results-at-a-glance)). A submission should name the level it targets and give its evidence.

| Level | What goes wrong | Example task | Weight editing so far |
|---|---|---|---|
| Language binding | The instruction is parsed so that the wrong object counts as "current" | Cube then banana | Fixed: held-out 27/128 → 106/128 |
| Perception | The policy goes for a similar-looking wrong object | Stack yellow on red | No accepted edit in 9 attempts by two editors |
| Missing skill | The policy never produces a working motion | Pick up flat green box | 10 edits, 0 successes |
| Contact physics | The decisions are right, but the grasp slips | Mustard into left bin | No decision-level lever found |
| Gripper commitment | Close and open decisions sit near the threshold | All tasks | One weight on 7 tasks: 117/240 vs 117/240 |
| Motor decision, skill present | A competent skill stalls at one step | None among the 16 tasks | Untested here; others report large gains ([report §2](study/blog/failure-levels.md#2-what-others-have-found)) |

## 4. Slots and seeds

A **slot** fixes one episode's policy-noise stream. Slot `s` runs in group `g = s // 16` on environment `e = s % 16`, and replan `k` uses the noise seed `((g·64 + e)·100 + k)`. Base and candidate evaluated on the same slot are therefore **paired**. GPU physics and rendering are not bit-deterministic, so pairing reduces variance but does not make episodes identical.

| Slots | Use | Visible to |
|---|---|---|
| 20000–20255 | Practice: `owsp eval NAME --n N --offset K`, in multiples of 16 | Editor |
| 60000–60015 | Operator base survey. Some of our later projects also practised from 60000. | Operator |
| ≥ 80000 (90000+ by convention) | Held-out scoring. `heldout.py` rejects lower slots, and `owsp` cannot read the results. | Operator only |
| 20032–20063 | Astra original protocol: development | Editor |
| 30016–30047 and 100000–100063 | Astra fixed16 protocol: development and held-out | Editor and operator only, respectively |

## 5. Editor rules

1. **No gradients, no training, no search.** No autodiff, optimizers, fine-tuning, or regressions fitted over many samples to produce weights. No mass-evaluated random perturbations or evolutionary search.
2. **Closed-form edits only**, computed from weights and activations: means and contrasts, projections, SVD/PCA, rank-1 edits along an identified direction, neuron scaling, silencing or sign flips, and head rewiring. The accepted binding edit is a minimum-norm key→value update, `ΔW = Kᵀ(KKᵀ + λI)⁻¹V`, built from 6 recorded decisions. The report treats it as closed-form ([§5.4](study/blog/failure-levels.md#54-the-edit)). Disclose any similar solve.
3. **Hypothesis first.** Before evaluating an edit, write its mechanism, the effect you expect and the breakages you expect in `NOTEBOOK.md`. Afterwards, record whether the prediction held.
4. **Fixed instruction.** Candidates are scored with the task's own instruction. Do not change prompts, inference code, wrappers, the gripper threshold, task predicates or the simulator. Prompt overrides are diagnostics only (`prompt_overrides` in the config; they are disabled on capability tasks such as `stack-yellow-red`).
5. **bf16-exact edits**, validated by `owsp diff` against the original checkpoint.
6. **Tool isolation.** The editor gets `owsp` (or the protocol's typed tools) plus file tools in its working directory. It gets no web access, no MCP servers and no subagents. The drivers configure this; the Astra driver's open checks are listed in [docs/COMPARING_EDITORS.md](docs/COMPARING_EDITORS.md).
7. **No leakage.** The editor never sees held-out slots or results. A fresh editor never receives other editors' outcomes ([study/AGENTS.md](study/AGENTS.md)). Disclose any hints you provide, such as a `PRIOR_FINDINGS.md`.
8. **One answer.** `FINAL.md` names one candidate, or `base`, with a mechanism account. Report every candidate, including rejected ones.

## 6. Budgets

**owsp protocol** (this repository). The editor works over all 730 editable tensors. It has a CPU numpy sandbox (`owsp py`: about 100 GB of RAM and a 30-minute limit per call), closed-loop practice evals, and activation capture on exact recorded inputs. The base practice episodes are pre-run by the operator and are not charged. Wall time is not capped: the drivers resume the session until `FINAL.md` or `STOP` exists, for at most 40 resumes.

| Config | Practice episodes | Activation items (per call) | Prompt overrides |
|---|---:|---:|---|
| `configs/stack-yellow-red.json` | 1,200 | 6,000 (96) | off |
| `configs/cube-then-banana.json` | 1,200 | 6,000 (96) | on |
| `configs/banana-then-cube.json` | 1,600 | 6,000 (96) | on |
| Our green-box and mustard projects | 800 | 6,000 (96) | off |

**Astra fixed16 protocol** (sealed in [`study/reproduction/fixed16/`](study/reproduction/fixed16/README.md); [protocol](study/reproduction/fixed16/reference/protocol.json), [brief](study/reproduction/fixed16/bundle/weightsmith/operator/rq123-v2/design/BRIEF.md)).

| Astra fixed16 | |
|---|---|
| Task and editor | `StackYellowOnRedTask`; `gpt-6-astra` at effort `max`, one adaptive run of 3 rounds |
| Tools | Typed `wsp_*` tools only. There is no shell and no arbitrary code. |
| Editable | The 18 action-expert MLP down-projections, `action.blocks.LL.mlp_1.linear` [4096, 1024]. At most **128 changed rows across at most 2 layers**, cumulatively, after bf16 rounding. |
| Operations | Observation summaries, activation contrasts and row statistics, then `row_scale`, `row_add_direction` or `row_project_out` |
| Evidence | A fresh 32-episode base on development slots 30016–30047 |
| Budget | 3 candidate evaluations × 32 episodes; 96 activation items, failed ones included (512 MiB per capture, 1 GiB retained); 7,200 s including queue waits |
| Held-out | Slots 100000–100063, operator only: 64 base and 64 candidate episodes per unique accepted deployment, in ABBA blocks of 32, plus an identity comparison |

The **original Astra protocol** used the same task, rows, operations, budget and gate on development slots 20032–20063, across two adaptive and two non-adaptive searches. [Claude under the Astra protocol](study/results/opus/claude-astra-protocol-BRIEF.md) mirrors it on the `owsp` runtime. It has the same 32 development slots, 3 evaluations, 96 activation items and 7,200 s, and its `owsp py` sandbox is restricted to the same operations. The sealed dispatcher expects two GPUs; read [study/README.md](study/README.md) before reusing it.

## 7. Acceptance gate

**Held-out acceptance** applies to every protocol and defines "accepted" on the leaderboard.
1. Freeze one candidate per editor run and task before any held-out result is seen.
2. Score it with `weightsmith/heldout.py`, or the protocol's own held-out schedule. Use at least 64 paired slots (a multiple of 32), all ≥ 80000 and never used for practice. Base and candidate run on identical slots, with the order alternating by half. Use a slot block that earlier submissions on the task have not used, or disclose the reuse.
3. The candidate is **accepted** if it has more successes than base **and** the exact two-sided McNemar p < 0.05. Otherwise it is **rejected**.
4. Publish `heldout.py`'s `results.json` unmodified. Held-out runs are never repeated or overwritten.

On the leaderboard, a run is **rejected** if it evaluated at least one weight edit in closed loop and none passed. It is **none submitted** if no weight edit reached a closed-loop evaluation, for example because the run stopped early or rejected every candidate offline.

**Development gate** (Astra protocols, frozen). On its 32 development slots, a candidate is eligible only if:
- it has strictly more successes than base;
- it breaks at most 2 base successes;
- it changes at least one deployed bf16 value.

Eligible candidates rank by more successes, then smaller post-bf16 ΔL2, then earlier proposal. Unchanged-policy repeats broke 3–10 base successes in the study, so this gate is strict. Keep it fixed for comparability ([study/RESULTS.md](study/RESULTS.md)). Candidates that fail it are rejected without held-out scoring. The **owsp protocol** has no automatic development gate: the editor confirms its own candidate on more practice episodes and checks how many base successes it breaks.

## 8. Metrics

- **Primary.** Held-out success of base and candidate on identical slots; fixed (base failed, candidate succeeded) and broke counts; exact McNemar p; Wilson 95% intervals. `heldout.py` reports all of these. Also report the paired difference with a paired bootstrap 95% interval (as in `study/results/scripts/plot_opus_forest.py`).
- **Edit size.** The number of bf16 values changed (`owsp diff`), the tensors and layers touched, and the share of the 3,353,433,872 parameters. For low-rank edits, also give the factor size.
- **Cost.** Practice episodes, activation items, wall time and API spend.
- **Side effects.** Paired comparisons on other tasks from §2.
- **Process.** Every candidate, its written hypothesis, and whether the prediction held.

## 9. Noise floor and sample size

Unchanged weights do not give unchanged outcomes ([report §7](study/blog/failure-levels.md#7-how-not-to-fool-yourself)):
- About 28–36% of paired episodes flip with unchanged or near-neutral weights. Two runs with identical weights agreed on only 46 of 64 episodes.
- The unedited policy scored 14/16, then 8/16, on the same butter-on-raisin slots (0 fixed, 6 broke). Exact McNemar on those counts gives p ≈ 0.03, although nothing changed.
- The Astra study's unchanged reruns moved by −5 to +14 points.

The smallest change one paired comparison detects (about 30% of pairs flipping, 80% power, α = 0.05):

| Pairs per arm | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| Detectable change | ~27 pp | ~19 pp | ~14 pp | ~10 pp |

Treat a 16-episode screen as a filter, not as evidence. Confirm practice gains on at least 64 fresh practice slots. Score held-out on at least 64 pairs, and on 128 for a headline claim.

## 10. Reporting

[results/README.md](results/README.md) lists every published result file and what a submission must include.
