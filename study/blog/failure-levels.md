# Which robot failures can a language model fix by editing weights? A failure-level study on π0.5

**Ali-Adeeb Abbas and Esmaeil Seraj**  
Draft for coauthor review, November 5, 2026. This report adds a second cohort, Claude Opus as the editor on our own runtime, to the completed [Astra study](draft.md). Measured outcomes, literature reports and hypotheses are labelled separately throughout. Nothing here changes the frozen Astra data.

## Abstract

Recent reports show large language models repairing robot policies by reading and editing their weights directly, with no gradient training. We asked a narrower question: *which kinds of failure* in a generalist vision-language-action policy are reachable this way? We surveyed 16 RoboLab manipulation tasks with the π0.5 DROID joint-position checkpoint and sorted failures into levels:

- perceptual target confusion;
- language binding;
- missing motor competence;
- contact physics;
- gripper commitment.

We diagnosed five failing tasks, one per level, mostly with Claude Opus as an autonomous editor. On three of them we ran sparse weight edits closed-loop, each aimed at a mechanism localized first in the activations. We also replicated the Astra study's strongest control as a one-scalar weight edit. Across about 1,400 simulated episodes, **no weight edit improved closed-loop success on any task**:

- **Claude, stacking:** 3 edits, each 21/64 against 28/64.
- **Astra, stacking:** 4 edits, none accepted.
- **Flat box (missing competence):** 10 edits, all 0 successes.
- **Gripper bias, 7 tasks:** 117/240 against 117/240, with 37 episodes fixed and 37 broken.

Two results stand out against that background. First, the policy's own unchanged-weight variability is large: about a third of paired episodes flip outcome between runs. Second, the one large effect we measured came from rewording an instruction, not from editing weights: 3/32 → 22/32 on a two-step task whose failure we traced to how the language model binds "in the bowl". The largest outside success was reported on task-fine-tuned checkpoints with recurring motor-decision failures. We found no failure of that kind in the zero-shot generalist setting.

![Paired effect of every intervention, grouped by failure level](figures/opus_forest.png)

*Figure 1. Success rate of each candidate minus the unedited policy on the same simulator slots, with paired 95% bootstrap intervals. Blue: Claude Opus weight edits (this cohort). Orange: GPT-6 Astra weight edits ([Astra study](../RESULTS.md)). Green: interventions that are not weight edits. Grey: unchanged-weight reruns. Right column: candidate vs base successes. Rows from different runtimes are not pooled.*

## 1. Introduction

A robot policy can perceive a scene correctly and still fail in the last few centimetres. *Weight-space programming* (WSP) asks whether a reasoning model can diagnose such a failure inside the network and patch it directly, the way a programmer patches code. The idea was demonstrated on small networks and language models ([Takehi et al., Nov 2](https://rikiyat.github.io/blog/weight-space-programming/)). It is now being tried on robot policies.

The practical question for robotics is not whether a patch *can* exist. It is which failures an editor can find and fix reliably, and how one would know. Our working hypothesis, stated before the experiments in this cohort, was that **local motor failures** (a policy that knows the task but executes one step badly) would be editable, while **perceptual failures** (choosing the wrong object) would be distributed and hard. This report tests that hypothesis on one generalist policy and reports where it held and where it did not.

**Contributions.**
1. A base-rate survey of 16 RoboLab tasks (256 episodes) with each failure classified by level.
2. Mechanistic diagnoses of five failing tasks, one per level, from activations, kinematics, event logs and camera frames.
3. Every closed-loop weight edit we ran, including nulls, with paired intervals: 3 perceptual, 10 motor, and a one-scalar gripper edit on 7 tasks.
4. A replication of the Astra study's strongest control as a weight edit, on 7× its task coverage.
5. A measurement of unchanged-weight variability, and what it implies for how large a study has to be to detect a real repair.

## 2. What others report

These are **reported** results, not ones we reproduced. Detailed assessments are in our [literature survey](../review/literature-survey.md) and [Guo–Xie review](../review/guo-xie.md).

| Work | What was edited | Reported outcome | Scope we note |
|---|---|---|---|
| [Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/) | Small policies; language-model loops | Reading, construction and repair demonstrations | Not a large VLA policy |
| [Gupta et al.](https://codnay.github.io/blog/llms-fix-robots-by-editing-weights/) | π0.5 neurons on 10 tasks | Direct gains on 5/10 tasks; separate folding result 51% → 75% | The folding update uses a difference between trained models |
| [Guo and Xie](../review/guo-xie.md) | π0.5 readout and attention | One task 28/92 → 82/92 held-out with 4 parameters | Each checkpoint already **fine-tuned on its task**; recurring failure such as "hovers, never presses" |
| [Seraj (Astra study)](draft.md) | Action-MLP rows, π0.5 DROID | 4 edits on stacking, none accepted | Generalist checkpoint; restricted interface |

The strongest positive report (Guo and Xie) has two features our setting lacks. Its checkpoints were fine-tuned on the task being repaired, and its failures were recurring motor decisions inside an otherwise competent skill. We do not know whether Gupta et al.'s checkpoints were task-trained.

## 3. Setup

**Policy.** OpenPI `pi05_droid_jointpos`, about 3.35B parameters:
- a PaliGemma vision-language model (18 layers, width 2048);
- a separate Gemma action expert (18 layers, width 1024) that runs 10-step flow matching.

Every 15 simulator steps it outputs 15 actions: 7 joint-position deltas and 1 gripper command, which the simulator binarises at 0.5. The checkpoint was trained on DROID, not on any RoboLab task, so every result here is zero-shot.

**Simulator.** NVIDIA RoboLab on Isaac Sim, using the `performance` rendering preset and 16 parallel environments per evaluation. A *slot* fixes the policy's noise-seed stream and the scene layout. All runs used one RTX 3090. The host's second GPU is power-limited (about 5 TFLOPS), which is why every number here comes from a single card.

**Editor (this cohort).** Claude Opus, at extra-high or max effort, run headless with one command, `owsp`, plus file tools. `owsp` gives:
- a CPU numpy sandbox over 730 weight tensors;
- activation capture on the *exact* recorded inputs of past decisions, parity-checked against the live policy;
- closed-loop practice rollouts paired against the base;
- event timelines and camera frames.

**Rules** follow Takehi et al.:
- no gradients, optimizers, training or black-box search;
- every edit has a mechanistic hypothesis written down before it is evaluated;
- instructions cannot be changed.

Some edits in §5.3–5.6 were made by the operator session (a Claude Opus session run by the authors) under the same rules.

**Runtimes.** Our runs use the repository's top-level runtime. The Astra study used its own "original" and "fixed16" runtimes on a separate two-GPU host. **Rows from different runtimes are shown side by side but never pooled.** The base rates agree reasonably on the shared stacking task: ours 27–28/64, Astra's 23–26/64 and 11/32. On the Astra development slots 20032–20063 our runtime gives 12/32, against Astra's 11/32.

**Statistics.** Each comparison is paired by slot:
- *fixed* means base failed and the candidate succeeded;
- *broke* means base succeeded and the candidate failed.

We report the difference in success rate with a paired bootstrap 95% interval (20,000 multinomial draws over pair types, the same method as the Astra study) and an exact McNemar test. Our bootstrap reproduces the Astra study's published interval for its threshold control exactly (+20.3, [+4.7, +35.9]). No held-out evaluation was run, because no candidate qualified for one.

## 4. A survey of failures by level

Before editing anything, we ran the unedited policy for 16 episodes on each of 16 tasks. Each failure was classified from RoboLab event logs and recorded gripper commands.

| Task | Base | Dominant failure classes (count of 16) |
|---|---:|---|
| Cube into bowl | 16/16 | — |
| Banana onto plate | 16/16 | — |
| Butter box onto raisin box | 14/16 | 1 grabbed then dropped, 1 closed but missed |
| Stack three cubes | 11/16 | wrong object first ×5 |
| Block tower, any order | 11/16 | wrong object first ×5 |
| Stack left bowl on right bowl | 10/16 | wrong object first ×5 |
| Stack yellow on red | 6/16 | wrong object first ×9 |
| Pick up eyeglasses | 5/16 | wrong object first ×6, grabbed but not lifted ×5 |
| Mustard into left bin | 3/16 | **grabbed then dropped ×13** |
| Cube then banana into bowl | 2/16 | wrong object first ×8, placed but not success ×4 |
| Cube left of bowl | 1/16 | wrong object first ×14 |
| Marker in mug | 1/16 | closed but missed ×6, mixed |
| Yogurt into bowl | 1/16 | wrong object first ×7, mixed |
| Cube right of bowl | 0/16 | wrong object first ×16 |
| Grab a bagel | 0/16 | wrong object first ×7, mixed |
| Pick up flat green box | 0/16 | grabbed but not lifted ×10 |

*"Wrong object first" relies on RoboLab's wrong-object event, which the [Astra study's audit](../RESULTS.md) found produces false annotations. Treat those counts as upper bounds; we confirmed the stacking case separately from kinematics and frames (§5.1).*

From the survey, the diagnoses below, and outside reports, we use six levels. The last two columns give what we measured and what others report:

| Level | What fails | Our example | Weight edits here | Outside reports |
|---|---|---|---|---|
| **L1 Gripper commitment** | Close/open decisions near the threshold | All tasks (§5.6) | One scalar, 7 tasks: no effect | Runtime threshold 0.4 on stacking: +20 pp (Astra study), comparable to its own unchanged repeat (+14 pp) |
| **L2 Motor decision, skill present** | A competent skill stalls at one step ("hovers, never presses") | **None found** in 16 zero-shot tasks | Not testable here | Guo–Xie: large gains on task-fine-tuned checkpoints. Gupta et al.: direct gains on 5/10 tasks |
| **L3 Contact physics** | The policy does the right thing but the grasp slips | Mustard (§5.4) | Offline checks failed; gripper bias: no effect | — |
| **L4 Missing competence** | The policy never produces a working motion | Flat green box (§5.3) | 10 edits change motion, 0 successes | Reports agree editing does not add missing skills |
| **L5 Perception: target choice** | Goes for a similar-looking wrong object | Stack yellow on red (§5.1) | Claude 3 edits, −11 pp (n.s.); Astra 4 edits, n.s. | — |
| **L6 Language binding** | Instruction parsed so the wrong object is "current" | Cube then banana (§5.2) | Mechanism localized; no closed-loop edit yet | — |

The level the outside successes target, L2, is the one we could not find. A generalist policy run zero-shot seems to fail mostly at L4–L6: it lacks the motion, picks the wrong object, or misreads the instruction. A task-fine-tuned policy has already learned the motion and the object, so what remains is more often an L2 decision glitch. This is a **hypothesis** consistent with the evidence, not something we have tested on matched checkpoints.

## 5. Case studies

### 5.1 L5, perception: stack the yellow block on the red block

**Base.** 28/64 (43.8%) on practice slots; 27/64 in a later independent rerun.

**Diagnosis (Claude Opus).** The editor combined forward kinematics, event timelines and frames. A Rubik's cube with its yellow face up sits beside the red block, and the arm descends on it first in 48 of 63 episodes. Outcome depends on what happens next:

| First approach | Episodes | Success |
|---|---:|---:|
| Rubik's cube, and the gripper closes on it | 28 | 5 (18%) |
| Rubik's cube, lifts away and redirects to the yellow block | 20 | 14 (70%) |
| Straight to the yellow block | 15 | 9 (60%) |

In the network, the vision-language model's wrist-camera tokens separate the yellow block from the Rubik's cube almost perfectly (d′ ≈ 10–20). The action expert barely reads that signal (d′ ≈ 2–3, against a permutation null of about 1.7). Direct logit attribution places the gripper close/open decision mainly in action-expert layer-16 attention.

**Edits.** Each was validated offline before rollout.

| Edit | Values changed | Success | Paired difference |
|---|---:|---:|---|
| `v1a1`: gain on the identity axis in action-expert attention outputs, layers 1–16 | 30.4M (~0.9%) | 21/64 | −10.9 pp [−26.6, +4.7] |
| `dkv2`: gain on the identity axis in VLM key/value projections, layers 1–13 | 12.0M (0.36%) | 21/64 | −10.9 pp [−26.6, +6.2] |
| `combo1`: `dkv2` plus a rank-1 rewire of one layer-13 attention head | 12.2M | 21/64, and 21/64 again on rerun | −10.9 pp [−26.6, +4.7] |

**What went wrong.** The editor found that amplifying the identity signal also suppressed the policy's own learned veto against closing on the Rubik's cube, the behaviour behind the 70% recovery row. The information exists, but routing it to the decision without breaking a behaviour the policy needs was not a sparse edit. The Astra study's four edits on the same task, through action-MLP rows only, were also within noise (Figure 1, orange).

### 5.2 L6, language binding: "Put the cube then the banana in the bowl"

**Base.** 9/64 (14.1%). The arm goes to the cube first in 63 of 64 episodes. The dominant failure is that it grasps the cube, lifts it a few centimetres, then opens the gripper well short of the bowl. When the cube does reach the bowl first, the episode succeeds about 7 times in 9.

**Diagnosis (Claude Opus).** The editor replayed 21 recorded premature-release decisions under different instructions, keeping images, state and noise fixed:

| Instruction at the replayed decisions | Releases (of 21) |
|---|---:|
| Original: "Put the cube then the banana in the bowl" | 21 |
| "Put the rubiks cube then the banana in the bowl" | 21 |
| "Put the banana then the cube in the bowl" | 3 |
| "Put the cube in the bowl then the banana" | 3 |
| "Put the cube in the bowl then put the banana in the bowl" | 2 |
| "Put the cube in the bowl" | 1 |

What these replays show:
- **"In the bowl" binds to the nearest noun.** In "Put X then the Y in the bowl", the destination phrase binds to Y, so while holding X the policy treats it as the wrong object and lets go. Renaming the cube changes nothing; moving "in the bowl" next to the cube fixes it.
- **The binding shows up in attention.** Vision-language attention from "in"/"bowl" goes to the adjacent noun far more than to the first one (e.g. layer 1 head 1: 0.69 vs 0.02).
- **The decision forms early.** It is already visible at the image and state tokens in VLM layers 10–16.

**Ceiling.** Rewording the instruction explicitly ("Put the cube in the bowl then put the banana in the bowl") lifts the *same weights* from 3/32 to **22/32** on paired slots (20 fixed, 1 broken, p < 0.001). This is a prompt result, not a weight edit, and is labelled green in Figure 1. It shows that the motor skill is fully present and that the failure is a localized language-binding computation. That makes it the best-supported target for a weight edit in this study, and the first one we will attempt next (§7).

An earlier editor run on the mirrored task ("banana then cube"; base 99/128) found the same release mechanism. Replays reproduced the premature release in 28 of 29 cases, and explicit wording cut it to 9 of 29. That editor measured the ceiling of a perfect fix there at about +3 pp, too small to resolve, and correctly submitted no edit.

### 5.3 L4, missing competence: pick up the flat green box

**Base.** 0/64. Two failures, diagnosed by Claude Opus:
- **F1, wrong wrist angle.** In 12 of 16 first approaches the box's long side is wider than the 85 mm gripper opening, so the fingers land on top of it.
- **F2, no commitment.** In the rotated pose where a grasp could work, the gripper command flip-flops between close and open.

**Edits.** Ten, from three parallel editors and the operator:
- readout surgery (`ro8`, `and3`, `and3rot`);
- a neuron flip (`rw1`);
- a layer-16 "stay closed" latch (`e2-contact6`);
- one- and two-scalar wrist and gripper biases (`gb4`, `wr1gb4`, `wr2gb4`, `wr2gb6`, `wr2e1`).

**All ten scored 0** (0/16 each, `e2-contact6` 0/32). The edits did change behaviour: the strongest wrist bias raised the share of episodes reaching the graspable wrist angle from 9/16 to 16/16. A grasp-and-lift never followed. Some apparent near-misses turned out, on inspection, to be the gripper closing on air. We read this as an L4 case: an edit can steer the motions the policy already makes, but cannot assemble a grasp it never learned.

### 5.4 L3, contact physics: put the mustard in the left bin

**Base.** 13/64 (20.3%), and the skill clearly exists. All 13 survey failures share one pattern:
1. The policy grabs the bottle by its narrow cap at step 34–54.
2. It keeps the gripper commanded closed through the lift.
3. The bottle slips out mid-carry.
4. The policy re-grasps the fallen bottle 4–8 times and times out.

![Wrist camera: a cap grasp that slips and one that holds](figures/opus_mustard_slip.jpg)

*Figure 2. Wrist-camera inputs for a slipping grasp (top) and a holding grasp (bottom) of the same bottle on the same scene. Both grip the cap. The difference is not visible in the grasp itself.*

**Diagnosis (operator, 64 base episodes).**
- **It's a slip, not a decision.** In 15 of 16 first drops the gripper was commanded closed throughout, and the simulator binarises the command, so grip force is fixed.
- **Little predicts which grasps hold.** Close height, wrist yaw, the pause between closing and lifting, and carry speed, acceleration and tilt do not separate holding grasps from slipping ones (|Spearman ρ| ≤ 0.26).
- **One weak signal.** Grasps more than 2 cm off-centre never succeeded (0/16), against 13/48 within 2 cm. A perfect centring fix would be worth only about +7 pp.

**Edits.**
- **Neuron flip** (Gupta-style). We captured action-expert MLP activations at 192 grasp decisions. The best layers separated holding from slipping grasps with held-out AUROC 0.66 (layers 14–15). Flipping the top 64 neurons per layer in two layers changed the actions by 0.0005 rad and flipped 0 of 96 close decisions. A stronger version (256 neurons in each of 4 layers) moved the representation *toward* the slipping side. Both were stopped by their offline checks before any rollout.
- **Gripper bias** (§5.6): 13/64 against 13/64.

### 5.5 L2: the level we could not find

The published successes describe a competent policy stalling at one step, such as hovering over a bell without pressing. None of the 16 surveyed tasks showed this. The closest candidates (green box F2, a gripper that won't commit) sat inside a larger L4 failure. Without an L2 failure we cannot test the outside reports on this checkpoint. A matched test needs a task-fine-tuned checkpoint (§7).

### 5.6 L1, gripper commitment: replicating the Astra threshold control as one weight

The Astra study's largest effect was a runtime change: binarising the gripper at 0.4 instead of 0.5 gave 36/64 against 23/64 on stacking. Its own unchanged-weight repeat, however, also moved from 26/64 to 35/64. We asked whether the effect survives as a weight edit and on more tasks.

**Edit (`gthr04`).** The gripper channel is quantile-normalised over [0, 1], and the flow-matching output integrates the action head's bias over 10 steps. So shifting `action.action_out_proj.bias[7]` by −0.3 raises the gripper command at every action. That is **one scalar** of 3.35B. Offline, on 96 recorded decisions:
- the command rises by +0.11 on average (median +0.07), equivalent to a threshold of about 0.4;
- arm joints change by only 0.002 rad;
- closed action steps go from 674 to 781 of 1440.

| Task | Base | `gthr04` | Fixed / broke |
|---|---:|---:|---:|
| Stack yellow on red | 27/64 | 27/64 | 12 / 12 |
| Mustard into left bin | 13/64 | 13/64 | 10 / 10 |
| Pick up eyeglasses | 12/32 | 13/32 | 7 / 6 |
| Stack left bowl on right bowl | 19/32 | 18/32 | 6 / 7 |
| Butter box onto raisin box | 14/16 | 14/16 | 2 / 2 |
| Cube into bowl | 16/16 | 16/16 | 0 / 0 |
| Banana onto plate | 16/16 | 16/16 | 0 / 0 |
| **Total** | **117/240** | **117/240** | **37 / 37** |

The edit is exactly neutral in aggregate and breaks nothing on the three near-ceiling tasks. On our runtime, the +20 pp threshold effect does not appear as a weight edit. The Astra study's own repeat control suggests the original effect may have been within run-to-run variation. A remaining difference is that a bias does not reproduce a runtime threshold exactly, because the network partly compensates.

## 6. The noise floor

Unchanged weights do not give unchanged outcomes:
- **Same seeds, same weights, different result.** Two runs of `combo1` with identical weights and seeds agreed on only 46 of 64 episodes (28% flipped).
- **Astra's unchanged reruns.** They spanned −4.7 to +14.1 pp (grey rows in Figure 1).
- **Near-neutral edits.** On the four tasks with headroom, 70 of 192 pairs (36%) flipped under the near-neutral `gthr04`, half in each direction.

The cause is GPU physics nondeterminism, amplified by open-loop 15-step action chunks. The Astra study's first-input audit found different images and actions in all 64 identity pairs, despite equal recorded state and seeds.

With about 30% of pairs discordant (28–36% above), the standard error of a paired difference at n pairs is roughly √(0.3/n). The smallest effect detectable at 80% power and α = 0.05 is about 2.8 times that:

| Pairs per arm | Smallest reliably detectable change |
|---:|---:|
| 32 | about 27 pp |
| 64 | about 19 pp |
| 128 | about 14 pp |
| 256 | about 10 pp |

Several reported aggregate gains sit in the 10–20 pp range, such as Guo and Xie's single-edit RoboTwin aggregate of 31% → 47%. Below about 128 paired episodes per arm, a single comparison can neither confirm nor rule out gains of that size. We also caution against treating a 16-episode screen as evidence in either direction. Several of our early "first movement" readings did not survive a larger sample.

## 7. Discussion and next steps

**What the levels suggest.**
- **L1, gripper commitment:** a global threshold-style edit is neutral here.
- **L3, contact physics:** there was nothing for a decision-level edit to act on.
- **L4, missing competence:** edits steered motion without producing success.
- **L5, perception:** the signal existed, but routing it broke a needed behaviour.
- **L6, language binding:** the mechanism is localized and the ceiling is known (+59 pp from the same weights), so it is the most promising target.
- **L2:** the level where others report large wins, and the one this zero-shot setting did not offer.

Our original hypothesis, that motor failures would be easy and perceptual failures hard, therefore needs refining. **"Motor" is not one category.** A motor *decision* inside a learned skill (L2) may be editable, while motor *competence* (L4) and *contact physics* (L3) were not here.

**Why outside results may differ.**
- **Task fine-tuning.** The largest positive report used checkpoints trained on the task being repaired, which plausibly places their failures at L2.
- **Interface.** Guo and Xie also used attention-level edits, which the Astra study's MLP-only interface excluded.
- **Evaluation size.** Ours and Astra's evaluations were noisy enough that modest true gains could be missed.

**Planned experiments, in priority order.**
1. **Cube then banana (L6).** A closed-form edit to the VLM binding pathway so that the original instruction behaves like the explicit one, scored against base at 128+ pairs.
2. **A matched L2 test.** A task-fine-tuned π0.5 checkpoint with a recurring "never commits" failure, with the Guo–Xie interface (readout plus attention).
3. **Claude on the Astra protocol.** A Claude Opus editor under the Astra study's brief and budget (3 proposals × 32 slots, 96 activation items, ≤128 rows in 2 layers). It is prepared on our runtime (base 12/32 on the same development slots) and not yet run. The exact fixed16 runtime needs a second working GPU.
4. **Calibrate gates on repeated unchanged-weight runs** before searching, as the Astra study recommends.

## 8. Limitations

- **Single policy, simulation only, zero-shot.** No physical-robot results and no cross-checkpoint comparison.
- **Practice slots only.** No candidate qualified for held-out scoring, so every interval here is a practice estimate.
- **Some edits were operator-built.** The one-scalar `gthr04` and the mustard neuron flips were built by the operator session under the editor rules, not by an autonomous editor.
- **Unequal sample sizes.** n differs across tasks (16–64 pairs), and the three near-ceiling controls cannot show improvement.
- **Failure classes are imperfect.** They depend on RoboLab events, some of which are mislabelled. Levels L3–L6 were assigned from diagnosis, not from an independent rater.
- **Costs are partial.** Recorded editor API spend for this cohort is about $800 (the stacking editor alone, $632). Operator-session and GPU costs were not metered.

## 9. Reproducibility

- **Evidence record:** [`results/opus/EVIDENCE.md`](../results/opus/EVIDENCE.md).
- **Comparison data:** [`results/opus/comparisons.json`](../results/opus/comparisons.json).
- **Figures:** `python3 results/scripts/plot_opus_forest.py` and `python3 results/scripts/make_mustard_strip.py <slip frames> <hold frames>`.
- **Runtime:** top-level `weightsmith` at `f503fb3`, with the patches listed in `third_party/`.
- **Raw episodes, notebooks and editor briefs:** on the GPU host under `~/vibes/opus-weight-surgery/projects/`, project names as in the evidence record.
