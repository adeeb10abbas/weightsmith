# Track A: weight-space programming on Cosmos3 Edge Policy DROID

Status (2026-11-08 11:35 ET): T8 practice and held-out, the failure map (practice), the where-to-edit comparison, three
preservation variants and the held-out side-effect map of `t8und` are done. Still running: the unedited policy under the
TF wording on held-out starts (teacher, for recovery %), the T6 practice side-effect cell, and the extension of the
practice map to slots 30016-30031.

Question: how well does weight-space programming (WSP, closed-form instruction distillation) fix the failures of
NVIDIA's Cosmos3 Edge Policy DROID, against GR00T (Track B) and π0.5 (Track C, the reference) on the same RoboLab
tasks and slots?

## Headline (all ali-ms, default render, paired on the same starts)

1. **Fix.** T8 (move the mustard right of the raisin box, reference-first wording). Unedited Cosmos3 Edge solves 0/16
   practice and 0/64 held-out starts. The target-first wording solves 15/16. The closed-form text-pathway edit `t8und`
   (RF wording unchanged) solves **13/16 practice and 56/64 held-out** (56 fixed / 0 broke, p = 2.8e-17). Its
   same-size random-direction control solves 1/16 and **0/64**.
2. **Where to edit.** Offline, the text pathway closes 91% of the RF -> TF action gap and the video/action pathway 23%.
   In closed loop the video/action edit `t8gen` solves 0/16.
3. **Side effects (held-out 96000-96031).** The edit breaks the other two instructions that say "left": T5, mustard in
   the left bin, **9/32 -> 0/32** (p = 0.0039), and T3, cube left of the bowl, **13/32 -> 0/32** (p = 2.4e-4). Elsewhere
   there is no significant change: R1 24 -> 28/32, T1 18 -> 18/32, T4 24 -> 22/32, T6 1 -> 1/32, T2 4 -> 1/16, T7 0 -> 0/16.
4. **Why, and what does not fix it.** RF and TF have equal token counts, so the distillation maps word onto word ("left"
   onto "right", "raisin" onto "mustard"). Preservation keys from 65 generic instructions (`t8und-p`) change nothing for
   T5 (practice 0/16). Keys from 24 left/right instructions of other tasks (`t8und-r`) keep T8 (13/16) and cut the offline
   deviation by 57-78%, but T5 (2/16) and T3 (1/16) stay broken. Mapping only the shared template positions
   (`t8und-s`) leaves T5 and T3 alone but also leaves T8 unfixed (1.9% of the gap). The wording's effect is carried by
   the instruction-word states themselves.

## Setup

| Item | Value |
|---|---|
| Model | `nvidia/Cosmos3-Edge-Policy-DROID` @ `a7c7288` (Hugging Face, not gated): 3.37B parameters, Nemotron-dense mixture of transformers, 28 layers, width 2048, MLP 9216 (squared ReLU), 16 query / 8 key-value heads |
| Serving | cosmos-framework `cf5d68c` (the commit the "Same Goal, Different Words" study pinned for Edge), `RobolabPolicyService`, venv `uv sync --all-extras --group=cu128-train --group=policy-server` (torch 2.10 + cu128; driver 535) |
| Sampling (the study's E3 settings) | UniPC, 4 steps, shift 5, classifier-free guidance 3 inside timesteps (960, 1001), prompt formatted as JSON, history 1, seed per request; `decode_video=False` (the predicted future video is not used) |
| Observation | RoboLab's `Cosmos3Client` composite: wrist view 360x640 on top of the two over-the-shoulder views at 180x320 each (540x640), plus 7 joint positions and the gripper position |
| Action mapping | 32 rows of [7 absolute joint-position targets (rad), gripper] per request, executed open loop (32 steps at 15 Hz = 2.1 s) by RoboLab's `DroidJointPositionActionCfg` (absolute `panda_joint1-7` targets, no offset; `finger_joint` binary: command > 0.5 closes to π/4 rad). The server flips the gripper convention in and out (`1 - g`), as the stock server does |
| Control rate | 15 Hz (sim dt 1/120 s, decimation 8) |
| Simulator | RoboLab v0.2.1 on Isaac Sim (`~/vibes/weight-space-policy-lab/robolab_checkout`), **default render preset** (primary; one performance-preset T3 run is kept as secondary sensitivity data in `results/secondary_performance_8env/`), three cameras (wrist, left and right shoulder), environment seed 0 |
| Envs | **8 per sim process**, two processes per 16-slot group, server and sim on the same RTX 3090 (24 GB; about 21 GB in use). 16 envs did not fit: beside the 9.2 GB unslimmed server PhysX ran out of memory (performance preset, first T3 run); beside the 6 GB slim server at the default preset every 16-env process died at reset with `CUDA out of memory` (00:30-00:37 ET, six jobs with 0 slots completed, kept in `~/cross/cosmos/runs/failed_oom16_default` and not counted) |
| Server memory | `edge_common.slim()` moves the parts action generation never reads off the GPU (lm_head 0.5 GB, token-embedding table 0.5 GB looked up on the CPU, video VAE decoder 1.0 GB): 5.6 GB allocated instead of 7.6 GB; actions are bit-identical with and without it (`scripts/mem_probe.py`, eager and compiled) |
| Determinism | replaying 5 recorded requests through the server code path on ali-ms gives bit-identical action chunks (`results/offline/ali-ms/replay_self_ali-ms.json`); the same 5 requests (recorded on GPU1) replayed on ali-ms GPU0 are also bit-identical (`results/offline/ali-ms/replay_gpu0_vs_gpu1.json`), and every paired comparison still runs both arms on one GPU (each run record carries `gpu`; records without it are GPU1).
The simulator is a different matter: rerunning T8-base-s96000 on GPU0 (`T8-detg0`) gives the same initial joints and
noise seeds, but the first rendered frame differs in 30-71% of pixels (mean about 1 intensity level). The first action chunks therefore
differ by up to 0.35 rad and the rollouts diverge. Both runs score 0/16 (`results/offline/ali-ms/sim_gpu0_vs_gpu1.json`).
"Paired" here means the same scene configuration, robot state and noise seeds per slot, not bit-identical pixels.
Whether same-GPU reruns render bit-identically was not tested. On the Korea pod (RTX PRO 4000 Blackwell, same lockfile) the same 5 requests differ (max abs 0.02-0.15, chunk-end end-effector 0.3-2.4 cm, gripper agreement 97-100%; `results/offline/korea-rtxpro4000/replay_vs_ali-ms.jsonl`), so results are paired only within one machine |
| Slots (shared with Tracks B and C) | practice and failure map 30000-30031; held-out 96000-96063. Slot s = 16 g + j: group g is the sampling-seed base, slot s gets the noise stream ((g * 64 + j) * 100 + k), k = replan index (weightsmith's scheme; with 8 envs per process an env offset keeps the seeds identical). Every process runs one episode per env, so every slot is the first episode of a fresh process |
| GPU | From 04:53 ET a second lane runs on ali-ms GPU0 (released by Track B; port 8502, the gpu0 flocks, `scripts/sessions/lane_gpu0.sh`): side effects on T5, T4, T6. Before that, GPU1 of ali-ms (RTX 3090), shared through the flocks `~/.cache/weightsmith/locks/gpu1.lock` and `~/vibes/opus-weight-surgery/state/gpu1.lock`; locks are released after every 16-slot job and the server is stopped with them. A Korea pod (4x RTX PRO 4000 Blackwell, GPUs 0-1 for this track) was set up at 01:00 ET (serving verified, see Determinism) but its SSH port went down at 06:20 UTC before any closed-loop job finished; nothing from it is counted |
| Statistics | `weightsmith.stats`: Wilson 95% intervals, exact two-sided McNemar on fixed/broke pairs, paired bootstrap (20,000 draws) |

## Tasks

| Code | RoboLab task | Default instruction | Horizon |
|---|---|---|---|
| T1 | RubiksCubeThenBananaTask | Put the cube then the banana in the bowl | 60 s |
| T2 | PickGlassesTask | Pick up the eye glasses | 30 s |
| T3 | RubiksCubeLeftOfBowlTask | Put the rubiks cube to the left of the bowl | 30 s |
| T4 | StackYellowOnRedTask | Stack the yellow block on the red block | 60 s |
| T5 | MustardInLeftBinTask | Put the mustard in the left bin | 30 s |
| T6 | Stack3RubiksCubeTask | Stack the rubiks cubes in a tower | 60 s |
| T7 | PickUpGreenObjectTask | Pick up the green vegetable block | 30 s |
| T8 | MustardRightOfRaisinRFTask (`tasks/`, from `../../sgdw/tasks/`) | Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle. | 40 s |
| R1 | RubiksCubeTask (retention) | Put the cube in the bowl | 40 s |

## Files

| Path | What |
|---|---|
| `scripts/edge_common.py`, `scripts/edge_server.py` | model loading with the E3 settings, seeded requests, weight edits applied at load (digests reported) |
| `scripts/cosmos_client.py`, `scripts/run_cosmos_batched.py` | RoboLab client (stock packing, batched requests, per-slot seeds, request recording) and sim runner |
| `scripts/pool.py` | queue runner: takes the GPU1 locks per 16-slot job, starts server and sim processes, writes one row per slot |
| `scripts/probe.py`, `scripts/capture.py` | model internals probe; replay of recorded decisions under wordings or edits with activation capture |
| `scripts/build_edit.py` | closed-form distillation (text or video/action pathway) and the matched random-direction control |
| `scripts/action_check.py` | action-mapping check and frame sheets from recorded requests |
| `analyze.py` | tables, `../metrics_cosmos.json` and plots from `results/runs/*.json` |

## Results

All numbers: ali-ms GPU1, default render, 8 envs per sim process, paired on the same starts. Regenerate with
`python3 analyze.py` after `scripts/sync_results.sh`; every number is in `../metrics_cosmos.json`.

### T8 (mustard right of raisin box, reference-first wording): the edit target

| Arm (slots 30000-30015, practice) | Success | Paired vs unedited |
|---|---|---|
| unedited, RF wording (task default) | 0/16 | |
| unedited, TF wording (known-good) | 15/16 | 15 fixed / 0 broke, p = 6.1e-5 |
| **edited `t8und`, RF wording** | **13/16** | 13 fixed / 0 broke, p = 2.4e-4; recovers 86.7% of the TF wording's gain |
| random-direction control of `t8und` (same rank and singular values per layer) | 1/16 | 1 fixed / 0 broke, p = 1 |

| Arm (slots 96000-96063, held-out: never used for any choice) | Success | Paired |
|---|---|---|
| unedited, RF wording | 0/64 (Wilson 95% 0.0-5.7%) | |
| **edited `t8und`, RF wording** | **56/64 (77.2-93.5%)** | vs unedited: 56 fixed / 0 broke, p = 2.8e-17; +87.5 points (paired bootstrap 95% 78.1-95.3) |
| random-direction control of `t8und` | 0/64 | vs edit: 56 fixed / 0 broke, p = 2.8e-17 |

Per 16-start group (edited): 15/16, 13/16, 14/16, 14/16; unedited and control 0/16 in every group.
Clip: `videos/t8_96003_unedited_vs_t8und.mp4` (held-out start 96003).

`t8und`: closed-form distillation of the TF wording's text-token states into the RF wording's, in the text pathway
(`mlp.down_proj` of layers 0-26, the weights the instruction tokens pass through), 171 matched positions plus 19
unconditional-pass keys that pin the empty prompt; lam_rel 1e-3; built from one prompt pair (the text states do
not depend on the image: max abs difference 0.0 across observations). Plot: `plots/edit_t8und_ali-ms.png`.
Clip: `videos/t8_30008_unedited_vs_t8und.mp4` (unedited picks up the raisin box; edited moves the mustard).

### Where to edit (offline, T8)

Replaying 30 RF-failure decisions held aside from every build (slots 30006-30011, replans 1-5) with the same seed,
the chunk-end end-effector distance to the TF-wording action chunk:

| Weights | Distance to TF actions | Gap closed | Its random control |
|---|---|---|---|
| unedited | 8.75 cm | | |
| text pathway `t8und` | 0.77 cm | 91.2% | -1.8% |
| video/action pathway `t8gen` (`mlp_moe_gen.down_proj`, layers 8-20) | 6.70 cm | 23.4% | 0.7% |

As on π0.5, the edit to the language pathway carries the wording's effect; the edit to the action-generation pathway,
built from 30 other decisions of the same task, carries a fifth of it. In closed loop on the practice starts the
video/action-pathway edit `t8gen` solves 0/16 (unedited 0/16, text-pathway `t8und` 13/16). `results/offline/ali-ms/t8_offline_check.json`.

### Clips (`videos/`, H.264, 2x speed, top = unedited, bottom = edited, same start)

| File | What it shows |
|---|---|
| `t8_96003_unedited_vs_t8und.mp4` | T8, held-out start 96003: the unedited policy fails under the RF wording; `t8und` moves the mustard to the right of the raisin box |
| `t8_30008_unedited_vs_t8und.mp4` | T8, practice start 30008: the unedited policy picks up the raisin box; `t8und` moves the mustard |
| `t8_30000_rf_vs_tf.mp4` | T8, practice start 30000: the unedited policy under the RF vs the TF wording (top RF, bottom TF) |
| `t5_96000_unedited_vs_t8und.mp4` | side effect, T5, held-out start 96000: unedited puts the mustard in the left bin; `t8und` does not |
| `t5_30001_unedited_vs_t8und.mp4` | side effect, T5, practice start 30001: `t8und` carries the mustard to the other bin |
| `t3_96004_unedited_vs_t8und.mp4` | side effect, T3, held-out start 96004: unedited puts the cube left of the bowl; `t8und` puts it elsewhere |

### Side effects of `t8und` on the other tasks (practice starts 30000-30015, each task's default instruction)

Each task is run unedited and with `t8und`, on the same 16 starts and the same GPU. Table, plots
(`plots/tasks_t8und_ali-ms.png`, `plots/side_effects_t8und_ali-ms.png`) and `../metrics_cosmos.json`
(`experiment: side_effect`) are regenerated as cells land. The map is the honest map of the edit that was scored on held-out starts.

The first cells show a clear pattern. The edit breaks the two other instructions that say "left": T5, "put the mustard in the left bin",
goes 8/16 -> 0/16 (8 broke, 0 fixed, p = 0.0078), and the edited policy carries the mustard to the other bin
(`videos/t5_30001_unedited_vs_t8und.mp4`). T3, "put the rubiks cube to the left of the bowl", goes 4/16 -> 0/16
(p = 0.125). Tasks without left/right are unchanged or slightly better (T1 10 -> 12/16, R1 9 -> 13/16, neither
significant). Distilling "the raisin box is to the left of the mustard" into "the mustard is to the right of
the raisin box" taught the text pathway something like left -> right, beyond the one sentence it was built from.

Held-out starts confirm the break. On slots 96000-96031, which no task except T8 had ever used, T5 goes 9/32 -> 0/32
(9 broke, 0 fixed, p = 0.0039) and T3 goes 13/32 -> 0/32 (13 broke, p = 2.4e-4) (`experiment: side_effect_heldout`;
plots `plots/tasks_heldout_t8und_ali-ms.png`, `plots/side_effects_heldout_t8und_ali-ms.png`). R1, T1, T4, T6, T2 and T7
on held-out starts are running.

Follow-up `t8und-p` (running): the same build plus zero-change preservation keys from 65 generic RoboLab instructions.
The set is fixed by rule before the build (`tasks/preserve_generic.json`): all RoboLab default instructions minus the shared-set
tasks, their near-duplicates and every instruction that mentions mustard, raisin, a bin, left or right. So a T5 recovery
cannot come from a hand-picked mirror instruction. It runs first on T5 and on T8 practice; held-out and the full map
follow only if it keeps T8 and restores T5.

#### `t8und-p` result offline, and the relation-aware variant `t8und-r` (selection rule fixed before its build, 07:58 ET)

Offline, `t8und-p` keeps T8 (it closes 93.1% of the RF -> TF action gap on the 30 held-aside decisions; `t8und`: 91.2%)
but leaves T5 where `t8und` put it. On 38 decisions from T5 practice episodes the unedited policy solved, the edited chunk
ends 13.9 cm from the unedited one (`t8und`: 13.7 cm). Generic preservation keys do not protect the semantic axis the
edit moves. Its closed-loop T5 and T8 practice checks are queued so the record is complete.

`t8und-r` is the same build with zero-change preservation keys from left/right spatial-relation instructions of
**other** RoboLab tasks only. Rule (`scripts/make_preserve_relation.py`, output `tasks/preserve_relation.json`): every
instruction variant (default / vague / specific / ...) of a RoboLab v0.2.1 task that contains "left" or "right", minus
every shared-set task (T1-T8, R1, all of their variants), minus any instruction with T3's (rubiks cube + bowl), T5's
(mustard + bin) or T8's (mustard + raisin box) object pair, minus anything mentioning mustard or raisin; duplicates
removed. So T3 and T5 stay unseen tests. It yields 24 instructions (bowls, spring clamp, hammers, tools, food
packing, mugs, spoon). A random-direction control with the same rank and singular values per layer is built with it.
Offline gates before any closed loop: T8 gap closed >= 85%, and the T5 and T3 deviations from the unedited policy
clearly below `t8und`'s. If it passes: T5 and T3 practice, then T8 practice. If those hold: held-out T5 and T3
(96000-96031) and T8 (96000-96063), each with the control.
(A broader targeted variant, `t8und-a`, preserving every non-shared-set RoboLab instruction including the mirror
"Put the mustard in the right bin", was started at 07:49 and stopped at 07:55 before its first layer; it is not used.)

**`t8und-r` passed its offline gates (08:59 ET; `results/offline/ali-ms/t8undr_gate.json`).** It closes 93.6% of the T8 RF -> TF gap
(its random control: -10.5%). On T5 decisions its chunks end 5.9 cm from the unedited policy's (`t8und`: 13.7 cm). On
20 decisions from T3 practice episodes the unedited policy solved, they end 6.5 cm away (`t8und`: 29.6 cm). Closed loop now:
T5 practice (GPU0), T3 and T8 practice (GPU1). A rerun of T8-base-s96000 on GPU0 (`T8-detg0`) tests whether
simulator outcomes are identical across the two GPUs, so the held-out runs of `t8und-r` can use both.
`t8und-p` in closed loop: T8 practice 12/16 (unedited 0/16), T5 practice 0/16 (unedited 8/16). It does not recover T5.

**`t8und-r` in closed loop (practice starts): it keeps T8 but does not repair T5 or T3.** T8 13/16 (unedited 0/16, `t8und` 13/16);
T5 2/16 (unedited 8/16, `t8und` 0/16; 6 broke, 0 fixed); T3 1/16 (unedited 4/16, `t8und` 0/16; 4 broke, 1 fixed).
Relation-aware preservation removes 57-78% of the offline action deviation on T5 and T3, which is not enough to keep
those tasks working. Per the plan, no held-out runs were made for it.

**`t8und-s` (approved 09:40 ET, building).** Hypothesis: the break comes from the word-for-word alignment. RF and TF have
equal token counts, so `--align all` maps "left" onto "right" and "raisin" onto "mustard". `t8und-s` maps only the shared
template prefix and suffix states (`--align ps`) and leaves the differing instruction words unconstrained, plus the 65
generic preservation prompts of `t8und-p` (the suffix tokens are identical in every prompt). Gates: T8 gap closed >= 85%
and T5/T3 deviation below `t8und-r`'s (5.88 / 6.46 cm). A T8 gap below 85% would itself say that the fix lives in the
word-aligned positions. Closed loop on T5, T3 and T8 practice runs only if both gates pass; held-out with control only if T5 and T3 then
recover with T8 held.

**`t8und-s` fails the T8 gate (10:42 ET; `results/offline/ali-ms/t8unds_gate.json`).** Mapping only the shared template
states closes 1.9% of the T8 RF -> TF gap (its random control: 2.6%; `t8und`: 91.2%), even though the build matches
those positions' residuals to within 5-6% at the last layers. It leaves T5 and T3 nearly untouched (deviation from the
unedited policy 2.0 cm and 0.8 cm; `t8und`: 13.7 and 29.6 cm). So the wording's effect on the actions is carried by
the instruction-word positions themselves. The video/action tokens read every text position's keys and values. To
change T8's behaviour, the edit has to move the word states, and moving "left" toward "right" is what spills into T5 and
T3. No closed loop for `t8und-s` (gate failed).

| Variant | Preservation keys | T8 gap closed (offline) | T5 dev. (cm) | T3 dev. (cm) | T8 practice | T5 practice | T3 practice |
|---|---|---|---|---|---|---|---|
| `t8und` (word-aligned) | none (uncond. pass only) | 91.2% | 13.7 | 29.6 | 13/16 | 0/16 | 0/16 |
| `t8und-p` | 65 generic RoboLab instructions | 93.1% | 13.9 | - | 12/16 | 0/16 | - |
| `t8und-r` | 24 left/right instructions of other tasks | 93.6% | 5.9 | 6.5 | 13/16 | 2/16 | 1/16 |
| `t8und-s` (template positions only) | 65 generic | 1.9% | 2.0 | 0.8 | - | - | - |
| unedited | | 0% | 0 | 0 | 0/16 | 8/16 | 4/16 |

(T5 and T3 deviation = chunk-end end-effector distance of the edited policy's action chunk from the unedited one, on 38 / 20
decisions of practice episodes the unedited policy solved. Practice = slots 30000-30015, paired with the unedited runs.)
