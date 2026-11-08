# Track B: weight-space programming on NVIDIA GR00T N1.7 DROID (RoboLab)

Status: **overnight run complete** (2026-11-07/08, ali-ms GPU0, RTX 3090, label `ali-ms-rtx3090`; every arm paired on the
same machine). All tables are regenerated from the raw per-run summaries (`results/runs/*.json`) by
`scripts/analyze.py` into [`results/tables.md`](results/tables.md) and [`../metrics_groot.json`](../metrics_groot.json);
plots in `plots/` (SVG + PNG at 2x; grey = unedited, teal = edited, light teal = alternative wording, light grey =
random control); clips in `videos/`. Renderer preset: RoboLab's default (`balanced`) everywhere, the primary setting for
every track.

## Headline

1. **GR00T fails most of the shared tasks, and its failures are execution failures.** Unedited success on the paired
   practice starts: T1 9/32, T2 0/32, T3 1/32, T4 0/16, T5 0/32, T6 0/16, T7 0/32, T8 0/32; retention task R1 28/32.
   NVIDIA's unmodified server + client also scores 0/16 on T7 (and 16/16 on BananaOnPlate, as this adapter), so the
   zeros are the model's, not the adapter's.
2. **Rewording rescues none of the failures except, weakly, T1.** T1 9/32 -> 16/32 under the explicit wording (11
   fixed / 4 broke, exact p = 0.12); T3 1/32 -> 0/32; T8 0/32 -> 0/32 under the target-first wording that rescues
   Cosmos3 Edge (Track A). In T3 GR00T grasps the cube and holds it; in T8 it grasps the mustard (22/32 episodes under RF, 32/32 under TF)
   but never ends with it right of the raisin box.
3. **Distilling T1's explicit wording breaks GR00T in both edit locations.** Backbone edit (Cosmos-Reason2 LLM MLPs)
   0/32 and action-head edit (DiT MLPs) 0/32, vs 9/32 unedited (0 fixed / 9 broke each, exact p = 0.0039); the
   same-size random-direction controls score 11/32 (backbone) and 10/32 (action head), so the break is the distilled
   direction, not the size of the change (edited vs control, paired: 0 / 11, p = 9.8e-4; 0 / 10,
   p = 0.002). The edited robot fails at the first grasp. The cause shows offline: the wording's internal change is near-constant over the episode, but in the
   teacher the inserted clause keeps the start-of-episode actions unchanged; the edit transfers the change without the
   clause, so it acts from the first reach. No held-out or side-effect runs were made: neither edit passed practice.

![Failure map](plots/failure_map.png)
![Wording screens](plots/wording_screens.png)
![Where to edit, T1](plots/where_to_edit_T1.png)
![Offline checks, T1](plots/offline_checks_T1.png)

## Failure map (metric 1)

Unedited, each task's own instruction, slots from 30000 (32 starts; 16 for the 60-s stacking tasks T4 and T6), Wilson
95% intervals in `results/tables.md`. How the failed episodes end (RoboLab's failure reason):

| Task | Success | How failures end |
|---|---|---|
| T1 cube then banana | 9/32 | banana never grabbed 18, cube never grabbed 4, other 1 |
| T2 pick up glasses | 0/32 | glasses never lifted above the table 30, never grabbed 2 |
| T3 cube left of bowl | 1/32 | cube grasped and held, never left of the bowl 30, never grabbed 1 |
| T4 stack yellow on red | 0/16 | yellow block never grabbed 16 |
| T5 mustard in left bin | 0/32 | mustard never grabbed 32 |
| T6 stack three cubes | 0/16 | no stack completed 16 |
| T7 pick up green block | 0/32 | block never lifted 32 (official path: 0/16) |
| T8 mustard right of raisin box (RF) | 0/32 | mustard not right of the raisin box 22, never grabbed 10 |
| R1 cube in bowl | 28/32 | cube never grabbed 3, not in bowl 1 |

## Wording screens (metric 2)

Unedited, paired on slots 30000-30031, exact McNemar on the discordant pairs:

| Task | Own wording | Alternative | Own | Alt | alt only / own only | p |
|---|---|---|---|---|---|---|
| T1 | Put the cube then the banana in the bowl | Put the cube in the bowl then put the banana in the bowl | 9/32 | 16/32 | 11 / 4 | 0.12 |
| T3 | Put the rubiks cube to the left of the bowl | Pick up the rubiks cube and place it on the table to the left of the bowl | 1/32 | 0/32 | 0 / 1 | 1 |
| T8 | ... so that the raisin box is to the left of the mustard bottle. (RF) | ... so that the mustard bottle is to the right of the raisin box. (TF) | 0/32 | 0/32 | 0 / 0 | 1 |

Under TF all 32 T8 episodes grasp the mustard (under RF 10 never do), but none places it. For scale: a same-size
random weight perturbation flips 16 of 32 T1 outcomes (9 / 7, below), so T1's 11 / 4 is a weak signal.

## Closed-form distillation on T1 (metric 3) and where to edit

Built from the 6 build decisions (the replan where the two wordings' replays diverge most, in each of the first 6
failed practice episodes); dW = K^T (K K^T + lam I)^-1 V per layer, lam = 1e-3 mean diag(K K^T); the random control has
the same singular values along random orthonormal directions (`default_rng(0)`). Fingerprints (sha256[:16]): bb `b5523fe55b6d94a2`, bb control `1d2b485f3c0d1321`, dit `a998bdc75628fbe6`, dit control
`afc8a78428b9ec07`. Details:
`results/edits.json`, `results/wsp/t1/*_build.json`, and the edit report inside each run file.

| Arm (practice slots 30000-30031) | Location | Values changed | Success | vs unedited (fixed / broke, p) |
|---|---|---|---|---|
| unedited | | | 9/32 | |
| explicit wording (teacher) | | | 16/32 | 11 / 4, p = 0.12 |
| backbone edit `bb` | Cosmos-Reason2 LLM layers 0-15, `mlp.down_proj` (rank 8 each) | 167,481,274 | **0/32** | 0 / 9, p = 0.0039 |
| backbone random control | same layers, same singular values | 179,039,148 | 11/32 | 9 / 7, p = 0.80 |
| action-head edit `dit` | 32 AlternateVLDiT blocks, `ff.net.2` (rank 164 each) | 298,523,070 | **0/32** | 0 / 9, p = 0.0039 |
| action-head random control | same blocks, same singular values | 300,072,426 | 10/32 | 9 / 8, p = 1 |

Edited vs its own control, paired: backbone 0 vs 11 (p = 9.8e-4); action head 0 vs 10 (p = 0.002).

What the edited robot does (clips, same start): `videos/RubiksCubeThenBananaTask_30004_*.mp4` (unedited succeeds,
cube then banana; the teacher wording fails on this start; the backbone edit reaches for the banana first and never
grabs the cube; the DiT edit never grabs the cube) and `..._30003_*.mp4` (unedited fails at the banana, the teacher
wording succeeds, both edits fail at the cube). Wording clips: `RubiksCubeLeftOfBowlTask_30000_*` and
`MustardRightOfRaisinRFTask_30000_*` (both wordings fail the same way).

**Clips** (`videos/`, H.264, 960x270, exterior | wrist camera, 2x speed, no audio; same start within each slot; outcomes from the run files):

| Clip | Run | Outcome | Size |
|---|---|---|---|
| `MustardRightOfRaisinRFTask_30000_reference-first.mp4` | fm-T8 | fail: object_right_of(object=mustard_bottle, reference_object=raisin_box) (step 600) | 0.84 MB |
| `MustardRightOfRaisinRFTask_30000_target-first.mp4` | wd-T8-tf | fail: object_right_of(object=mustard_bottle, reference_object=raisin_box) (step 600) | 0.57 MB |
| `RubiksCubeLeftOfBowlTask_30000_original.mp4` | fm-T3 | fail: object_left_of(object=rubiks_cube, reference_object=bowl) (step 450) | 0.40 MB |
| `RubiksCubeLeftOfBowlTask_30000_place-on-table.mp4` | wd-T3-alt | fail: object_left_of(object=rubiks_cube, reference_object=bowl) (step 450) | 0.43 MB |
| `RubiksCubeThenBananaTask_30003_edited-bb.mp4` | ed-bb-T1 | fail: object_grabbed(object=rubiks_cube) (step 900) | 0.70 MB |
| `RubiksCubeThenBananaTask_30003_edited-dit.mp4` | ed-dit-T1 | fail: object_grabbed(object=rubiks_cube) (step 900) | 0.63 MB |
| `RubiksCubeThenBananaTask_30003_teacher-explicit.mp4` | wd-T1-expl | success (step 326) | 0.41 MB |
| `RubiksCubeThenBananaTask_30003_unedited.mp4` | fm-T1 | fail: object_grabbed(object=banana) (step 900) | 1.08 MB |
| `RubiksCubeThenBananaTask_30004_edited-bb.mp4` | ed-bb-T1 | fail: object_grabbed(object=rubiks_cube) (step 900) | 0.82 MB |
| `RubiksCubeThenBananaTask_30004_edited-dit.mp4` | ed-dit-T1 | fail: object_grabbed(object=rubiks_cube) (step 900) | 0.53 MB |
| `RubiksCubeThenBananaTask_30004_random-control-bb.mp4` | ctl-bb-T1 | success (step 346) | 0.43 MB |
| `RubiksCubeThenBananaTask_30004_random-control-dit.mp4` | ctl-dit-T1 | fail: object_grabbed(object=banana) (step 900) | 0.65 MB |
| `RubiksCubeThenBananaTask_30004_teacher-explicit.mp4` | wd-T1-expl | fail: object_grabbed(object=banana) (step 900) | 0.72 MB |
| `RubiksCubeThenBananaTask_30004_unedited.mp4` | fm-T1 | success (step 304) | 0.37 MB |

**Offline checks** (recorded decisions, images, state and noise seed fixed; `results/wsp/t1/`, plot
`plots/offline_checks_T1.png`). Before the closed-loop run the backbone edit recovered 0.28 of the teacher's chunk change
on 17 check decisions from other failed episodes (random control 0.00) and the DiT edit 0.21. Two checks added after
the closed-loop failures show what that missed:

| Edit | Grasp phase: EE-target shift vs unedited (16 decisions; teacher 0.88 cm) | Grasp phase: gripper rows = teacher's (unedited 121/128) | Unseen failures: teacher projection (11 decisions) |
|---|---|---|---|
| backbone `bb` (run closed loop) | 1.86 cm | 120/128 | -0.15 |
| backbone, every image token pinned to zero | 1.84 cm | 120/128 | -0.17 |
| backbone, span-aligned values | 3.50 cm | 119/128 | -0.19 |
| action head `dit` (run closed loop) | 1.21 cm | 106/128 | +0.22 |
| backbone random control | | | 0.00 |

Grasp phase = k 4, 8, 12, 16 of slots 30016-30019; unseen failures = the 11 failed practice episodes used by neither
the 6-decision nor a 12-decision build (a 12-decision build gives -0.17 there, so more decisions do not help).
At 4 grasp-phase decisions the explicit wording's change at the 8 aligned prompt positions matches the banana-phase
build decisions (median per-layer cosine 0.93 to the build mean, norm ratio 1.03; `early/value_similarity.json`): the
change is near-constant over the episode. In the teacher, the inserted clause ("in the bowl then put"), which has no
position in the original prompt but is read by the action head, keeps the start-of-episode actions almost unchanged.
The backbone edit transfers the aligned part without it and moves the first reach (toward the banana). The DiT edit
mostly flips gripper calls in the grasp phase (106/128 rows agree with the teacher vs 121/128 unedited). Pinning the
image tokens rules out leakage into the visual tokens.

Not run: held-out scoring (96000-96063) and side-effect maps, because neither edit passed practice (the coordinator's
call); no further T1 variants.

## GPU time

ali-ms GPU0 with the flock held: about 4.7 h measured: 15,481 s of closed-loop evaluation (18 runs), 981 s of offline GPU jobs (capture,
replays, the official-path T7 check), about 330 s for the official-path BananaOnPlate check, plus a few minutes of untimed
setup smoke tests. Offline activation work (variants, grasp-phase and span checks)
ran on CPU only.

## Run scheme

One GPU lane (`scripts/worker.sh GPU PORT`) runs jobs from an editable queue (`results/jobs/queue.txt` is the copy
pulled from the host), one at a time, holding the GPU flocks for each job. An evaluation job is one
`scripts/pool_groot.py` call (one server, one fresh Isaac process per 16-slot group); a `cmd` job runs an offline step
(activation capture, edit build, offline check) under `scripts/locked.sh`. Gated jobs (`after:...`) wait for earlier
jobs or for a gate opened by hand after an offline check. Practice / design slots 30000-30031; held-out slots
96000-96063, used only for the final scoring of an edit fixed beforehand.

## Setup

| Item | Value |
|---|---|
| Checkpoint | `nvidia/GR00T-N1.7-DROID`, revision `05e7cc97e40dbd33b0890c35cc0214fcb0547ab5` (official, ungated; 3,144,016,000 parameters) |
| VLM backbone | `nvidia/Cosmos-Reason2-2B` (Qwen3-VL, gated; access OK), truncated to 16 LLM layers (`select_layer` 16), hidden 2048, MLP 6144 |
| Action head | 4-layer VL self-attention (201M) + 32-block AlternateVLDiT (1.09B), flow matching, 4 denoising steps, 40-step chunks |
| Code | Isaac-GR00T `9fbe86a` (main, 2026-11-07) in its own venv (`uv sync --python 3.12`, torch 2.9.0+cu128) |
| Embodiment tag | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` (the validated RoboLab setting) |
| RoboLab | weightsmith's pinned checkout (v0.2.1 `0aef241` + 6 weightsmith patches), Isaac Sim 5.0; its `policies/gr00t/client.py` is NVIDIA's validated N1.7 client and matches upstream main except for instruction-type plumbing |
| Host / GPU | ali-ms, GPU0 (RTX 3090 24 GB): GR00T server (about 7.4 GB, micro-batches of 4) and one Isaac process with **16 envs** (15.6 GB, default renderer preset) |
| Control | 15 Hz; open-loop horizon 8 (8 of 40 predicted rows executed per replan, the validated setting) |
| Seeds | RoboLab environment seed 0; slot s = 16·g + env, g = sampling-seed group; every 16-slot group is the first and only run of a fresh simulator process; noise seed of env e at replan k in group g = ((g·64)+e)·100+k (as weightsmith's π0.5 client) |
| Practice / failure-map slots | 30000–30031 (shared with Tracks A and C) |
| Held-out slots | 96000–96063 (never used for design) |

### Observation and action mapping

The adapter keeps NVIDIA's validated RoboLab contract and only batches it:

| GR00T key | RoboLab source | Format |
|---|---|---|
| `video.exterior_image_1_left` | `image_obs/over_shoulder_left_camera` (1280×720) | HWC uint8 resized to 180×320 by INTER_AREA, no letterbox (`[B,1,180,320,3]`) |
| `video.wrist_image_left` | `image_obs/wrist_cam` (1280×720) | same |
| `state.joint_position` | `proprio_obs/arm_joint_pos` | 7 rad |
| `state.gripper_position` | `proprio_obs/gripper_pos` | 1, 0 = open |
| `state.eef_9d` | `proprio_obs/ee_pos`, `ee_quat` (base frame) | xyz + first two rows of R·R_corr (rot6d), R_corr = [[0,0,-1],[-1,0,0],[0,1,0]] |
| `annotation.language.language_instruction` | task instruction or `OWSP_PROMPT` override | string (the processor lowercases it and strips punctuation) |

Actions: the server decodes `action.joint_position` (relative, converted back to absolute with the current state)
and `action.gripper_position` (absolute), 40 rows each; the client executes rows 0–7 as RoboLab DROID joint-position
targets, with the gripper binarised at 0.5 (1 = close). Checked on recorded episodes: the first executed row lies within
0.01–0.09 rad of the measured joints (absolute targets, not deltas); successful grasps show the commanded gripper at 1
and the measured gripper rising to 0.4–0.83.

Files: `adapters/groot_server.py` (server: Gr00tPolicy + Gr00tSimPolicyWrapper, per-slot seeded noise, closed-form
edits applied at load), `adapters/groot_batched_client.py` + `adapters/run_groot_batched.py` (RoboLab side, one batched
request per sim step), `adapters/groot_acts.py` (activation capture on recorded decisions), `scripts/pool_groot.py`
(one evaluation: GPU0 flocks, server, one fresh Isaac process per 16-slot group, per-slot rows),
`scripts/groot_build.py` (closed-form distillation and the matched random control), `scripts/analyze.py` (tables,
`../metrics_groot.json`, plots).

Seeded-noise checks (server, synthetic inputs): equal seeds give identical chunks; changing one sample's seed changes
only that sample; the same sample in a batch of 1 and a batch of 3 differs by at most 0.01 rad (GPU batch numerics).
Batched latency: 0.78–0.85 s for 16 envs; peak server memory 8.1 GB.

## Validation against NVIDIA's published baseline: the renderer preset

NVIDIA reports 40/40 on `BananaOnPlateTask` for this checkpoint in RoboLab (Isaac-GR00T `examples/RoboLab`). Three runs on
the same 16 starts (slots 0–15, 16 envs, environment seed 0):

| Run | Server / client | Renderer preset | Success | Successful episodes end at |
|---|---|---|---|---|
| this adapter | `groot_server.py` + batched client | `--rendering-type performance` (weightsmith's setting, used by Tracks A and C) | **5/16** | steps 309–502 |
| NVIDIA's official path | unmodified `run_gr00t_server.py --use-sim-policy-wrapper` + RoboLab `policies/gr00t/run.py` (serial) | default (unset) | **16/16** | steps 127–167 |
| this adapter | `groot_server.py` + batched client (server micro-batch 4) | default (unset) | **16/16** | steps 126–151 (one at 384) |

The adapter reproduces the official path; the gap is the renderer preset. The `performance` preset renders without the
RTX denoiser, so every surface carries heavy speckle noise (below: the first frame of slot 0, exterior and wrist
cameras, performance on top, default below). GR00T N1.7 DROID loses most grasps under that noise (11/16 never grasp the
banana; the commanded gripper flickers within chunks while the measured gripper stays below 0.11); π0.5 had tolerated
it in the weightsmith runs. **All Track B evaluations therefore use the default preset**, NVIDIA's validated
configuration, with the shared slots, environment seed and process scheme (the default preset is the primary setting
for every track). The default preset needs 15.6 GB for 16 Isaac envs, so the server splits each request into micro-batches
of 4 (same per-slot seeds; the action chunks are the same up to GPU numerics).

![Renderer presets](plots/validation_render_preset_step0.png)

Raw files: `results/validation/smoke-bop-s0.json` (performance), `results/validation/official-bop16.episode_results.jsonl`
(official path), `results/validation/smoke-bop-s0-default.json` (adapter, default preset).

A second official-path run on a task the adapter fails, T7 `PickUpGreenObjectTask` (16 envs, RoboLab's own starts,
default preset, `scripts/official_path.sh`), also scores **0/16**, every episode ending with the block never lifted, as
in the adapter's 0/32 (`results/validation/official-t7-16.episode_results.jsonl`).
