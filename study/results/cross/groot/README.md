# Track B: weight-space programming on NVIDIA GR00T N1.7 DROID (RoboLab)

Status: **in progress** (overnight run, 2026-11-07/08). This file is updated after each milestone; numbers appear only
once measured. Track A (Cosmos3 Edge) and Track C (π0.5 control) run the same tasks and slots. All tables are
regenerated from the raw run summaries by `scripts/analyze.py` into [`results/tables.md`](results/tables.md) and
[`../metrics_groot.json`](../metrics_groot.json); plots are in `plots/`.

Renderer preset: every Track B evaluation uses RoboLab's default preset (`--rendering-type` unset, which IsaacLab
resolves to `balanced`), the primary setting for all tracks. Performance-preset numbers are sensitivity data only.
Machine: every result so far is from ali-ms GPU0 (RTX 3090, label `ali-ms-rtx3090`); arms are paired only within one
machine.

## Results so far (numbers regenerated in [`results/tables.md`](results/tables.md))

**Failure map** (unedited, each task's own wording, slots 30000-30031): T1 cube-then-banana 9/32, T3 cube left of bowl
1/32, T8 mustard right of raisin box (reference-first) 0/32, R1 cube in bowl 28/32; T2, T5, T7, T4, T6 running.

**Wording screens** (same slots, paired): T1 9/32 -> 16/32 under "Put the cube in the bowl then put the banana in the
bowl" (11 fixed / 4 broke, exact p = 0.12), GR00T's only wording-sensitive task. T3 1/32 -> 0/32 under "Pick up the
rubiks cube and place it on the table to the left of the bowl"; T8 0/32 -> 0/32 under the target-first wording. In T3
GR00T grasps the cube and holds it until the episode ends; in T8 it grasps the mustard (always under TF) but never ends
with it right of the raisin box. These are execution failures that rewording does not change, unlike Cosmos3 Edge on
T8 (Track A).

**Backbone edit on T1 (negative).** The closed-form distillation of the explicit wording into Cosmos-Reason2's LLM
MLPs (layers 0-15, `mlp.down_proj`; 6 build decisions; details in `results/edits.json`) recovered 0.28 of the teacher's
action change offline on 17 check decisions (matched random control 0.00), but on practice it scored **0/32 vs 9/32
unedited (0 fixed / 9 broke, exact p = 0.0039)**: every edited episode reaches for the banana first and never grabs the
cube (clips `videos/RubiksCubeThenBananaTask_30004_{unedited,edited-bb,teacher-explicit}.mp4`). On the 11 failed
episodes used by no build the offline recovery is -0.15, so the 0.28 did not predict transfer. Token alignment covers
only the 8 original positions after the first difference; the explicit wording's inserted clause ("in the bowl then
put") has no position in the original prompt, so the transferred change at "the banana in the bowl" arrives without
the cube clause. The same distillation in the action head (32 DiT blocks, `ff.net.2`) recovers +0.22 on those 11
episodes, yet it breaks the same way closed loop: **0/32 vs 9/32 (0 fixed / 9 broke, p = 0.0039)**, 31/32 failing to
grab the cube (clip `..._edited-dit.mp4`). A grasp-phase offline check, added after the closed-loop result, shows the
problem both checks above missed: on 16 early decisions (k = 4-16 of slots 30016-30019, none used to build) the
explicit wording moves the executed end-effector target by only 0.88 cm, the backbone edit by 1.86 cm in another
direction (pinning every image token to zero changes nothing: 1.84 cm), a span-aligned variant by 3.5 cm. Captured
at 4 of those early decisions (k = 4), the explicit wording's change at the 8 aligned positions is the same as at the
banana-phase build decisions (median per-layer cosine to the build mean 0.93, norm ratio 1.03;
`results/wsp/t1/early/value_similarity.json`). So the distilled change is near-constant and acts from the first
replan. In the teacher, the inserted clause tokens, which the action head also reads, keep the start-of-episode actions
almost unchanged; the edit transfers only the aligned part, so the robot's first reach moves toward the banana.
Same-size random controls for both locations are queued to separate "this direction" from "any change of this size".

Clips (same start, slot 30004, H.264, 2x speed): unedited succeeds (cube, then banana); the teacher wording fails on
this start (banana not grabbed); both edits fail to grab the cube.

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
