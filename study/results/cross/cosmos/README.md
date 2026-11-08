# Track A: weight-space programming on Cosmos3 Edge Policy DROID

Status (2026-11-08 04:30 ET): closed loop runs on ali-ms at RoboLab's default render preset (the primary setting for
every model and track). Done: T8 practice (unedited / TF wording / edit / control) and T8 held-out (64 starts x
unedited / edit / control). Running now: the side-effect map (t8und vs unedited on R1 and T1-T7, which is also the
failure map), then the video/action-pathway edit in closed loop. Every number here comes from `results/` and `../metrics_cosmos.json` (run `analyze.py`); the
latest table is in [Results](#results).

Question: how well does weight-space programming (WSP, closed-form instruction distillation) fix the failures of
NVIDIA's Cosmos3 Edge Policy DROID, against GR00T (Track B) and π0.5 (Track C, the reference) on the same RoboLab
tasks and slots?

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
| Determinism | replaying 5 recorded requests through the server code path on ali-ms gives bit-identical action chunks (`results/offline/ali-ms/replay_self_ali-ms.json`); the same 5 requests (recorded on GPU1) replayed on ali-ms GPU0 are also bit-identical (`results/offline/ali-ms/replay_gpu0_vs_gpu1.json`), and every paired comparison still runs both arms on one GPU (each run record carries `gpu`). On the Korea pod (RTX PRO 4000 Blackwell, same lockfile) the same 5 requests differ (max abs 0.02-0.15, chunk-end end-effector 0.3-2.4 cm, gripper agreement 97-100%; `results/offline/korea-rtxpro4000/replay_vs_ali-ms.jsonl`), so results are paired only within one machine |
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
built from 30 other decisions of the same task, carries a fifth of it. `results/offline/ali-ms/t8_offline_check.json`.
