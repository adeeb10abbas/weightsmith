# Where should the edit go? VLM, action expert, or action head

Status: pre-registration (written 2026-11-07, before any closed-loop run of this study). Results are added below
this section later; this section is not edited after the first closed-loop run starts.

## Question

The paper's closed-form instruction distillation (Section 2) copies the explicit wording's per-layer residual change
into the VLM's MLP down-projections. Does the location matter? We apply the same distillation, built from the same
6 recorded premature-release decisions and the same teacher (the unedited policy under the explicit wording
"Put the cube in the bowl then put the banana in the bowl"), to different parts of pi0.5:

| Arm | Location | Construction |
|---|---|---|
| V1-full | VLM, layers 0-16 | the paper's `v2mi-f` (byte copy, fingerprint `63db0c52a51306c9`) |
| V1-0.16% | VLM, image cells, layers 8-16 | the paper's `abl-imgl8` (byte copy, `5bf8c4329c1e416a`) |
| A1 | action expert, `action.blocks.00-17.mlp_1.linear` | same principle on the action expert (below) |
| H1 | action head, `action.action_out_proj.kernel` | least-norm map of the original wording's head input to the explicit wording's output (below) |
| P1 (reference) | action head, gripper column | the blind editor's gated rank-1 edit `ctrl-ctb/cands/p1` (byte copy, `4ca8023b0592b79f`) |
| VA | VLM + action expert | A1's construction computed on the V1-full model and added to it |

All arms live in the new project `projects/where-to-edit` on the GPU host; the copied candidates were checked
against their source fingerprints (`owsp` fingerprint = sha256 over the sorted edit files).

### A1 construction (decided before any result)

- Keys: the input of `action.blocks.LL.mlp_1.linear` (the gated-GELU hidden, 4096-d) under the original wording,
  averaged over the 6 decisions, one key per action token (15) and per denoise step (10): 150 keys per layer.
  Separate keys per step because the action expert's MLP input changes strongly with the flow-matching time (adaRMS
  time modulation and the noisy action x_t), so a key averaged over steps matches no real input; all 10 steps
  contribute equally to the final chunk (x_0 = noise - 0.1 sum_s v_s), so none is dropped. Separate keys per token
  because the action tokens are the action expert's positions, the analogue of the VLM edit's aligned positions.
- Values: the per-layer residual increment under the explicit wording minus that under the original,
  V_L = d_L - d_{L-1}, d_L = mean over decisions of (explicit - original) residual after block L. Both wordings are
  replayed with the same images, state and noise (`owsp acts` with `action_layers` 0-17, `denoise_steps` 0-9, all
  15 action tokens). Because the explicit wording changes the velocity, the noisy actions x_t differ from denoise
  step 1 on, so the block-0 input differs too; d_{-1} is that input difference, computed exactly from the captured
  head inputs: x_s^x - x_s^o = -0.1 sum_{j<s} (h_j^x - h_j^o) W_out, embedded by `action_in_proj`.
- Update: one closed-form ridge solve per layer with lambda = 1e-3 tr(KK^T)/n, n = 150. One difference from the VLM is
  unavoidable: in the action expert the MLP output is multiplied by the adaRMS gate of `pre_ffw_norm_1` before it is
  added to the residual (residual += g_{L,s} * (h W)), with |g| below 0.05 for 27-99% of the dimensions depending on
  the layer. Eq. (1) assumes an ungated residual. A1 therefore solves the same ridge objective through the gate,
  min ||g * (K dW) - V||^2 + lambda ||dW||^2, whose solution is dW = K^T Z with Z_j = D_j (D_j G D_j + lambda I)^-1 V_j
  for every output column j (G = K K^T, D_j = the gates of column j at each key's step). With all gates equal to 1
  this is exactly Eq. (1). The gates are a function of the denoise time only and are computed from the weights
  (`time_mlp_in/out`, `pre_ffw_norm_1.Dense_0`); the computation is checked by rebuilding the captured head input
  from the layer-17 residual with the final adaRMS norm (relative error 0.3%). The literal Eq. (1), ignoring the gate
  (`a1-nogate`), is built as an offline control only.

### H1 construction

Keys: the head input h (after the final adaRMS norm, 1024-d) under the original wording, averaged over the 6 decisions,
per denoise step and action token (150 keys). Values: the explicit wording's velocity minus the original's,
(h^x - h^o) W_out, in the 8 dimensions the robot uses (7 joints, gripper). dW = K^T (K K^T + lambda I)^-1 V on those 8
kernel columns, lambda as above; bias unchanged; rank at most 8. All 8 used dimensions rather than only the gripper,
because distillation copies the teacher's output, not a hand-picked part of it, and the VLM and action-expert arms also
change the arm's motion; the 24 padded dimensions are dropped by the output transform and left alone. The gripper-only
variant (`h1-grip`, rank 1, the same size as P1) is built as an offline comparison.

### VA construction

The VLM edit is V1-full. Keys and the student side of the values are captured on the V1-full model under the original
wording (capture `va-orig`); the teacher side is the unedited model under the explicit wording (the same teacher as
every arm). The action-expert update (A1's gated solve) is added to V1-full's edits, so VA closes the gap that remains
in the action expert after the VLM edit.

## Hypothesis and predictions (pre-registered)

Main hypothesis (given by the study owner, adopted here): VLM edits transfer to other tasks, action-side edits do not.
The reasoning: the VLM edit changes how the instruction is read into the image tokens that the action expert consults,
a representation shared across tasks; action-side edits are keyed to the action expert's internal states at this task's
premature releases.

Predictions, written after building the arms and checking only the V1 and P1 arms offline (V1-full 0/21 releases,
V1-0.16% 2/21, P1 0/21, all with unchanged carries and phase-two decisions; this reproduces the paper and the blind
editor), before the offline check of A1, H1 and VA and before any closed-loop run:

1. Held-out (64 new slots 94016-94079, all arms paired with one base run): V1-full and V1-0.16% give large gains
   (base about 10/64, edits about 45-55/64) and do not differ significantly from each other.
2. P1 gives a moderate gain (about +15 to +25 points), smaller than V1.
3. A1: the action expert's own residual stream carries the decision (the explicit-minus-original difference grows from
   layer 13 to 16 and reaches 40% of the residual norm), so reproducing it needs updates as large as the weights
   (|dW|/|W| of 0.9-2.2 in layers 4-16 in the build). Prediction: such an update disturbs other decisions; closed-loop
   gain smaller than V1, possibly none.
4. H1: a rank-8 readout shift is nearly the same for every state the head sees, so it should flip the premature
   releases offline but also close the gripper where it should stay open (phase-two approach); closed-loop gain
   smaller than P1, possibly none or negative.
5. VA: no significant difference from V1-full (the VLM edit already recovers most of the explicit wording's effect).
6. Transfer (32 paired slots each, arms that fix the target task): V1 arms raise PickGlassesTask and
   RubiksCubeLeftOfBowlTask as before; action-side arms (P1, A1, H1) do not raise them significantly. Retention
   (cube into bowl, RubiksCubeTask): no arm is predicted to lower it, though an ungated gripper-hold edit could.

Analysis plan: successes per arm; fixed/broke against the shared base; exact two-sided McNemar test on the discordant
pairs; paired bootstrap 95% interval of the difference (20,000 resamples of slots, seed 0), as in
`study/results/opus/controls/ctl_analyze.py`. Main comparisons: each arm vs base; A1, H1, P1, VA vs V1-full;
V1-0.16% vs V1-full. No correction is applied for the number of arms; p-values are reported as they are.

### Deviation noted during the run (added 2026-11-07 15:30, before any arm other than base had finished)

`ctl_run.py` takes the second GPU only if it is free when a chunk starts. Other projects share the GPUs, so some
32-slot chunks run on one lane: both groups of 16 slots then run in one simulator process, the second group as that
process's second episode. The first observations show that a process's second episode starts from a different initial
scene than its first (mean absolute wrist-image difference 4.4 grey levels, as large as the difference between two
environments, against 1.3 for the same slot in two processes). The first group of each chunk (slots 94016-94031 and
94048-94063) is the first episode of lane 0 in every case, so it is identical across arms. Pre-specified handling: the
primary analysis uses all 64 slots as planned; a sensitivity analysis repeats every comparison on those 32 slots. The
lane count of every chunk is recorded. (Stopping the runner to enforce two lanes was not possible in this session.)

### Transfer plan (added 2026-11-07 15:36; only base and V1-full had finished one 32-slot chunk of the held-out)

Transfer and retention runs use `where_run.py` (a copy of `ctl_run.py` that holds both evaluation GPUs for every chunk,
so all slots are first episodes of their process) on fresh slots 94016-94047 in new projects `where-glasses`
(PickGlassesTask), `where-left` (RubiksCubeLeftOfBowlTask) and `where-cube` (RubiksCubeTask, "Put the cube in the bowl";
retention), with the unedited policy under the task's own instruction as the shared base. Arms: V1-full, P1 and A1, one
per location; H1 is added if its held-out difference from base has a bootstrap interval above zero; VA and V1-0.16% only
if GPU time remains.

### Deviation: transfer runner lock pattern (added 2026-11-07 15:50)

`where_run.py --all-lanes` takes the two GPU locks in a fixed order and blocks on each, so while it waits for GPU1 it
holds GPU0 idle. Launched at 15:34 to interleave with the held-out run, it made later held-out chunks run on one lane
and delayed other projects' GPU0 jobs while it waited. Stopping it was not permitted in this session, so it runs to the
end (3 tasks x 4 arms x 32 slots). The pre-specified sensitivity analysis above covers the one-lane held-out chunks.

## Results (added after the runs)

Every number below is read from the JSON files in `results/` (`write_results.py`, `make_tables.py`).

**Summary.** On 64 new held-out slots, all arms paired with one unedited run (5/64): V1-full 55/64, VA 43/64, V1-0.16% 41/64, P1 30/64, A1 20/64, H1 0/64. Every arm except H1 beat the unedited policy, and every arm was below V1-full (all p <= 0.017). Only the VLM edit transferred: eyeglasses 25 vs 14 of 32, cube left of bowl 20 vs 4 of 32; the action-side edits that fixed the target task (P1, A1) raised neither (all p >= 0.38), and no arm changed the retention task (cube into bowl). The fix lives in the VLM's reading of the instruction; copied into the action expert it needs updates as large as the weights and closes the gripper at the start of half the episodes, and copied into the action head without a gate it closes the gripper almost always.

### Offline: the paper's 49 recorded decisions

Replays with images, robot state and flow-matching noise fixed (`check49.py`; a chunk counts as a release when most of its 15 gripper commands are below 0.5). The 21 premature releases include the 6 distilled decisions; the next column counts only the other 15. Phase two: 7 approaches to the banana (gripper open), 5 banana grasps and 7 banana carries (closed). Last column: mean chunk-end end-effector distance to the teacher's chunk at the 21 releases.

| Arm | Location | Values changed | Rank | Releases left (of 21) | of the 15 not distilled | Carries released (of 9) | Phase-2 decisions flipped (of 19) | Chunk-end distance to teacher, cm |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Unedited | - | 0 | - | 21 | 15 | 0 | 0 | 8.67 |
| Teacher wording, unedited | - | 0 | - | 2 | 2 | 0 | 0 | 0 |
| V1-full | VLM MLPs, layers 0-16 | 450,468,837 (13.4%) | <=72 x 17 layers | 0 | 0 | 0 | 0 | 4.73 |
| V1-0.16% | VLM MLPs, image cells, layers 8-16 | 248,057,499 (7.4%) | <=32 x 9 layers | 2 | 1 | 0 | 0 | 4.5 |
| A1 | action-expert MLPs, layers 0-17 | 60,378,838 (1.8%) | 150 x 17 + 128 (layer 17) | 1 | 1 | 0 | 2 | 4.82 |
| H1 | action head, 8 used columns | 8,056 (0.00024%) | 8 | 2 | 1 | 0 | 6 | 5.98 |
| P1 (blind editor) | action head, gripper column, gated | 1,024 (3.1e-05%) | 1 | 0 | 0 | 0 | 0 | 7.87 |
| VA | V1-full + action-expert MLPs | 505,694,140 (15.1%) | <=72 x 17 (VLM) + 150 x 18 (action) | 0 | 0 | 0 | 0 | 4.56 |
| A1 without gate (control) | action-expert MLPs, Eq. (1) literally | 60,935,134 (1.8%) | 150 x 18 | 21 | 15 | 0 | 0 | 8.88 |
| A1, layers 13-17 | action-expert MLPs, layers 13-17 | 16,538,869 (0.49%) | 150 x 4 + 128 | 2 | 2 | 0 | 1 | 5.33 |
| H1, gripper only | action head, gripper column | 1,016 (3e-05%) | 1 | 3 | 2 | 0 | 5 | 8.14 |

- Every location flips the recorded releases, but not equally cleanly. V1-full, VA and P1 flip all 21 and change no carry or phase-two decision; A1 leaves 1 and closes the gripper at 2 of the 7 phase-two approaches; H1 leaves 2 and closes it at 6 of 7.
- In the action expert the MLP output reaches the residual stream only through the adaRMS gate (below 0.05 for 27-99% of the dimensions, by layer). Eq. (1) applied as if the gate were 1 does nothing (`a1-nogate`, 21 of 21 releases remain). Solved through the gate, the minimum-norm update is as large as the weights: median |dW|/|W| per layer 0.99 (range 0.04-2.22), against 0.017 (0.005-0.083) for V1-full and 0.030 for V1-0.16%. The explicit wording moves the action expert's residual stream by about 40% of its norm at layer 16 (and the VLM's image tokens by 2-4%, paper Section 5): by the time the decision reaches the action expert it is a large signal. (Correction to the pre-registration text: in layers 4-16 A1's |dW|/|W| ranges 0.36-2.22, not 0.9-2.2; layer 6 is the exception.)
- The VLM edit makes the action expert compute what it computes under the explicit wording without touching it. At the 6 distilled decisions V1-full reproduces 104-114% of the teacher's change of the action-expert residual stream in every layer (projection on the teacher's change; relative error 12-36%), and its gripper-velocity change per denoise step is -1.55 to -1.22 against the teacher's -1.56 to -1.14. A1 overshoots (112-126%, gripper -1.66 to -1.49); VA comes closest (99-105%, relative error 9-18%) (`results/gap_closed.json`).
- The action head cannot separate the premature release from the approach to the banana without a gate. At denoise step 0 the least-norm head edit changes the gripper velocity by -1.09 at the premature releases and -1.10 at the phase-two approaches; the blind editor's gated edit gives -1.46 and -0.15 (`results/head_selectivity.jsonl`).

### Closed loop on the target task: held-out slots 94016-94079

All arms and one shared base on the same 64 new slots (two 32-slot chunks, arm order reversed in the second chunk).

Slots 94016-94079 (64 paired slots).

| Arm | Successes | Wilson 95% | Fixed / broke vs base | Difference vs base, points [95% CI] | McNemar p | Fixed / broke vs V1-full | p vs V1-full |
|---|---:|---|---:|---|---:|---:|---:|
| Unedited (base) | 5/64 | 0.03-0.17 | - | - | - | - | - |
| V1-full | 55/64 | 0.75-0.92 | 50 / 0 | +78.1 [+67.2, +87.5] | 1.8e-15 | - | - |
| V1-0.16% | 41/64 | 0.52-0.75 | 38 / 2 | +56.2 [+42.2, +68.8] | 1.5e-09 | 5 / 19 | 0.0066 |
| A1 | 20/64 | 0.21-0.43 | 20 / 5 | +23.4 [+9.4, +37.5] | 0.0041 | 1 / 36 | 5.5e-10 |
| H1 | 0/64 | 0.00-0.06 | 0 / 5 | -7.8 [-15.6, -1.6] | 0.062 | 0 / 55 | 5.6e-17 |
| P1 (blind editor) | 30/64 | 0.35-0.59 | 28 / 3 | +39.1 [+25.0, +53.1] | 4.6e-06 | 4 / 29 | 1.1e-05 |
| VA | 43/64 | 0.55-0.77 | 41 / 3 | +59.4 [+45.3, +73.4] | 1.6e-09 | 5 / 17 | 0.017 |

Sensitivity analysis (pre-specified): the 32 slots that were the first episode of lane 0 in every chunk.

Slots 94016-94031 and 94048-94063 (32 paired slots).

| Arm | Successes | Wilson 95% | Fixed / broke vs base | Difference vs base, points [95% CI] | McNemar p | Fixed / broke vs V1-full | p vs V1-full |
|---|---:|---|---:|---|---:|---:|---:|
| Unedited (base) | 2/32 | 0.02-0.20 | - | - | - | - | - |
| V1-full | 29/32 | 0.76-0.97 | 27 / 0 | +84.4 [+71.9, +96.9] | 1.5e-08 | - | - |
| V1-0.16% | 18/32 | 0.39-0.72 | 17 / 1 | +50.0 [+31.2, +68.8] | 1.4e-04 | 1 / 12 | 0.0034 |
| A1 | 11/32 | 0.20-0.52 | 11 / 2 | +28.1 [+9.4, +46.9] | 0.022 | 1 / 19 | 4.0e-05 |
| H1 | 0/32 | 0.00-0.11 | 0 / 2 | -6.2 [-15.6, +0.0] | 0.5 | 0 / 29 | 3.7e-09 |
| P1 (blind editor) | 12/32 | 0.23-0.55 | 11 / 1 | +31.2 [+12.5, +50.0] | 0.0063 | 1 / 18 | 7.6e-05 |
| VA | 19/32 | 0.42-0.74 | 18 / 1 | +53.1 [+34.4, +71.9] | 7.6e-05 | 1 / 11 | 0.0063 |

Chunks that ran on one lane: a1-94016, a1-94048, base-94048, h1-94016, h1-94048, p1-94016, p1-94048, v1full-94016, v1full-94048, v1img816-94048, va-94016, va-94048 (of 14).

Stages and behaviour (RoboLab score and events; `where_stages.py`, `where_behaviour.py`):

| Arm | Cube placed | Success | Banana-step conversion | First chunk closes the gripper | Closed share of chunks before the first grasp | Wrong-object grabs per episode |
|---|---:|---:|---:|---:|---:|---:|
| Unedited (base) | 17/64 | 5/64 | 29% | 2% | 19% | 2.89 |
| V1-full | 60/64 | 55/64 | 92% | 2% | 15% | 0.19 |
| V1-0.16% | 50/64 | 41/64 | 82% | 2% | 14% | 0.80 |
| A1 | 27/64 | 20/64 | 74% | 52% | 58% | 4.52 |
| H1 | 0/64 | 0/64 | - | 94% | 97% | 0.00 |
| P1 (blind editor) | 40/64 | 30/64 | 75% | 9% | 20% | 1.55 |
| VA | 56/64 | 43/64 | 77% | 2% | 12% | 0.86 |

### Transfer and retention (32 fresh slots per task, 94016-94047)

Eyeglasses ran as one 32-slot chunk per arm on both GPUs (`where_run.py --all-lanes`). After the deviation noted above, cube left of bowl and cube into bowl ran on GPU0 alone in 16-slot chunks with the arm order reversed in the second chunk (`run_transfer2.sh`), so every slot is the first episode of its process; the queued `--all-lanes` runs for those two tasks were cancelled before they started. Each task's arms share one unedited run under the task's own instruction.

| Task | Arm | Successes (arm vs base) | Fixed / broke | Difference, points [95% CI] | McNemar p |
|---|---|---:|---:|---|---:|
| Pick up eyeglasses | V1-full | 25 vs 14 of 32 | 15 / 4 | +34.4 [+9.4, +56.2] | 0.019 |
| Pick up eyeglasses | P1 (blind editor) | 12 vs 14 of 32 | 5 / 7 | -6.2 [-28.1, +15.6] | 0.77 |
| Pick up eyeglasses | A1 | 11 vs 14 of 32 | 7 / 10 | -9.4 [-34.4, +15.6] | 0.63 |
| Cube left of bowl | V1-full | 20 vs 4 of 32 | 16 / 0 | +50.0 [+34.4, +68.8] | 3.1e-05 |
| Cube left of bowl | P1 (blind editor) | 1 vs 4 of 32 | 1 / 4 | -9.4 [-21.9, +3.1] | 0.38 |
| Cube left of bowl | A1 | 2 vs 4 of 32 | 1 / 3 | -6.2 [-18.8, +6.2] | 0.62 |
| Cube into bowl (retention) | V1-full | 32 vs 32 of 32 | 0 / 0 | +0.0 [+0.0, +0.0] | 1 |
| Cube into bowl (retention) | P1 (blind editor) | 32 vs 32 of 32 | 0 / 0 | +0.0 [+0.0, +0.0] | 1 |
| Cube into bowl (retention) | A1 | 31 vs 32 of 32 | 0 / 1 | -3.1 [-9.4, +0.0] | 1 |

- glasses: first chunk closes the gripper in Unedited (base) 0%, V1-full 0%, P1 (blind editor) 16%, A1 28% of episodes.
- left: first chunk closes the gripper in Unedited (base) 0%, V1-full 0%, P1 (blind editor) 6%, A1 47% of episodes.
- cube: first chunk closes the gripper in Unedited (base) 0%, V1-full 0%, P1 (blind editor) 0%, A1 44% of episodes.

### Paired statistics, predictions and interpretation

**Held-out, each arm against the shared base.** V1-full 55/64 vs 5/64, 50 fixed and 0 broken, +78.1 points [+67.2, +87.5], exact p = 1.8e-15. V1-0.16% 41/64 vs 5/64, 38 fixed and 2 broken, +56.2 points [+42.2, +68.8], exact p = 1.5e-09. A1 20/64 vs 5/64, 20 fixed and 5 broken, +23.4 points [+9.4, +37.5], exact p = 0.0041. H1 0/64 vs 5/64, 0 fixed and 5 broken, -7.8 points [-15.6, -1.6], exact p = 0.062. P1 30/64 vs 5/64, 28 fixed and 3 broken, +39.1 points [+25.0, +53.1], exact p = 4.6e-06. VA 43/64 vs 5/64, 41 fixed and 3 broken, +59.4 points [+45.3, +73.4], exact p = 1.6e-09.

**Against V1-full.** V1-0.16% 41/64 vs 55/64, 5 fixed and 19 broken, -21.9 points [-35.9, -7.8], exact p = 0.0066. A1 20/64 vs 55/64, 1 fixed and 36 broken, -54.7 points [-67.2, -42.2], exact p = 5.5e-10. P1 30/64 vs 55/64, 4 fixed and 29 broken, -39.1 points [-53.1, -25.0], exact p = 1.1e-05. VA 43/64 vs 55/64, 5 fixed and 17 broken, -18.8 points [-32.8, -4.7], exact p = 0.017.

**Transfer and retention (arm vs base on 32 paired slots, fixed/broke, exact p).** Eyeglasses: V1-full 25 vs 14 of 32, 15/4, p = 0.019; P1 12 vs 14 of 32, 5/7, p = 0.77;
A1 11 vs 14 of 32, 7/10, p = 0.63. Cube left of bowl: V1-full 20 vs 4 of 32, 16/0, p = 3.1e-05; P1 1 vs 4 of 32, 1/4, p = 0.38; A1 2 vs 4 of 32, 1/3, p = 0.62. Cube into bowl (retention): V1-full 32 vs 32 of 32, 0/0, p = 1;
P1 32 vs 32 of 32, 0/0, p = 1; A1 31 vs 32 of 32, 0/1, p = 1.

**The pre-registered predictions, checked.** (1) V1-full and V1-0.16% both gave large gains, but they did differ:
V1-0.16% was lower than V1-full on these slots (see above), unlike the paper's held-out runs on slots 90000-90127, where
the two edits were scored against separate unedited runs (106/128 and 102/128) and the 0.16% edit scored 54/64 against
50/64 for the full edit on the first 64 of those slots. Prediction failed. (2) P1 gained more than predicted (+15 to +25
points) and less than V1-full. (3) A1 gained less than V1-full. Held. (4) H1 closed the gripper where it should stay open
and fixed nothing. Held. (5) VA was predicted to equal V1-full; it was lower. Prediction failed. (6) Transfer: V1-full
raised both transfer tasks; P1 and A1 raised neither (H1 was not run, as planned, because it fixed nothing). Retention
was unchanged for every arm. Held. The unedited policy scored lower on these slots (5/64) than in any earlier
held-out run of this task (14-21%); the comparisons are paired, so this lowers absolute rates, not the differences.

**Interpretation: where the fix lives.** In how the instruction is read, upstream of the action expert. A change of
about 1.7% of the norm of each VLM MLP down-projection (median over layers) makes the action expert compute, at the
distilled decisions, what it computes under the explicit wording (104-114% of the teacher's residual change in every
action-expert layer), and it does so only where the instruction matters: the first action chunk closes the gripper no
more often than for the unedited policy (2% of episodes, as for base), no phase-two decision flips offline,
the banana step succeeds in 92% of the episodes that place the cube, and the gain carries over to two other tasks
without changing the retention task. The VLM is also where the explicit wording's effect is small (it moves the image
tokens by 2-4% of their norm, paper Section 5); by the time it reaches the action expert it is a large signal tied to the
robot's state, and copying it there costs more and leaks:

- In the action expert the same distillation needs updates as large as the weights (median |dW|/|W| 0.99): the MLP
  output is gated down by adaRMS, and the decision is already a large change of the residual stream. Offline it flips
  20 of 21 releases, but in closed loop the first action chunk closes the gripper in 52% of the episodes, before
  the arm has reached anything, and the gain is a third of V1-full's. The same leak appears on the transfer tasks
  (first chunk closed in 28-47% of episodes there).
- In the action head a least-norm copy is a gripper-closing bias of about the same size at the premature releases and at
  the approaches to the banana (offline, above): the first chunk closes the gripper in 94% of the episodes and no
  episode succeeds. The blind editor's edit shows that the head can carry
  a fix when a gate separates the premature release from other states (first chunk closed in 9% of episodes);
  that gate was built from negative examples (initial approach, banana approach and grasp, release over the bowl), which
  the distillation recipe does not use, and even so it recovers about half of V1-full's gain and transfers nowhere.
- Adding the action-expert update on top of the VLM edit (VA) brings the action expert closest to the teacher at the
  distilled decisions, yet in closed loop VA is worse than V1-full: it places the cube about as often (56 vs
  60 of 64) but completes the banana step less often (77% vs 92%). Matching the teacher more
  closely at 6 decisions bought nothing; the extra action-side update acts in states it was not built from.

So the answer to "where should the edit go" is: where the known-good input differs from the failing one, which here is
the instruction's reading in the VLM. Action-side edits can be made to work on the target task (A1, P1), but they need
either very large updates or a hand-built gate, they recover less, they change behaviour in states that were never
distilled, and they do not transfer. Limits: one task, one teacher wording, 64 held-out and 32 transfer pairs per
comparison; A1 needed one change to the recipe (solving through the adaRMS gate), and other action-side constructions
(keys from more states, preservation keys, an explicit gate) might do better; 12 of the 14 held-out chunks ran on one
lane, and the pre-specified sensitivity analysis on the 32 slots unaffected by that gives the same ordering and the
same significant differences.

### GPU time

Closed loop: 3.65 GPU-hours on the target task and 1.94 on transfer and retention (wall time of each chunk times its lanes); activation captures at most 0.31 GPU-hours (capture time plus 60 s of model load per call). Total 5.90 GPU-hours on the two-GPU host, i.e. about 3.0 hours of the whole host (`results/gpu_time.json`).

### Files

Scripts (run on the GPU host with `owsp` against `projects/where-to-edit`, or locally where noted):

- `make_specs.py`, `specs/`: the `owsp acts` capture specs (6 distilled decisions with full action-expert traces; 49 check decisions); `run_acts.sh` runs them.
- `build_action.py` (A1, VA, `a1-nogate`, `a1-late`; the CFG line of each is in `builds/`), `build_head.py` (H1, `h1-grip`): `owsp py` sandbox builds.
- `check49.py` (offline table), `gap_closed.py`, `head_selectivity.py`, `rel_norms.py`: offline analyses.
- `ctl_run.py` (copied unchanged from `study/results/opus/controls/`; held-out run), `where_run.py` (its copy with lane recording and `--all-lanes`; transfer runs), `run_transfer.sh`, `run_transfer2.sh`.
- `where_analyze.py` (paired statistics, as in `ctl_analyze.py`), `where_stages.py`, `where_behaviour.py`, `gpu_time.py`, `collect_results.sh` (local: runs them and copies the JSON), `make_tables.py`, `write_results.py`.

Results (`results/`): `offline_49.json`, `diff_*.json` (`owsp diff`), `build_*.json`, `size_summary.json`, `vlm_rel_norms.jsonl`, `gap_closed.json`, `head_selectivity.jsonl`; `heldout_results.json` (per-slot outcomes), `heldout_plan.json` (arms, fingerprints, schedule), `heldout_analysis.json`, `heldout_analysis_run0.json`, `heldout_stages.json`, `heldout_behaviour.json`; `transfer_{glasses,left,cube}_{results,plan,analysis,behaviour}.json`; `gpu_time.json`.

Candidates and fingerprints (projects `where-to-edit`, `where-glasses`, `where-left`, `where-cube`): `v1-full` 63db0c52a51306c9, `v1-img816` 5bf8c4329c1e416a, `p1-blind` 4ca8023b0592b79f (byte copies of the sources, fingerprints verified), `a1` 09b7c9f5bfe26912, `h1` d965cd706aed9dec, `va` 5ee25e8d2e78a555.
