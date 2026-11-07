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
