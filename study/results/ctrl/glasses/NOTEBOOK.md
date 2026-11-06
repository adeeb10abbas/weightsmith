# Notebook (glasses editor)

## Setup
- Task: PickGlassesTask, "Pick up the eye glasses". Dev slots 70000-70031 (offset 0, n 32).
- Budget (supersedes BRIEF): max 2 candidates (p1, p2), each eval once n32 offset0; 96 episodes total incl base; 192 acts items.
- Sandbox: NAMES is empty under `base`; READABLE_NAMES exists (730). Helpers: load/base/save/bf16/obs/acts/episodes.
- action_out_proj: cols 0..7 near-orthogonal, norms ~0.75-0.85; bias tiny.

## Session 2 (continuation)
- Session 1's base eval never registered (status showed 0 episodes). Re-launched base eval n32 offset0.

## Base result: 12/32 (37.5%). Episodes cap at 450 steps (30 replans).
OK: 00 01 02 03 05 12 16 18 20 21 22 30. FAIL: the other 20.
- "WRONG_OBJECT_GRABBED_FAILURE: 'glasses'" is a labelling artifact (target names group1/group2); the grabbed object IS the glasses.
- Success = object_above(glasses, table, z_margin=0.05), fires right after a lift (~z_EE 0.12-0.18, our FK point is flange+0.21 m).
- Gripper obs: 0 open, ~0.85-0.88 holding glasses, 1.0 closed on nothing.
- FK table in /work/notes/fk_base.json (step, x, y, z, g_obs, a0_grip, mean_grip).

### Failure taxonomy (20 fails)
1. HELD-NOT-CLEARED (9: s04 s11 s15 s23 s24 s25 s26 s28 s29): holds glasses (g~0.85) and lifts to z_EE 0.14-0.25
   but no ABOVE; frames (s11, s25) show glasses held by one temple arm, dangling, frame still touching table.
   Policy then wanders and re-opens (releases) after 1-7 replans, regrasps etc. Successes also often grasp
   a temple (s12) but the hang height differs.
2. HESITATION (s06 s07 s09 s13 s17 s27 s31, partly s10): descend to z~0.07, gripper action hovers 0.3-0.8 for a
   replan, does not commit, retreats to z~0.25, repeats.
3. CLOSED-ON-NOTHING persisting (s08, s19 whole episode; s14, s24, s26 partly): g=1.0, policy keeps carrying nothing.

### Hypothesis for p1: lift higher once holding
Mode 1 is the largest single class, and lifting higher only helps (success fires the moment the glasses clear).
Plan: rank-1 state-gated edit on action.action_out_proj: dW = a (x) d, where a = head_input contrast
(HOLD minus OPEN states; network's own "holding" feature) and d = joint-space velocity for +dz EE lift
(least-norm Jacobian solution with orientation fixed, converted into normalized action units).
- Capture a1 (90 items, base): HOLD 36, EMPTY 14, OPENLOW 24, OPENHIGH 16; action layer 17, all 10 denoise steps.

### a1 analysis
- Actions are absolute joint targets (rad) but the model emits normalized DELTAS vs current joint state:
  calibrated (regress -0.1*sum_t v_t on a - q) scale ~[0.66 0.77 1.0 1.18 0.90 1.21 1.0] rad/unit (noisy for j0, j2).
- Least-norm joint change for +0.10 m EE z with orientation held (FK Jacobian pinv at hold configs):
  dq = [-0.021 -0.244 0.036 -0.021 0.154 -0.205 -0.13] rad (gives xyz +(-0.011, 0, 0.097) +- 0.017).
- Linear "holding" readout on head_input is NOT selective: OPENHIGH projects at -2, OPENLOW at -1, HOLD at -1.5..1.6.
  A linear edit would push the arm DOWN when open. Low-projection HOLD items are those where the policy is about to release.
- No existing block-17 MLP unit is a clean gate (best one, 3750, overlaps).
- adaRMS verified: final-norm reconstruction from resid_out has 2.6% rel err; block-17 MLP input reconstructed
  (token-mean, mid = resid_out - gate*mlp_out), recomputed GeGLU hidden corr 0.99 with captured.

## p1 (built, pre-registration)
- Mechanism targeted: HELD-NOT-CLEARED (mode 1). When the policy is committed to holding, raise the joint targets
  so the dangling glasses clear the table (+5 cm z_margin).
- Edit: repurpose action.blocks.17 MLP unit 2437 (least-used unit: usage 0.0015 vs median 0.23).
  gate column = alpha*(g_hat - thr*e_hat/kappa), where g_hat = (committed-HOLD minus OPEN) mean contrast of the
  reconstructed block-17 MLP input (orthogonalized to the constant direction e_hat), thr = max over all open items/steps
  (1.27), alpha=2.98; up column = e_hat/kappa (makes x.u ~ 1). So the unit ~ ReLU(Z - thr): exactly 0 on all 40 open
  items (token-mean), 0.5-9 on 22/26 committed holds, ~0 on 10 release-intent holds, some on closed-on-nothing.
  down row 2437 = least-norm 1024-vector such that, through ffw adaRMS gate, final-norm gain and action_out_proj,
  the integrated action change = dq(+8 cm) in normalized delta units at the median hold activation; gripper dim fixed at 0.
- Size: 3 vectors (2x1024 gate/up columns, 1x1024 down row) = 3072 values, rank-1 per tensor.
- Offline (linearized, a1 items): predicted |dq| HOLD mean 0.20 rad (~5 cm; max 0.65), EMPTY 0.06, OPEN 0.00.
- Validation capture v1 (62 items, cand p1): 16 committed holds, 4 empty, 30 about-to-close / mid-closing, 6 openlow, 6 openhigh.

### v1 result: p1 (all-steps push) REJECTED before eval
- Holds: dz mean +2 cm but range -13..+23 cm, xy displacement 6-22 cm, max|dq| 0.43 rad; 4/16 holds lost gripper-close
  tokens (release). Linear prediction (+5 cm clean) failed: pushing v at every denoise step shifts x_t and the flow
  model reacts nonlinearly at later steps.
- Closing (about-to-close) states: unit leaked (0.2-2.3), some disruption (s05 dz -11 cm). Plain open states: clean (0).

## p1 (revised, never evaluated before; this is what gets evaluated)
- Same mechanism + same gate unit 2437, two changes:
  1. Step-9-only push: down row = least-norm solution of 296 linear constraints: zero velocity change on ALL 32 output
     dims at denoise steps 0..8 (so x_t and every later input are untouched: no feedback), and at the last step (t=0.1)
     -0.1*dv = dq(+8 cm) in normalized delta units at the reference activation (gripper dim 0). Exact: x0 = x_0.1 - 0.1 v.
     After bf16 rounding, early-step |dv| <= 7e-4.
  2. Gate direction re-derived from the step-9 MLP input; thr = max over all a1 open items (incl. 14 closing-like) + 0.5.
- Size: 3072 values (gate col, up col of gating_einsum[.,.,2437]; down row linear[2437]); row norm 1247 (scaled by the
  tiny ffw adaRMS gate ~0.002).
- Offline (exact up to token-mean gate approx, a1 items): dz on committed holds 0 for 7/26, +0.7..+13 cm for 19/26
  (mean +3 cm); EMPTY mean +0.7 cm; all 40 open items 0.000.
- Prediction (n32 offset0): 14/32 (base 12); fixed ~3 (from s04 s11 s23 s24 s25 s26 s28 s29 class), broke ~1 (noise);
  i.e. within same-weight noise (~30% pair flips). Risk: residual gate leak at closing moments -> missed grasps.

### p1 eval result (n32 offset0): 18/32 vs base 12/32; fixed 9, broke 3
- Fixed: s07 s10 s13 s14 s15 s17 s19 s25 s26. Broke: s05 s12 s20.
- Prediction (14, +3/-1) under-shot. The direction held, the size did not. Sign test on 12 discordant pairs: one-sided p ~0.07.
- Mechanism check (scripts/mech_p1.py): mean planned chunk rise at held replans 0.096 m (base) -> 0.108 m (p1).
  Clear increase in held-not-cleared fixes: s25 +0.009->+0.077, s26 0.000->+0.049, s10 +0.034->+0.121, s15 +0.015->+0.036.
  But s07 s17 s19 were hesitation / closed-on-nothing in base (no hold), where the gate cannot act before a hold, so
  part of the gain is likely closed-loop divergence/noise. Broke s20: held, rise -0.007 (did not lift).
- Stopped here (time budget ~2h exceeded); no p2.
