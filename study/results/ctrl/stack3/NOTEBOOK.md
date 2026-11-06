# Notebook (stack3: Stack3RubiksCubeTask, "Stack the rubiks cubes in a tower")

## Log
- 11:05 Read BRIEF + PRIOR_FINDINGS. Budget 0 used. Started `eval base --n 32 --offset 0` (slots 70000-70031).
- Session 2: status showed 0 episodes used (previous base eval never ran). Re-started `eval base --n 32 --offset 0`.
  Budget for this session: max 2 candidates (p1,p2), 96 episodes total incl. base, 192 acts items.
- Base done: 16/32. Successes: 70000,02,04,05,07,08,09,11,13,14,15,22,23,24,25,26.
  Failures (all score 0.5, i.e. 1 of 2 subtasks, all time out at 900): 01,03,06,10,12,16,17,18,19,20,21,27,28,29,30,31.
  Note: eval wrapper locally gets killed at the 30-min background limit but the remote job completes; use long timeout.

## Diagnosis (base, FK with Panda mod-DH + 0.21 m tip; z is relative, table grasp z~0.09-0.13)
- Layout is ~fixed: cube A (0.50,0.03), cube B (0.43,0.19), cube C (0.39,-0.09). Typical plan: A onto C (first stack ~step 110-180), then B onto stack.
- All 16 failures complete the first stack (score 0.5) and fail the 3rd cube. Modes (scripts/seg.py, dither.py; /work/notes/phase.json):
  1. **Stack-dwell / re-grasp at stack** (main): after releasing A on C, the gripper is empty and hovers at the stack; the policy commands
     CLOSE at stack height (z~0.18-0.26, dS<0.07) -> closes on air (g=1.0) in loops, or grabs/un-stacks the top cube.
     70001 (B picked only at 660), 70010 (never), 70018 (555), 70019 (un-stacked top cube at 165), 70021 (555), 70003 (re-pick near stack), 70014/70015/70004 (successes but slow, same loop).
  2. Lingering at stack without closing: 70012, 70030 (never reach B), 70031 (B at 495).
  3. 3rd-cube placement failure then arm retracts and idles high (tail z~0.32-0.35): 70006, 70017 (B picked fast, placed misaligned), 70028.
  4. Different order / mess: 70016, 70020, 70027, 70029.
- Frames: 70010 shows 3rd cube far left while gripper fiddles with stack top; 70006/70017 3rd cube left leaning next to stack, arm parked high.

## Candidate idea: gated gripper-open readout for "empty gripper, high at stack" closes
- Capture a1-stackclose (84 items, parity_err 0.12): T = 22 decisions after first stack with empty gripper, dS<0.08, z>0.15, cmd close;
  P1 = 24 table-level grasp closes; P2 = 22 carrying (holding, cmd close); P4 = 16 same state as T but cmd open. Selection: scripts/select.py, /work/notes/sel1.json.
  Captured gripper close-frac: T 0.89, P1 0.86, P2 0.99, P4 0.00.
- Planned edit: rank-1 on action_out_proj kernel column 7 (+ bias[7] centering), dv7 = beta*(h.u - c), u = contrast direction.
- Offline (scripts/an1-an4.py):
  - head_input mean-diff direction: in-sample perfect (T>=0.52 vs P<=0.41) but split-half (even/odd episodes) fails: held-out T overlaps held-out P1/P2 (P2 up to 1.18 vs T 0.29-1.13). Not selective.
  - single MLP units (action L8, L17): no unit separates T from P even in-sample (best margins < 0, near-dead units).
  - action resid L17 (pre-final-norm, token mean): held-out separation is clean (split0 T min 21.6 vs P max 12.5, d'=6.3; split1 T min > P max).
    So a state feature exists, but it is not readable linearly after the final norm: mapped to head space (u = d/ratio), per-token T-vs-P gap ~0.04
    with per-token spread ~0.03; flipping targets (da7 ~ -0.5) needs beta ~50 -> +-1.5 gripper swings on carrying/grasp tokens (mid-carry drops).
    beta<=2 changes nothing.
  - VLM L17 task/state slots: weak held-out separation (FPR 0.12-0.41).
- Decision: rejected offline. No readout-level gate is both effective on targets and safe for carrying/grasping. A nonlinear gate (new unit) would be needed;
  no time/budget left to build and validate one (queue waits ~25 min per GPU job). Submit base. Used 32/96 episodes, 84/192 acts items.
