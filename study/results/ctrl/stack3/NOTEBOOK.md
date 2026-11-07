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
- Round 1 FINAL.md = base.

## Round 2 (operator extended time, same budget: <=2 candidates, 64 episodes, 108 acts items left)
- Plan: nonlinear gate = repurpose a near-dead unit r of action.blocks.17.mlp_1. gate row G0[:,r] reads the "empty-gripper, high at stack"
  state direction in the MLP-17 input (normed) space with a threshold; up row G1[:,r] reads a near-constant dim; down row linear[r] writes
  the least-norm head direction that raises v7 only (orthogonal to joint columns W0..6), mapped back through final-norm scale and ffw gate.
- Capture a2-pertoken-L17 (96 items, per-token, action layers 16/17, all 10 denoise steps): T 22, T2 8 (closed on air at stack, cmd close),
  P1 24, P2 22, P2s 16 (holding cube over stack, cmd close = most confusable), P4 4. Selection scripts/select2.py -> /work/notes/sel2.json.
- scripts/recon.py: rebuild adaRMS cond from time MLPs (sincos min 4e-3 max 4.0, swish), verify final-norm(r17)==head_input and MLP17 hidden.
  Result: final-norm rel err 0.003, MLP17 hidden rel err 0.005 (corr 1.000) -> exact forward model of layer-17 MLP + head.
  Layer-17 ffw adaRMS gate is tiny (|g17| <~ 0.013), so the unit needs a large down row.
- Separability per token (design.py/design2.py): plain contrast in MLP17 input overlaps per token (flow-noise per token). Removing the top-k
  eigvecs of WITHIN-ITEM (token/step) covariance before the contrast fixes it. design3.py held-out (train even/test odd and vice versa),
  threshold = train P max + 0.25*(T median - P max), bias via most-constant linear feature v=(C+lam I)^-1 mu (x.v = 1 +- 0.01):
  k=256: held-out P token fire 0.0000 / 0.0000; held-out T token fire 1.00 / 0.81; T items >50% tokens 1.00 / 0.81.

### p1 (created) — mechanism, edit, size, prediction
- Mechanism: block the "empty gripper, hovering high at the 2-stack, commanding close" decisions (stack re-grasp loop / un-stacking), leave
  every grasp/carry/holding-over-stack decision untouched.
- Edit (scripts/build.py, all-data fit): repurpose near-dead unit 2437 of action.blocks.17.mlp_1 (max|h| 0.13 on 96 items).
  gating_einsum[0][:,2437] = kappa*(d - theta*v)  (d = contrast T∪T2 vs P1∪P2∪P2s in MLP17-input space, top-256 within-item eigvecs projected out;
  theta=0.634, kappa=27.8); gating_einsum[1][:,2437] = v (constant ~1 channel); linear[2437] = 0.81*o, o = weighted least-norm solution with,
  at every denoise step s: (1+sF_s)*g17_s*W_j . o = 0 for joints j=0..6, =1 for gripper j=7, and g17_s*rbar_s . o = 0 (no 1st-order rms change).
  Size: 3072 values (2048 in gating_einsum, 1024 in linear), one unit. max|delta| 280 (linear row), 3.2 (gating).
- Offline (first-order, exact final norm, bf16 weights + bf16 inputs): T close-frac 0.89->0.07, T2 0.90->0.17; P1/P2/P2s unchanged (|da7|<=0.001,
  0 token flips); max|dJoint| 0.058 on T, 0.0015 on P. bf16 x: pre-act deviation <=0.10 (kappa units ~28), P max pre-act -3.26, T 94% tokens fire.
- Exact replay check a3-p1check (12 held-out decisions not used in fitting, real model incl. denoising feedback; scripts/check3.py):
  preservation 8/8 unchanged (4 P2s holding-over-stack, 3 P1 grasps, 1 P2 carry: a7 identical to 2 dp, max|dJoint| <= 0.005).
  held-out T2 (closed on air at stack): 2/4 flipped to open (a7 0.97->0.17, 0.92->-0.04), 1 weakened (0.97->0.68), 1 barely (0.98->0.89).
  -> safe, partially effective. Budget after: 108/192 acts items... (84+96+12 = 192 used). Proceed to closed-loop eval.
- Prediction for eval p1 (32 slots, paired vs base 16/32): removes the close-on-air/un-stack loops in ~6 failure eps (70001,70003,70010,70018,
  70019,70021) but the arm may still linger at the stack; modes 2-4 unaffected. Predict 18/32 (fixed ~4, broke ~2; range 14-22 given ~30% flip noise).
- **p1 result: 17/32 (base 16/32), fixed 6 (12,16,17,19,21,28), broke 5 (00,02,13,15,26).** Prediction (18, fixed~4, broke~2): success roughly held,
  broke higher than predicted. Mechanism confirmed closed-loop (scripts/cmp.py): empty-gripper close decisions at stack 17 -> 2,
  closed-on-air-at-stack replans 73 -> 5, B reached in 27 vs 24 eps, median pick-B step 285 vs 300. But loop-rescued eps (70001, 70010, 70030)
  still fail later, and broken eps pick B on time (225/210/240) and then fail -> success now limited by 3rd-cube placement / later mess.
  Net +1 is noise.
- Post-hoc (scripts/place.py, sig.py, frames p1 70002/70013): new p1 failures = 3rd cube released over stack, slides off, lands beside it,
  arm parks high and idles (same as base 70006/70017). 3rd-placement success given B picked: base 16/25, p1 17/28 (unchanged).
  No premature-release signature: mostly-open chunks while holding high at stack base 0.30 vs p1 0.27, holding elsewhere 0.08 vs 0.05;
  only already-open release chunks get slightly more negative a7 (-0.03..-0.10). Loop-involved eps (13): base 6 ok -> p1 7 ok; faster
  successes (70004 415 vs 719, 70009 280 vs 658, 70022 256 vs 435). Non-loop eps (19): 10 -> 10.
- p2 considered: stricter margin 0.5 keeps P fire 0 but held-out T fire 0.92/0.73 (vs 1.00/0.81). No validatable lever on placement/recovery
  (acts budget exhausted), and a near-replicate cannot resolve +-3 episodes on 32 slots. p2 NOT created; 32 episodes left unused.
- Round-2 decision: submit p1 (FINAL2.md).
