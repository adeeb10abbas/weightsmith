# Notebook — RubiksCubeLeftOfBowlTask ("Put the rubiks cube to the left of the bowl")

## Log
- Started base eval (n=32, offset 0) in background.

## Base result: 5/32 (s70000, 70001, 70002, 70016, 70028)
Notes on events: "WRONG_OBJECT_GRABBED_FAILURE: rubiks_cube" fires on every cube grasp (target-group naming
quirk) — ignore. Episodes cap at 450 steps. Success terminates the episode early; the left_of condition is
checked even while the cube is held (s70016 succeeded with cube in hand at EE (0.50,+0.23)).

Geometry (FK, Panda mDH, tip 0.21 m; x fwd, y = robot left):
- Cube starts at ~(0.42,-0.06). Bowl is roughly at x~0.47-0.55, y~+0.1..+0.15 (cube dropped into it from EE (0.46,+0.19) in s70001).
- Base cam: robot +x -> image up-left, robot +y -> image left-down.
- Grasp is reliable (nearly all episodes grasp the cube by step ~50-200).

Failure classification (27 failures):
- ~22: "carry overshoot forward": after grasp the carry has big +x components (chunks with dx +0.08..+0.16) and the
  cube is released at x~0.57-0.66, y~+0.02..+0.19 = BEHIND the bowl, not left of it. Then the policy dithers.
- ~4: grasp/re-grasp loops near start (s70010, s70012, s70014, s70029), never carry.
- Successes carry mostly along +y at x~0.44-0.50 (s70016: dy +0.09,+0.16,+0.08 with dx ~0) reaching y~0.2.
=> Mechanism to target: during carry the policy's placement target is too far forward (+x) / not far enough +y.

Carry decision table: /work/notes/carry_rows.json (per replan while holding: p0, chunk displacement d14).

## Diagnosis
- a1 (72 items): 12 carry-start decisions from failures x 6 prompts (orig, right of, in the bowl, behind,
  in front of, "Move the rubiks cube to the left"). Question: does the spatial word steer the carry direction?
  RESULT (session 2, scripts/an_a1.py): mean chunk displacement d14 (x,y,z) over the 12 decisions:
    left of (+0.089,+0.088)  right of (+0.084,+0.010)  in the bowl (+0.008,+0.111)
    behind (+0.116,+0.055)   in front of (+0.046,+0.035)  "Move ... to the left" (+0.103,+0.075)
  => the word does steer. left-right contrast = (+0.005,+0.078): pure robot +y (correct axis).
     "left" itself is a 45deg diagonal (+x,+y) = exactly IMAGE-left in the base over-shoulder camera
     (robot +x -> image up-left, +y -> image left-down => image-left = 0.71(+x)+0.71(+y)).
     Hypothesis: the policy reads "left" partly camera-frame; the task checks robot-frame left (+y).
     Successes carry along +y at x~0.44-0.51; failures drift to x 0.55-0.66 ("behind" the bowl).
- Success geometry is murky (s70002/s70028 succeed with cube near y~0.12 dropped/held; s70007 held at
  (0.47,0.14) no success). Robust signal: successes stay at x<=0.51 during carry.
- a2 (24 items): same 12 decisions x {"to the front of the bowl","to the back of the bowl"} (token-aligned
  with "to the left of the bowl"), + image tokens. Goal: get an aligned front-back contrast (-x axis) to
  combine with left-right (+y) as a steering vector at the "left" token slot (slot 10).
  RESULT: front (+0.026,+0.034), back (+0.116,+0.040) => front-back = (-0.090,-0.006): pure robot -x.
  So the policy's relation words live on robot-frame axes; "left" just carries a "back"-like +x component.
- Layer-0 facts (scripts/res_a1.py): left-right resid difference at L0 is confined to slot 10 (|d|=86,
  slot 11 34, others ~0) and identical across items (cos 1.000). L0 MLP units 8023 (+9.01) and 3097 (+8.41)
  fire only on the token "left" (other prompt slots |h|<=0.2, image grid means <=0.14, "right" ~0).
  => a clean token-specific write port: adding v to rows 8023/3097 of language.blocks.00.mlp.linear adds
  ~9v to the layer-0 residual of the "left" token only.

## Candidate p1 (plan, written before eval)
- Mechanism: the carry goes diagonal (+x,+y) because "left" representation carries a +x ("back"-like)
  component; failures drift +x behind the bowl. Edit rotates the meaning of "left" toward robot +y.
- Edit (scripts/edit_steer.py): delta = 0.5*(r_left - r_right) + 0.8*(r_front - r_back), r = layer-0
  resid_out at slot 10 (from a1/a2, item-mean). Least-norm split over the two "left" units:
  W[u] += c_u*delta, c = h/(h.h) (c=0.059,0.055). Rank-1 (2 rows) of language.blocks.00.mlp.linear,
  4090 values changed, |delta|=87 vs |r_slot10|=479; leakage to other tokens <1 (0.2%).
- Linear prediction for carry chunk: dx 0.089 -> ~0.02, dy 0.088 -> ~0.12.
- Offline validation a3 (30 items on p1): 12 target carry decisions, 12 approach/pre-grasp decisions,
  6 success-carry decisions.
  RESULT a3 (scripts/val.py): injection verified exact (|d|=87 at L0 slot 10, ~0 elsewhere) but behaviour
  barely moved: target mean d14 (+0.090,+0.088) -> (+0.091,+0.084), mean |diff| 0.008 (bf16 noise ~0.01).
  => additive steering of the layer-0 "left" token is NOT read linearly downstream (token identity seems
  decoded categorically; the L-R difference grows 86 -> 182 from L0 to L8 only for a real token swap).
  Layer-0 version rejected without an eval (it would be a null edit).
- p1 redesigned (scripts/edit_l8.py, L0 edit dropped): same contrast but measured and injected at
  layer 8 resid_out slot 10: delta8 = 0.5*(L-R)_8 + 1.0*(F-B)_8, |delta8|=262 vs |r10|=754.
  Write port: least-squares over 64 layer-8 MLP units selective for slot 10 (gain at slot 10 = 0.96-1.02
  per item; leak gain <=0.041 other prompt slots, <=0.031 image grid cells; ridge 1e-2).
  Edit = rank-1 update on 64 rows of language.blocks.08.mlp.linear (outer(c, delta8)).
  Validation a4 (same 30 items) running.
  RESULT a4: effect is real now, in the predicted direction but ~1/3 of linear prediction:
    target carry mean d14 (+0.090,+0.088) -> (+0.061,+0.095), mean |diff| 0.043; strongest movers
    s70003 k16 (+0.126,+0.097)->(+0.030,+0.164), s70031 k6 (+0.081,+0.052)->(+0.011,+0.040);
    unresponsive: s70006, s70009, s70022 (large +x stays).
    success-carry decisions: dx -0.011 -> -0.042, dy 0.134 -> 0.120 (still +y carries).
    Side effect: 3 grasp-closing decisions flip to "don't close yet" (s70016 k4 0.60->0.48, s70017 k3
    0.71->0.01, s70024 k4 0.70->0.01) and s70028 k2 lifts instead of descending. (The near-null L0 edit
    had flipped only the knife-edge s70016 k5 and s70021 k4.) => expect delayed grasps / some re-grasping.

## p1 eval (written before eval)
- p1 = layer-8 slot-10 steering above (64 rows of language.blocks.08.mlp.linear, rank 1).
- Prediction: base 5/32 -> p1 ~6/32 (plausible range 3-9); fixed ~3, broke ~2. Carry should end at smaller
  x (fewer "behind the bowl" drops), but grasp delays may add re-grasp loops. Honest prior: likely within noise.
- RESULT p1: 3/32 (s70013, s70014, s70030) vs base 5/32; paired fixed 3, broke 5. Prediction did NOT hold.
  5 episodes never grasped the cube at all (s70004, s70007, s70009, s70015, s70025; base grasped in all 32)
  and many grasps came late (GRABBED@300+ in s70003, s70018, s70023, s70024, s70029). The grasp-closure
  side effect seen offline (3/5 closing decisions flipped) dominated closed loop. The carry-direction change
  was too weak (~1/3 of the linear prediction) to compensate.
- Decision: no p2. Time is ~2h50m (over the ~2h aim); a p2 would need a phase-gated write port (fire only
  while holding) plus another validation capture and eval. Submitting base.
