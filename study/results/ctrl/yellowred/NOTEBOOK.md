# Notebook (yellowred: StackYellowOnRedTask, "Stack the yellow block on the red block")

## Log
- Started base eval n=32 offset 0 (dev slots 70000-70031).
- Editable tensors: only `*.blocks.*` + action head/in/time/final_norm (no embedder, no vision stem).
- Session 2: status showed 0 episodes used (previous base eval never ran). Restarted base eval n=32 offset 0.
- Budget now: 2 candidates (p1,p2), 96 episodes total incl. base, 192 acts items.
- NOTE: background Bash default timeout (30 min) killed the eval client and the job did not count. Always run evals with Bash timeout 7200000. Restarted base eval (2nd try) with 2h timeout.

## Base eval (dev slots 70000-70031): 13/32
Successes: 00,02,03,07,08,10,11,16,21,26,27,29,31.
Note: "WRONG_GRABBED 'yellow_block'" events are spurious (fire on correct grab: target listed as group names).
Scene is (nearly) the same layout every slot: yellow block (dim lime-yellow) front-left at robot xyz~(0.34,0.40,0.08);
red block ~(0.51,0.23); rubiks_cube_1 with YELLOW top face right next to red block ~(0.43,0.195); other rubiks cubes,
blue block, lizard/dino figurines, birdhouse with yellow trim (~(0.6-0.75,0.3)), bowl.
Failure classes (19 fails):
- A. never grabbed yellow (13): 01,04,05,06,12,13,19,20,22,23,24,28,30. Wrong grabs mostly rubiks_cube(_1) (04,05,06,13,20,22,23,28,01),
  red block (22,23), blue block (05), lizard (01,06), birdhouse (12,24). 19: went straight to yellow but missed grasp then wandered to birdhouse.
- B. grabbed yellow, not stacked (6): 09,14,15,17,18,25 (repeated open/close at yellow, drops).
Trajectory pattern (FK, traj in /work/notes/traj_base.json): almost every episode first goes to (0.43,0.20,0.07) = yellow-topped rubiks_cube_1
next to red block (steps 45-105). Successes then lift and re-target to (0.34,0.40) yellow block; failures close on rubik / wander.
Only 02 and 19 went directly to yellow from k=1.
=> Main mechanism hypothesis: lookalike target selection - "yellow block" matched to the rubik's cube yellow face (located next to red block).

## Diagnosis
- a1 (64 items): 8 decisions (k=1 of 04,13,28,30; hover-over-rubik decisions 04k10,13k3,20k3,28k5) x 8 prompts; metric = FK of chunk end.
  RESULT: prompt wording has ~no effect. All 8 prompts (incl. "Pick up the yellow block", "lime", "yellow-green", "next to the green
  block", and even "Stack the GREEN block...") give the same chunk-end targets (~(0.41-0.43,0.16-0.20)) and same gripper closes at hover.
  => target choice is visually/prior driven, not language-driven; prompt-steering edits are pointless.
  Attention (attn_map, t=1): expert's base-cam attention on the yellow-block patches (base patches (10,7),(11,7); rows 0-55,176-224 are
  letterbox padding) is ~0.7% of base-cam mass (below uniform). In the wrist cam, L12 attends to yellow patches (the rubik's yellow face)
  at 3-10% of wrist mass (above uniform 2%). In the base cam the rubik face does not even register as yellow; only the block does.
  => expert seeks "yellow" in the wrist view (sees rubik face) and ignores the yellow block visible in the base view.
- a2 (47 items, base): 21 approach decisions that head to the yellow block (Y: 02k2-4, 19k1-3, re-target decisions 00k7-9, 16k9-11,
  27k9-10, 29k9-10, 10k7-8, 25k6, 18k9, 09k11), 12 approach decisions heading to rubik cluster (R), 14 carry decisions holding yellow (C).
  Action-expert resid_out (mean token) separates Y vs R with d'~2.8 (L0-11), jumping to ~4.5 at L13+ (L13 adds target info).
  Least-norm gate key in L12 MLP-hidden space (k.h_app=1, k.h_carry=0 on class means) responds Y~1.0, R~0.95, C~0 (range -0.45..0.4).

## Candidate p1 (pre-registration)
- Mechanism targeted: wrong target commitment during the empty-gripper approach (class A, 13/19 fails): policy commits to the
  yellow-topped rubik next to the red block. Language cannot redirect it (a1), so steer the expert's internal target state directly.
- Edit: action.blocks.12.mlp_1.linear += outer(k, delta), rank 1. delta = mean resid_out L12 (Y) - mean (R) (|delta|=10, resid ~90);
  k = least-norm key with k.mean_h(approach)=1, k.mean_h(carry)=0, so the shift is applied only while approaching with empty gripper.
  Size: rank-1, rel L2 0.155 of the tensor, max|dW| 0.24 (one tensor, 4.19M entries touched).
- Caveat: this is scene-layout-specific steering (yellow block sits at the same place in every slot), not a perceptual fix.
- Prediction (before offline check): 16/32 (fix ~5 class-A, break ~2 via grasp/stack disruption) -- to be updated after v1.

## Offline validation v1 (44 items, cand p1): REJECTED, not evaluated closed-loop
- 20 rubik-approach decisions, 14 carry, 8 grasp-at-yellow, 2 k=0: chunk-end fingertip targets move by <=0.002 m on average
  (max 0.01 m in z on one item); gripper commands identical. Carry/grasp also unchanged (good, but moot).
- Why: the edit is applied (L12 resid moves by |dr|~1.5-2.7 on approach items, ~0.1-0.8 on carry), but (i) the adaRMS gate on the
  MLP branch attenuates the injected delta ~5x (intended |delta|=10), and (ii) the Y-vs-R residual direction at L12 is apparently
  not causal for the target: downstream layers ignore it. The target information enters at L13+ (d' jump 3.7->4.8), consistent
  with it arriving through attention to image tokens rather than living in the L12 residual.
- Decision: do not spend 32 episodes confirming a no-op. Scaling alpha ~5x would mean a near-100% relative change of the tensor
  and still only ~0.01 m action shifts; steering at L16/17 would be a global approach-phase action bias that would also act during
  the descend-to-grasp on the yellow block (the gate is ~1 there too), which is likely to break grasps.

## Conclusion
Time budget (~2h) exhausted. Submit base. p2 not created. Episodes used: 32 (base only). Acts items used: 155/192.
