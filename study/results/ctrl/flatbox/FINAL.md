# FINAL: flatbox (PickUpGreenObjectTask, "Pick up the green vegetable block")

## Submission: `base` (unedited policy)

No edit is submitted. The one candidate I evaluated (p1) changed the decision it targeted, but it did not change the
outcome on the 32 development slots. I have no evidence that it would improve held-out success.

## Development results (slots 70000-70031, n=32)
| candidate | successes | fixed vs base | broke vs base |
|---|---|---|---|
| base | 0/32 | - | - |
| p1 (not submitted) | 0/32 | 0 | 0 |

## Failure analysis (base)
Success means grasping the flat green box and lifting it at least 5 cm (`object_above(..., z_margin=0.05)`). Base failed
32 of 32 times, but 31 episodes made contact (OBJECT_GRABBED). The failures happen in stages:
1. **Wrong yaw on the first approach (32/32).** The policy descends at yaw 0 and its jaws close along the box's long
   side, so the fingers land on top of the box. 17 of 32 episodes eventually rotate about 80-90 degrees (median step 300).
   Changing the wording ("box", "box of frozen vegetables") had no effect on this (capture a1, diagnosis only).
2. **Reopening when holding.** At a rotated straddle the jaws close on the box (observed gripper value 0.15-0.35, which
   becomes state token "40"-"90"). At the next replan the policy reopens in about 60% of cases (it commits in 20 of 42).
3. **A shallow grasp at the end of the box, which slips.** The frames (s70019 steps 380-425, s70005 steps 238-268) show
   the fingers on the near end or corner of the box, with the fingertips only about 1 cm below its top. Every
   full-commit lift that base produced slipped (the gripper reads 1.0 at the next replan).

## p1 (evaluated, not submitted)
- **Edit:** a rank-1 readout change on the gripper output only. `action.action_out_proj.kernel[:,7] += C*d` and
  `action.action_out_proj.bias[7] -= C*r`, with C = -1.25, 1021 values changed and |dK| = 1.25.
  - d is the unit-norm contrast of action-expert `head_input`: holding-box decisions (16) minus approach and open
    pre-grasp decisions (20).
  - Before taking the contrast, I projected out the top 40 principal directions of within-decision variation across
    tokens and denoising steps. These come from an SVD and use no labels.
  - r is the mean projection of the open pre-grasp class, so the edit adds no average shift there.
- **Offline replay (a3-p1, real denoising, recorded seeds):**
  - Holding decisions: g0>0.5 went from 6/16 to 11/16, and the close fraction from 0.39 to 0.70.
  - Out-of-sample open pre-grasp decisions: close fraction unchanged (0.29 to 0.29).
  - Out-of-sample approach decisions: all stayed open.
  - Joint outputs moved by 0.05 rad or less in most cases.
- **Closed loop:** 0/32. Commits at holding replans rose from 20/42 to 29/39, and p1 lifted the box-holding gripper to
  z 0.10-0.19 in 6 episodes. In every one the box slipped out during the lift. So the reopen decision was real, but it
  is not the binding constraint. The bottleneck is grasp placement and depth (stage 3), and fixing that needs
  state-dependent joint-output changes that I could not diagnose and validate within the budget.

## Confidence
High that p1 gives no real gain: it scored 0/32 with a clear mechanism for why (the box slips). That is why I submit
base. Expected held-out result for base: about 0/64.
