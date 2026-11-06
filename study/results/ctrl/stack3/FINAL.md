# FINAL (stack3: Stack3RubiksCubeTask, "Stack the rubiks cubes in a tower")

## Submission: `base` (no edit)

- Exact edit: none. No tensor values changed. Rank and size: 0.
- No candidate was created or evaluated (p1/p2 unused). Budget used: 32/96 practice episodes (base only), 84/192 activation items.

## Development result
- base: 16/32 on dev slots 70000-70031. Fixed 0, broke 0 (no edit).
- All 16 failures time out at 900 steps with score 0.5: the first two-cube stack is built and the third cube never ends up stacked.

## Mechanism story (diagnosis)
- The scene layout is nearly fixed across slots. Cube A is at about (0.50, 0.03), cube B at about (0.43, 0.19) and cube C at about (0.39, -0.09), in FK x/y metres. The usual plan is A onto C (by step ~110-180), then B onto the stack.
- Main failure, the stack-dwell / re-grasp loop. After releasing A on C, the gripper is empty and hovers over the stack. The policy then commands CLOSE at stack height (z ~0.18-0.26). It either closes on air in repeated loops (70001, 70010, 70018, 70021) or grabs and un-stacks the top cube (70019, 70003). Cube B is picked late (step 495-750) or never (70010, 70012, 70030).
  - Successes show the same loop but escape it sooner (70004, 70014, 70015).
- Secondary failures:
  - The third cube is placed misaligned (left leaning beside the stack), then the arm retracts and idles high (70006, 70017, 70028).
  - The cubes are stacked in a different order and the attempt falls apart (70016, 70020, 70027, 70029).

## Why no edit was submitted (offline evidence)
- I targeted the main loop with a gated gripper-open readout. The planned edit was rank-1 on `action.action_out_proj.kernel[:,7]`, with `bias[7]` centring, gated by a contrast direction. It was meant to fire only on "empty gripper, high at the stack, commanding close" decisions.
- Capture `a1-stackclose` covered 84 recorded decisions in four groups:
  - T: 22 target decisions.
  - P1: 24 table-level grasp closes that must be kept.
  - P2: 22 carrying decisions that must keep the gripper closed.
  - P4: 16 decisions in the same state as T that already command open.
- Results:
  - **Head-input mean-difference direction.** It separates the groups in-sample, but fails split-half validation by episode parity. Held-out targets overlap held-out grasp and carry decisions (carry projections reach 1.18 while targets span 0.29-1.13).
  - **Single action-MLP units (layers 8 and 17).** No unit separates T from P1/P2, even in-sample.
  - **Pre-final-norm action residual at layer 17.** It does carry a state feature that generalizes: held-out T and P separate cleanly (d' about 6 in one split, full separation in the other). After the final norm, however, the per-token gap is only about 0.04 against a per-token spread of about 0.03.
    - Flipping the target closes (Δa7 ≈ -0.5) would need a gain of about 50. At that gain, carry and grasp tokens would swing by about ±1.5 in the gripper action, i.e. mid-carry drops.
    - Gains of 2 or less change nothing.
- Conclusion: no closed-form readout edit was both effective on the target decisions and safe for grasping and carrying. On these numbers it would likely break more episodes than it fixes, so I rejected it offline instead of spending episodes.

## Suggested next step
Build a nonlinear gate: repurpose an action-expert layer-17 MLP unit. Set its gate and up rows to the pre-norm layer-17 state direction (with a bias-like offset from the T/P threshold), and set its down row to write along the gripper readout. Then validate with replay on held-out T/P items before closed-loop testing.

Confidence that base is the right submission given this evidence: moderate (about 0.6). Confidence that the diagnosis (stack-dwell re-grasp loop as the main failure) is correct: fairly high (about 0.75).
