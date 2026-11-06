# FINAL: RubiksCubeLeftOfBowlTask ("Put the rubiks cube to the left of the bowl")

## Submission: `base` (no edit)

The only candidate I evaluated, p1, did worse than base (3/32 vs 5/32, fixed 3, broke 5), and it broke
grasping. I have no edit I can stand behind, so I am submitting the unedited policy.

## Development results (slots 70000-70031)
| policy | successes | fixed vs base | broke vs base |
|---|---|---|---|
| base | 5/32 (s70000, 70001, 70002, 70016, 70028) | - | - |
| p1 (not submitted) | 3/32 (s70013, 70014, 70030) | 3 | 5 |

## Diagnosis (evidence)
- **Failure modes.** In about 22 of 27 failures the cube is grasped reliably, but the carry has large +x
  (forward) components. The cube ends up at x ~0.55-0.66, behind the bowl. In about 4 failures the robot
  loops on grasp/re-grasp. Successes carry along +y at x ~0.44-0.51.
- **Prompt-swap diagnosis (a1/a2, 96 items, 12 carry-start decisions from failures).** I measured the mean
  chunk displacement (FK of the action chunk) for each wording:
  - left (+0.089, +0.088)
  - right (+0.084, +0.010)
  - front (+0.026, +0.034)
  - back (+0.116, +0.040)
  - in the bowl (+0.008, +0.111)
  - The wordings "to the left/right/front/back of the bowl" align token for token, differing only at slot 10.
  - The left-right contrast is pure robot +y (+0.005, +0.078), and the front-back contrast is pure -x
    (-0.090, -0.006). So the policy's relation words sit on robot-frame axes.
  - "Left", however, also carries a back-like +x component. Its diagonal matches image-left in the
    over-shoulder camera. That component is the forward drift seen in the failures.
- **Layer-0 write port.** Two layer-0 MLP units, 8023 (+9.0) and 3097 (+8.4), fire only on the token
  "left": |h| <= 0.2 at every other prompt slot and on the image grid. At layer 0, the left-right
  difference in the residual sits only at slot 10 and is identical across items.

## Candidates tried
1. **Layer-0 steering at the "left" token** (rejected offline, never evaluated).
   - Edit: delta = 0.5(L-R) + 0.8(F-B), written through rows 8023/3097 of
     `language.blocks.00.mlp.linear` (2 rows, 4090 values).
   - The injection was verified exact: |delta| = 87 at slot 10, about 0 elsewhere.
   - Behaviour did not move: mean carry (+0.090, +0.088) became (+0.091, +0.084), at the bf16 noise level.
   - Conclusion: downstream layers do not read additive changes to the layer-0 token representation
     linearly.
2. **p1: the same contrast measured and injected at layer 8, slot 10** (evaluated).
   - Edit: delta8 = 0.5(L-R)_8 + 1.0(F-B)_8, with |delta8| = 262 against |r10| = 754.
   - It is written through a least-squares combination of 64 slot-10-selective units:
     W[U] += outer(c, delta8) on `language.blocks.08.mlp.linear`.
   - The update is rank 1 on 64 rows: 130,764 values changed, max |dW| = 10.5, relative L2 of the tensor
     0.45. Leak gain is <= 4% on other prompt slots and <= 3% on image grid cells.
   - Offline (a4, 30 items): the carry dx fell from 0.090 to 0.061 and dy rose from 0.088 to 0.095, in the
     right direction but only about 1/3 of the linear prediction.
   - Offline side effect: 3 of 5 grasp-closing decisions flipped to "don't close" (s70016 k4, s70017 k3,
     s70024 k4).
   - Closed loop: 3/32. In 5 episodes the cube was never grasped, and many grasps came late. The grasp side
     effect dominated, so the prediction (about 6/32) failed.

## Mechanism story and why I submit base
The words carry the failure: the policy's "left" is diagonal (+x,+y) instead of robot-frame +y. Steering
the representation of the relation word is a real lever:
- the layer-8 injection moved the carry in the predicted direction;
- but changing the global instruction context also changes grasp-closure decisions, which hurt more than
  the carry fix helped.

A workable fix would need a write port gated by the carry phase (active only while the cube is held). I
did not have the time or the budget to build and validate one. Base is the honest submission.

## Budget used
- Episodes: 64 of 96 (base 32 + p1 32).
- Activation items: 156 of 192 (a1 72, a2 24, a3 30, a4 30).
