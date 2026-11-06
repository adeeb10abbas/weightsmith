# FINAL: yellowred (StackYellowOnRedTask, "Stack the yellow block on the red block")

## Submission: `base` (no edit)

The one candidate I built (p1) failed offline validation. It was not run closed-loop, and I am not submitting it.

## Development result
- base: 13/32 on dev slots 70000-70031. Because base is the submission, the paired comparison is trivial: 0 fixed, 0 broke.
- p1: not evaluated closed-loop. Offline it was a no-op (see below).
- Budget used: 32/96 episodes, 155/192 activation items, 1 of 2 candidates.

## Failure diagnosis (base)
- The scene layout is the same in every slot. The dim lime-yellow block sits front-left at fingertip xyz ~(0.34, 0.40). The red block is at ~(0.51, 0.23). `rubiks_cube_1` sits right next to the red block at ~(0.43, 0.20), with its yellow face up. The other distractors are more Rubik's cubes, a blue block, figurines and a birdhouse with yellow trim.
- **Class A: wrong target, 13 of 19 failures.** In almost every episode the policy first goes to the yellow-topped Rubik's cube by the red block (sim steps 45-105). Successful episodes then lift and re-target the yellow block. Failures close on the Rubik's cube, the red block, the blue block, a figurine or the birdhouse, and wander. Only 2 of 32 episodes went straight to the yellow block.
- **Class B: yellow block grabbed but not stacked, 6 of 19 failures.** These involve repeated open/close cycles and drops at the yellow block.
- Note: the `WRONG_OBJECT_GRABBED 'yellow_block'` events are spurious. They fire on correct grabs.

## Mechanism evidence
- **Prompt replay (a1, 8 decisions x 8 wordings).** Wording has essentially no effect on where the arm heads. This holds for "Pick up the yellow block", "lime", "yellow-green", "...next to the green block", and even "Stack the green block on the red block": all give the same chunk-end targets and the same gripper closes over the Rubik's cube. Target choice is driven by vision and the policy's prior, not by language.
- **Attention.** The action expert puts about 0.7% of its base-camera attention on the yellow-block patches. That is below uniform, so it effectively ignores the block, which is clearly visible in the base view. In the wrist view, layer 12 concentrates on yellow patches, and those are the Rubik's cube's yellow face. This is the lookalike confusion: the policy seeks "yellow" in the wrist view and never reads the true block from the base view.
- **Residual contrast (a2).** Comparing approach decisions heading to the yellow block (21) with those heading to the Rubik's cluster (12), the expert residual separates them at d' ~2.8 through layer 11. The separation jumps to d' ~4.5 at layer 13, so target information arrives late, through attention.

## Rejected candidate p1: exact edit, for the record
- Tensor: `action.blocks.12.mlp_1.linear` [4096,1024], changed by `+= outer(k, delta)` (rank 1).
- `delta` = mean layer-12 resid_out over the yellow-heading decisions minus the mean over the Rubik-heading decisions. Its norm is 10; the residual norm is ~90.
- `k` = the least-norm key in MLP-hidden space with k·mean_h(approach) = 1 and k·mean_h(carry) = 0. Its measured response is ~1 on approach decisions and ~0 (-0.45..0.4) on carry decisions.
- Size: relative L2 change 0.155 of the tensor; max |dW| = 0.24.
- **Offline validation (v1, 44 items).** Chunk-end fingertip targets moved by at most 0.002 m on average, and gripper commands were unchanged.
- **Why it failed.** The edit is applied: layer-12 resid moves by ~2. But the adaRMS gate attenuates it about 5x, and the direction is not causal for the decision.
- **Why I did not scale it up.** A stronger version would either need an almost full-size rewrite of the tensor, or become a late-layer approach-phase action bias. Such a bias would also fire during the descent to grasp the yellow block and is likely to break grasps.

## Confidence
High that `base` is the right submission: no candidate showed a mechanism-level effect offline. A real fix would need to make the expert attend to the base-camera yellow block, for example by steering queries or keys toward its image-token features. That needs per-patch key features, which the capture tool does not expose, and more budget than remained.
