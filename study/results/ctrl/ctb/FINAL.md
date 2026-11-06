# FINAL (ctb: RubiksCubeThenBananaTask, "Put the cube then the banana in the bowl")

## Submission: **p1**

## Exact edit
- Tensor: `action.action_out_proj.kernel` [1024, 32]. Only column 7 (the gripper velocity) changes: 1024 values, rank-1 update.
  `K'[:,7] = K[:,7] + c7 * a`, c7 = -1.8, a in R^1024 (stored at /work/notes/p1_key.npy).
  Size: l2 of the delta 3.16 (0.69 of the whole kernel's norm; the column's own norm is 0.85). Max |delta| is 0.56. Nothing else changes.
- Effect: gripper velocity v7' = v7 + c7 * (head_input . a). The key `a` is a closed-form least-norm solution of class-mean
  constraints on recorded base activations (head_input, token-mean, at denoise steps t = 1.0, 0.5 and 0.1):
  mean(premature-release states) . a = 1, and mean(bowl-release) . a = mean(stage-2 approach / banana grasp) . a = mean(initial approach) . a = 0.
  The key is restricted to the orthogonal complement of the top-96 principal directions of within-chunk token deviations
  (99.8% of that variance). Those directions carry each action token's own noise/x_t sample, so this removes per-token noise from the key
  (token std of the key drops from ~0.3 to ~0.03). Built by scripts/build_p1b.py; no gradients and no per-sample fitting.

## Mechanism and evidence
- Base 4/32. Two failure modes, both meaning "the policy does not commit to the cube":
  (A) at the cube it does not close and goes for the banana (wrong-object grab);
  (B) it grabs the cube (gripper obs ~0.36) and opens again at the very next replan, still at the cube spot.
  Over base replans with the cube in hand at the cube spot: ~35 release vs ~14 hold. The chunk gripper is bimodal (0.0 vs 0.75-0.9).
- Causal test (capture a-diag1, diagnosis-only prompt replay). With "Put the cube in the bowl", 16/16 recorded premature-release
  decisions flip to hold (gripper 0.00 -> 0.66-0.83). With "Put the banana in the bowl", 5/6 cube holds stay holds.
  So it is the compound "cube THEN banana" that makes the policy treat a grasped cube as "step done, go to banana".
- Location: at denoise step 0 (identical noise input), the prompt-induced difference in the action expert is ~0 through layer 12.
  It appears at layer 13 and grows to layer 16, aligned with the natural hold-vs-release contrast. In velocity space it is almost purely
  the gripper (dv7 about -1.2 to -1.5, uniform over the 15 tokens; joint dv about 0.05).
- So the edit re-imposes the "hold" velocity on the gripper channel only, and only when head_input carries the premature-release
  signature. It was validated offline (first order) on 74 recorded base decisions:
  - Fires: 16/16 premature releases flip to hold; 4/10 grasp-misses flip to close.
  - Preserved: hold, close and carry decisions; 10/10 bowl releases (keys -0.3..+0.3); 14/14 stage-2 approach and banana-grasp
    decisions; 4/4 initial-approach decisions.

## Development result (slots 70000-70031, n=32)
- p1: **10/32** vs base 4/32. **Fixed 8, broke 2** (s70013, s70025; s70025 still placed the cube).
- Cube in bowl (stage 1): 15/32 vs base 8/32. Stage-2 conversion: 10/15 vs base 4/8, so the banana stage was not harmed.
- Pre-registered prediction (8-12 successes, ~7 fixed, 1-2 broke) held.

## Caveats
- Same-weight reruns flip about 30% of paired episodes. 8 fixed vs 2 broke is suggestive but not conclusive on 32 slots.
  The mechanism and offline evidence are the main support.
- Mid-carry releases (cube dropped 30-60 steps after the grasp) were not targeted and remain the main residual failure.
  The arm's "banana-mode" motion is not corrected; only the gripper channel is.
- The key was not constrained on the banana release over the bowl. Stage-2 conversion did not drop, so this seems fine.
