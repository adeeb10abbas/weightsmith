# FINAL: mustard ("Put the mustard in the left bin")

## Submission: p1

## Exact edit
- Tensor: `language.blocks.00.mlp.linear` (VLM block 0 MLP down projection, [16384, 2048]). No other tensor changed.
- Rows changed: 3097 and 8023 (2 rows x 2047 values = 4094 values; 1.2e-6 of all parameters; relative L2 change of
  this tensor 0.088; max |delta| 1.34).
- Update: `W[u] += gamma * dr0` for u in {3097, 8023}, stored as bf16.
  - `dr0` = mean over 32 recorded post-grasp decisions of (VLM residual after block 0 at prompt slot 8, the
    token '▁left', with the instruction "...left bin") minus (the same with "...right bin"). |dr0| = 85, |r| = 489.
  - `gamma = beta / (h3097 + h8023)` with beta = 2, where h = mean activation of the two units on the '▁left'
    token (7.38 and 6.28).
  - Net effect: the '▁left' token's residual after block 0 becomes r_left + 2*dr0, a "more-left" instruction token.
    The two units are lexical '▁left' detectors. On every other captured text token (both prompts) they stay below
    0.22, so leakage there is at most 2.7 against the 170 added. On image tokens, pooled activations stay at or
    below 0.27. With a "right" prompt nothing changes.
- Closed form: a mean contrast of recorded activations plus a scalar. No fitting and no search.

## Mechanism story and evidence
1. Base failures (6/32 succeed). The first grasp almost always closes on the cap or neck (gripper obs ~0.58).
   That grasp slowly slips over ~45-60 steps. Success needs a short, direct carry and a release over the bin.
   After the grasp, the policy is undecided between the bins. Within 5 replans of the first grasp, 14/32 episodes
   carried RIGHT (y < -0.08) and 13 went left. Some reached and released into the right bin (s70006, s70025).
   Many oscillated right and then left, and the bottle slipped on the way. The fallen bottle was then almost
   never re-grasped.
2. The direction is language-causal but weak for "left" (capture a1, 64 items). I replayed the first post-grasp
   decision of all 32 episodes. With "left" the chunk went left in 8 and right in 16. With "right" it went
   left in 0 and right in 27. The gripper output was unchanged. So the policy has a strong right-bin prior, and
   the "left" token only partly overrides it. The left/right contrast in the VLM sits almost only at the '▁left'
   and '▁bin' tokens. In the action expert it shows up at layers 12-17 at the first denoise step: its projection
   correlates +0.5 with the final chunk direction, and 0.9 by t=0.5.
3. Offline validation of p1 (capture v1, 48 items, same recorded inputs and noise):
   - First post-grasp decisions: base 9 left / 16 right, p1 21 left / 1 right. Mean dy went from -0.032 to +0.085.
   - 16 approach and grasp decisions: max joint change <= 0.054 rad, gripper unchanged, end-effector motion the same.
4. Closed loop confirmed the mechanism. Within 5 replans after the grasp, p1 carried left in 30/32 episodes and
   right in 0 (base: 13 and 14). p1's successes are fast (75-92 steps in 6 episodes).

## Development result (slots 70000-70031, n=32, paired with base)
- p1: 9/32 vs base 6/32. Fixed 7 (s70003, 04, 08, 10, 12, 18, 20). Broke 4 (s70019, 26, 28, 30).
- Caveat: +3 on 32 slots is inside the ~30% same-weight flip noise (sign test 7 vs 4, not significant).
- Remaining failure: p1 lifts a bit less during the carry. It now often releases or slips at the near rim of the
  left bin, where the bottle lands on the table. The cap-grasp slip is not addressed.
- The submission rests mainly on the mechanism: the wrong-bin carries are removed, and a carry to the right bin
  always fails. The success gain itself is modest and uncertain.
