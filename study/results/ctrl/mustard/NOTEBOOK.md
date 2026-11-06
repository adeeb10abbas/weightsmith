# Notebook (mustard: "Put the mustard in the left bin")

## Log
- Started base eval n=32 offset 0 (slots 70000-70031).
- [session 2] Base eval done: 6/32 success (s70015,19,26,28,29,30). Episodes max 450 steps.
  22 fail at step 1 (never object_grabbed mustard): 0,1,2,3,5,6,7,9,10,11,13,14,16,18,20,21,22,23,24,25,27,31
  4 fail at step 2 (grabbed, not in left bin / not detached): 4,8,12,17
- Failure analysis (FK on joints; +y = robot's left = left bin at y~+0.15..0.23; right bin at y<-0.15):
  * Mustard starts upright at x~0.59,y~0. First grasp ~step 45-60, usually at z~0.24-0.27 on the CAP/NECK
    (gripper obs ~0.58); a few lower body grasps (obs ~0.35-0.38) hold much better.
  * Cap grasp slowly slips (obs 0.58->0.8->1.0 within ~45-60 steps). Successes = carry + release before slip.
  * BIG: direction indecision after grasp. First post-grasp chunk moves RIGHT (dy<-0.08) in ~16/32 eps,
    then often flips back left; carry oscillates (s70007,08,10,11,14,20,28...). Some go fully to right bin
    (s70006, s70025 released at y~-0.23; s70009,22,31 slipped at y~-0.27). Time lost -> slip -> bottle falls
    over -> later grasps of lying bottle at z~0.11-0.15 mostly fail.
  * Hypothesis H1: left/right bin choice is weakly language-conditioned (flow sample picks a mode);
    strengthening the "left" signal should make carries direct and shorter -> fewer slips.
- Saved /work/notes/timeline.json, /work/notes/decisions.json (first & second post-grasp replans with dy).
- acts a1 (64 items): 32 first-post-grasp decisions x {left, right} prompts, all VLM+action layers, attn.
  RESULT (H1 confirmed, language is causal for direction but weak for "left"):
  * chunk dy at first post-grasp replan: "left" prompt 8 go left (dy>0.05) / 16 go right (dy<-0.05);
    "right" prompt 0 left / 27 right. Policy has a strong right-bin prior; "left" only partly overrides it.
  * gripper output unchanged by the prompt (grasp is not language dependent).
  * VLM: left-right contrast lives almost only at slot 8 ('left') and 9 ('bin') (rel 0.17 at L0 -> 0.4 at L13).
  * Expert: contrast direction appears in expert resid at layers 12-17 at denoise t=1 (proj corr +0.5 with
    final dy); by t=0.5 corr 0.8-0.94 (mode already chosen). Prompt attention mass 0.25-0.6 in layers 0-15.
  * VLM block-0 MLP has 'left'-token detector units 3097 (h=7.38) and 8023 (h=6.28); |h|<0.22 on every
    other captured text token (both prompts).
- p1 (built): language.blocks.00.mlp.linear rows 3097 and 8023 += gamma*dr0, dr0 = mean(r_left - r_right) at
  slot 8 after VLM block 0 (|dr0|=85, |r8|=489), gamma = beta/(h3097+h8023), beta=2 => the 'left' token's
  residual becomes r_left + 2*dr0 (a "more-left" instruction token). 4094 values changed (2 rows), rel L2 0.088
  of the tensor. Leak on other text tokens <=2.7 (vs 170 added at 'left'). Right-prompt unaffected.
  Mechanism: extrapolate the instruction's left-vs-right representation so the expert picks the left-bin mode.
- v1 capture (48 items, cand p1): 32 first-post-grasp decisions + 16 approach/grasp decisions; vlm L0 image
  tokens to check that units 3097/8023 stay silent on image tokens.
  RESULT (offline validation passed):
  * first-post-grasp chunk direction: base 9 left / 16 right -> p1 21 left / 1 right; mean dy -0.032 -> +0.085.
    Gripper command unchanged (|dgrip|<=0.03). p1 lifts a bit less (dz smaller) on some.
  * 16 approach/grasp decisions: max joint diff <=0.054 rad, gripper identical, dxyz ~identical -> preserved.
  * units 3097/8023 pooled image-token activation |.|<=0.27 (mean ~-0.01): no leakage onto image tokens.
- p1 EVAL PREDICTION (before running): mechanism = left/right bin mode selection; edit = 2 rows of VLM
  block-0 MLP down-proj (4094 values). Cap grasp slip is NOT addressed, so failures from slip-before-arrival
  and fallen-bottle regrasps remain. Predict ~11/32 (range 8-14), fixed ~7, broke ~2 vs base 6/32.
- p1 EVAL RESULT: 9/32 (base 6/32), fixed 7 (s70003,04,08,10,12,18,20), broke 4 (s70019,26,28,30).
  Prediction partly held: fixed=7 as predicted, but broke 4 (pred 2), total 9 (pred ~11, inside range).
  Mechanism held in closed loop: within 5 replans after first grasp, base 13 carried left / 14 went right
  (y<-0.08); p1 30 left / 0 right. p1 successes are fast (75-92 steps in 6 cases).
  New dominant failure: carry goes left but the arm lifts less (offline dz also smaller) and releases or
  slips at the near rim of the left bin (y~0.15-0.2), e.g. s70028 released at the rim -> bottle on the
  table; s70030 bottle swung on a cap grasp and slipped over the bin edge. Cap-grasp slip still unaddressed.
- No p2 (time budget ~2h exceeded; queue waits ~25-30 min per job). Submitting p1.
