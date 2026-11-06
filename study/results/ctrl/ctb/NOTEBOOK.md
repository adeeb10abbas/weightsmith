# Notebook (ctb: RubiksCubeThenBananaTask, "Put the cube then the banana in the bowl")

## Log
- Started base eval n=32 offset 0 (slots 70000-70031).
- BASE: 4/32 success (s70000, 70013, 70017, 70025).

## Base failure classification (32 dev slots)
Scene is nearly identical in every slot: banana lies just in front of the rubiks cube, bowl to the left (+y).
Robot descends to the cube (~x.42-.45,y-.06,z.07-.11 by FK) at replan k=3/4 (step 45/60).
Gripper obs: ~0.35-0.39 = holding cube, ~0.55-0.65 = holding banana, 1.0 = closed on nothing.
- A. Never closes on cube ("grasp-miss"): at k=3/4 gripper hovers at cube, chunk does not close (or starts closing then reopens), arm lifts/backs off and goes for banana -> WRONG_OBJECT_GRABBED (banana) ~step 100-170. ~15 eps: 01,02,03,05,07,08,12,14,15,18,19,21,24,26,30.
- B. Grabs cube, then releases it prematurely (next 1-3 chunks open gripper mid-carry, not over bowl) -> TARGET_DROPPED, then goes for banana -> wrong grab. ~13 eps: 04,06,09,10,11,16,20,22,23,27,28,29,31. Many release right at the replan after grasp (k=4 or 5, obs 0.36, chunk gripper 0.01).
- Banana, once grasped, is held firmly for many chunks (obs ~0.6, action ~0.85). Cube grasp/hold is "hesitant".
- Successes: close on cube at k=3/4, hold 3-4 chunks, release over bowl, then banana.
Hypothesis H1: the instruction's "banana" (adjacent to "in the bowl") pulls the policy toward the banana during stage 1; cube grasp/hold decisions are weak. Test: replay failing decisions with "Put the cube in the bowl" (diagnosis only).

## Session 2 (continuation; budget now 2 cands, 96 eps, 192 act items)
- scripts/replans.py -> /work/notes/replans.json (per replan: gobs, chunk gripper stats, FK xyz).
- Cube sits at FK ~(0.43-0.45, -0.06/-0.07, z .08-.12); banana grabs at ~(0.51-0.53, -0.12). Bowl release at y >= ~0.05.
- HOLD decision (gobs .31-.39 holding cube, at cube location) is bimodal: chunk gripper either ~0.75-0.9 (hold) or ~0.0 (release).
  Over all base replans at the cube spot with cube in hand: ~14 hold vs ~35 release. Banana in hand: always held (0.8+).
- GRASP decision at cube (gobs 0, z<=.12): ~17 no-close vs ~12 close. Non-closers move on to the banana and close there.
- Both failure modes = "policy does not commit to the cube".
- Capture a-diag1 (72 items): 16 release + 10 hold + 10 miss + 4 close under orig prompt; release+miss under "Put the cube in the bowl"; 6 hold under "Put the banana in the bowl". Meta in /work/notes/diag1_meta.json.
- RESULT a-diag1 (replay parity fine): release items orig g=0.00-0.03 (16/16 release). Same inputs with "Put the cube in the bowl": 16/16 HOLD (g .66-.83).
  Grasp-miss with cube prompt: only 1/10 flips (mean .00 -> .14), so misses are mostly not language-driven.
  Hold items with "Put the banana in the bowl": 5/6 still hold. So it is the compound "cube THEN banana" that makes a held cube get released
  (policy treats grasped cube as "step done, move to banana"). Attn mass (base,wrist,pad,prompt,act) does not differ between rel/hold.
- an2: at denoise step 0 (identical noise input) the prompt effect (cube-prompt minus orig) in the action expert residual is ~0 up to
  layer 12 and appears at layer 13 (|D|/|h| .07 -> .23), growing to .41 at 16; it aligns with the natural hold-minus-release contrast (cos .5-.8).
- an3: in velocity space the prompt effect is almost purely gripper: dv7 = -1.2 (step0) / -1.5 (steps 5,9), uniform over tokens; joint dv ~0.05.
  Normalized gripper: x7 final ~ -1 (open) vs +0.55 (hold); threshold x7=0. Need mean dv7 < ~-1.0 to flip.
  A head_input key built only from diag1 fires on any opening gripper at steps 5/9 (miss items .8) -> unsafe without preservation constraints.
- Capture a-pres2 (34 items, base): bowl releases (10), carry holds (6), stage-2 approach/banana grasp (14), initial approach (4).
- Plan p1: rank-1 edit of action_out_proj.kernel column 7: v7' = v7 + c7*(head_input . a), c7=-2, a = least-norm key with class-mean
  constraints per step: rel=1, bowlrel=0, stage2=0, approach=0 (scripts/build_p1.py).
- First version (build_p1.py) rejected offline: per-token key was noisy (step-0 head_input carries each token's noise sample);
  first-order prediction (a7_new = a7 - C7*key/2) scrambled chunks even for key~0 items.
- FINAL p1 (scripts/build_p1b.py): same constraints but key restricted to the complement of the top-96 PCs of within-chunk token
  deviations (99.8% of token-deviation variance), C7 = -1.8. Key token-std ~0.03.
  Offline first-order on captured base items: rel 16/16 flip to hold (key .82-1.22); miss 4/10 flip to close; hold/close/carry kept;
  bowlrel 0/10 blocked (keys -.30..+.30; s70000 k8 would release one chunk earlier); stage2 14/14 and approach 4/4 unchanged.
  Edit size: action.action_out_proj.kernel[:,7] only, 1024 values, l2 3.16 (rel. 0.69 of the whole kernel), rank 1.
  Untested risk: banana release over bowl in stage 2 (not in constraint set); mid-carry releases (not targeted); arm still in "banana mode".
- PREDICTION p1 (n=32, offset 0): 8-12/32 successes (base 4), fixed ~7, broke ~1-2. Cube-in-bowl count should rise sharply (base 8/32).
- RESULT p1: 10/32 (base 4/32), fixed 8, broke 2 (s70013, s70025). Prediction HELD.
  Cube-in-bowl (stage 1 done): 15/32 (base 8/32). Stage-2 conversion 10/15 (base 4/8), so the banana stage was not damaged.
  Early cube grasps now happen in nearly all episodes (GRABBED@46-97), but many still drop the cube ~30-60 steps later (mid-carry
  releases were not in the target set). That is the obvious next target; not pursued (time).
- Not running p2 (time budget exceeded). Submitting p1.
