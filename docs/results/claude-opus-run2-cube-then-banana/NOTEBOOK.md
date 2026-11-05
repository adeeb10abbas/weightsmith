# Notebook

(The editor keeps this file current: findings, hypotheses, candidates, results, current best.)

## Environment notes
- Bash: only bare `owsp ...` commands allowed (no mkdir/cat/pipes). Scripts in `scripts/` (Write tool), run `owsp py NAME scripts/x.py`. Specs in `specs/`.
- Frames go to <agent dir>/frames/.
- Sandbox persisted notes (/work/notes): fk.py (Franka FK, exec it), base_timelines.json (per replan: step, EE xyz, grip obs, frac of chunk commanded closed, chunk-end EE xyz), base_segs.json, items_base.json, sel_a1.json (PREM1/CCARRY1/BCARRY1/PREM2/CBR/INIT2 item lists).
- Gripper obs: holding cube 0.25-0.40, holding banana 0.45-0.8, open 0, empty-closed ~1.
- Scene (EE coords via FK): cube ~(0.44,-0.065), banana ~(0.51-0.54, -0.05..-0.12), bowl ~(0.455, 0.11).
- Prompt tokens: <bos> Task : ▁Put ▁the ▁cube ▁then ▁the ▁banana ▁in ▁the ▁bowl , ▁State : ... (cube=slot5, then=6, banana=8, bowl=11)

## Base practice (operator pre-run, s20000..s20063): 9/64 = 14.1%
Successes: s20026 s20029 s20030 s20038 s20040 s20042 s20044 s20050 s20061.

### Failure taxonomy (from FK grasp segmentation, scripts/segs.py)
- The initial approach goes to the CUBE in essentially every episode (EE at k=3 / step 45 ≈ (0.43,-0.065) in 63/64). So the approach DOES follow the order.
- First grasp object: cube in ~30/64, banana in ~34/64 (veered from the cube to the adjacent banana before closing).
- Of the phase-1 cube grasps (before any banana touch), 21 are PREMATURE releases (cube lifted a few cm and released on the very next replan, 15-20 cm from bowl; frames s20001 60-105) vs 9 carried to the bowl.
- After dropping the cube the policy takes the banana to the bowl -> wrong order -> failure (banana already in bowl, so "object_grabbed(banana)" step never satisfied unless re-grabbed).
- When the cube is carried to the bowl first: 7/9 success (s20011, s20060 failed later). So P(success | cube carried first) ~ 0.78.
- Lever: make the first cube grasp stick.

### KEY FINDING 1 (acts a1-orig, a1-cube, a1-v2, a1-rubiks): premature cube release is caused by the elliptical instruction
Items: 21 PREM1 (first-phase premature releases) + 9 CCARRY1 (first-phase cube carries).
- orig replay: PREM 21/21 release, CARRY 0/9 (parity exact on decisions).
- "Put the cube in the bowl": PREM 1/21 release (!), CARRY 0/9.
- "Put the cube in the bowl then put the banana in the bowl" (V2): PREM 2/21, CARRY 0/9.
- "Put the rubiks cube then the banana in the bowl": PREM 21/21, CARRY 1/9 -> naming is NOT the issue.
- Only item s20030 k7 (drop at x=0.60) releases under all prompts.
=> In "Put X then the Y in the bowl" the hold/release check treats Y (the noun adjacent to "in the bowl") as the current target; holding X -> "wrong object" -> release. The explicit form fixes it.

- a1-mirror ("Put the banana then the cube in the bowl"): PREM 3/21 release -> with the nouns swapped the held cube is KEPT.
- b1-parse: "Put the cube in the bowl then the banana" 3/21; "Put the cube then the banana" (no dest) 20/21; "Put the cube and the banana in the bowl" 9/21; "Put the banana in the bowl" 21/21.
- b2: "Put the cube before the banana in the bowl" 11/21; "Put the cube, then the banana in the bowl" 21/21.
=> Rule that fits all: the hold/release target = the NP bound to "in the bowl" (the adjacent one); w/o destination the last NP. The approach (first move) instead goes to the FIRST noun (cube) in 63/64 base eps.

### Finding 2 (acts c1-attn, orig vs mirror, 16 PREM1 items, full layers; scripts/recon.py, vlm_attn.py, img_attn.py)
- Expert residual difference orig vs mirror is ~0 through expert L12 and jumps at L13 (rel 0.03 -> 0.14 -> 0.30 at L16).
- Reconstructed expert->text attention (RoPE etc. from weights): the expert attends to state tokens / bos, almost never to the noun tokens; text-attn contributions explain only 2.7 of the 19.1 L13 delta difference -> the target info enters the expert mainly via IMAGE tokens (VLM-contextualized), plus state tokens at L14/L16.
- VLM text attention: "bowl"/"in" attend to the adjacent noun (slot 8) far more than slot 5 (e.g. L1h1 .69 vs .02, L5h2 .88/.11, L6h5 .82/.04, L9h5 .61/.06).
- Image->text attention (approx, cell-mean queries): image tokens attend to slot 8 noun >> slot 5 noun in many heads (L3h5 .30/.08, L6h4 .25/.05, L9h1 .13/.03, L9h3 .13/.02, L11h0 .15/.02).

### Finding 3 (acts c2-multi: 6 PREM1 items x 8 prompts, full layers; scripts/an_c2.py)
"Target-tag score" = projection on (cube-only minus banana-only) direction, normalized 0=banana-only, 1=cube-only.
- Image tokens (base & wrist) and state tokens, L10-16: orig 0.15-0.25 (release), nodest ~0.1 (release), and 0.4-0.6 (mixed 2/6), mirror 0.4-0.55 (hold), V2 0.6-0.8 (hold), cube-in-bowl-then-banana 0.6-0.8 (hold).
- Early layers L4-5: orig 0.3-0.38, mirror 0.06 -> early layers favor the FIRST noun; mid layers L6-11 shift toward the bowl-bound (second) noun.
- Release decision ~ threshold near 0.45 on this axis.

### Finding 4: closed-loop ceiling of the language fix (diagnostic eval, base@p89134f)
- `owsp eval base --n 32 --offset 0 --prompt "Put the cube in the bowl then put the banana in the bowl"` -> 22/32 = 68.8% vs base 3/32 on same slots (fixed 20, broke 1). Remaining fails: wrong-grab banana first or early cube drop.
- Phase-2 check (acts p2-orig/p2-v2/p2-cibtb/p2-cube; 7 approach + 5 grasp + 7 banana-carry items after cube in bowl): V2 and "Put the cube in the bowl then the banana" (cibtb) behave like orig in phase 2 (approach banana, grasp, carry) -> they are phase-aware; cube-only prompt wanders in phase 2.

## Candidates
(none yet)

## Budget log
- Episodes used: 0 / 1200. Acts items: 120.

## Current best
base (no edit yet).
