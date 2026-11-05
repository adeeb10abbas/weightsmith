# FINAL

## Chosen candidate: `base` (no edit)

No edit survived scrutiny. Every candidate I built was tested offline on recorded policy inputs. Each one either failed its mechanistic prediction (no effect) or was not selective, so it would trade one failure mode for another. None reached the point where a closed-loop practice eval was justified. I spent 0 of the 1600 practice episodes; I did not want to evaluate candidates whose own offline checks had already failed. That is why I submit the unedited policy.

- Tensors changed: none. Values changed: 0 (0% of the model).
- Practice results: base 99/128 = 77.3% on s20000..s20127 (operator pre-run). No candidate was evaluated closed-loop.

## What I found (mechanism story, with evidence)

### 1. Failure taxonomy (29 base failures, from event timelines and final frames)
- **A. Perched or adjacent cube, then idle (~7).** The cube lands on the banana or the bowl rim, or right beside the bowl. The policy then hovers with the gripper open until timeout. Examples: s20040, s20078, s20082, s20106, s20015, s20120, s20039.
- **B. Wrong order (~5).** The banana is released at the bowl edge (5–8 cm short of center) and falls outside. The policy treats it as done and puts the cube in first. Examples: s20060, s20071, s20108, s20110, s20011. These show "Completed 2/2" events but success=false.
- **C. Cube phase runs out of time (~13).** Repeated grasp and drop, premature releases, missed grasps on a cube wedged against the rim (s20097), and the cube pushed 15–35 cm away.
- **D. Banana phase trouble (~3).**

### 2. Premature cube release is caused by the elliptical instruction
"Premature release" means the policy grasps the cube, lifts it a few cm, then opens the gripper on the next replan, 15–25 cm from the bowl. It happens on about 29% of first cube grasps.
- Replaying the recorded inputs and noise reproduces it in 28 of 29 cases (acts `a1-release-cf`).
- With the prompt "put the cube in the bowl" it falls to 10/29. With "put the banana then the rubiks cube in the bowl" it falls to 8/29. With the explicit "put the banana in the bowl then put the cube in the bowl" (V2) it falls to 9/29 (acts `a1`, `a6`).
- V2 costs nothing in phase 1 (acts `a8`). Banana carries: 0/20 releases. Bowl releases: 28/30 cube, 18/18 banana. Missed-grasp reopens: unchanged. Initial approach: still the banana.
- The "rubiks" phrasing is a seesaw (acts `a7`): banana carries now release in 3/20.
- So the policy mis-parses "put the banana then the cube in the bowl". In stage 2 it sometimes still treats the banana as the target, and drops the held cube as the "wrong object". Swapping the prompts confirms the mismatch-release behaviour in both directions (acts `a2`): holding the cube under a banana prompt gives 7/20 releases; holding the banana under a cube prompt gives 7/20.
- The release decision is already formed in the VLM, before the flow noise. At the last prompt token in VLM layers 14–16, premature items look like the banana-prompt condition and carry items look like the cube-prompt condition (acts `a5`). In the action expert, the release push is written by attention in layers 14 and 16. Correct bowl releases use the same direction (cosine 0.5–0.9 between all release contrasts; acts `a3`).
- Ceiling: episodes with a premature release fail 9/27 of the time, versus 20/101 without. A perfect fix is worth about +3 pp, below what practice evals can resolve.

### 3. The larger failure groups (A, B) are visual completion misjudgements, not language
- In idle and wrong-order decision states, V2 and the cube-only prompt leave the actions essentially unchanged. Only "pick up the cube" makes the arm descend (acts `a4`, `a9`).

## Candidates tried (all rejected before closed-loop eval)
- **hold1** (rank-1 edit to `action.final_norm.Dense_0.kernel`, about 1.01M values).
  - Edit: a gripper-only velocity bias that applies only at the first denoising steps (t = 1.0, 0.9, 0.8), aimed at raising the release threshold.
  - Offline: premature releases suppressed 12/29. But correct bowl releases were also suppressed 9/30, missed-grasp reopens 8/16, and grasps closed earlier.
  - Verdict: not selective, rejected. hold2 (twice the bias) was built but never evaluated.
- **seq1** (one output row of `language.blocks.04.mlp.linear`, unit 8112, 2047 values).
  - Edit: unit 8112 detects the word "then" (activation ≥1.21 there, ≤0.15 on any other prompt token). I added the V2-minus-original shift at that token.
  - Offline: no effect at all, on premature releases or anything else.
  - Why: the V2 effect depends on the scene state at the deep-layer content tokens (cosine ~0 between premature and banana-carry items), so no constant injection can reproduce it.
- **sil1** (zeroed 81 MLP output rows in VLM layers 12–16, 166k values).
  - Edit: silenced units that fire specifically on premature releases at the final prompt tokens, selected on half the items.
  - Offline: selection half 15/15 still release; held-out half 12/14.
  - Verdict: these units are correlates, not causes. Rejected.
- **a16h\*, a14h\*:** single-head ablations of action-expert attention in layers 16 and 14, built for causal analysis only (not submission candidates). The GPU0 worker hung for a while, then recovered, and all 16 completed.
  - Probe set: 16 premature releases, 12 bowl releases, 6 cube carries, 8 banana carries.
  - No head selectively removes premature releases; the best flips 2 of 16.
  - Ablating layer-14 head 6 suppresses correct bowl releases (7/12 kept, gripper push +0.29 vs ~+0.68) far more than premature ones (14/16). It is the over-bowl release head.
  - Layer-16 head 5 and layer-14 head 3 carry "hold": ablating either makes 2 of 6 cube carries release.
  - So the language-driven release push is spread across heads, and no head-level edit can isolate it.

## Why base
The one cleanly identified mechanism, the instruction mis-parse behind premature cube releases, has a ceiling of about +3 pp. None of my closed-form edits reproduced the parse fix selectively. The bigger failure groups are visual and physical (perched or wedged cube, banana dropped at the bowl edge), and I found no editable component that explains them. Submitting an edit that failed its own offline checks would risk regressions on held-out seeds, so I am submitting base.
