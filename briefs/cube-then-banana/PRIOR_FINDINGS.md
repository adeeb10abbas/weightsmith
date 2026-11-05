# Prior findings from an earlier editor (Opus) on the MIRRORED task

The task was `BananaThenRubiksCubeTask`, "put the banana then the cube in the bowl". It used the same scene, policy and tools, with base success 77.3%.
- That editor submitted no edit. Its reason: the mechanism it could isolate (mis-parsing an elliptical "X then Y" instruction, which causes premature cube releases) was worth only ~+3 pp on that task.
- **On your task the order is reversed and base succeeds only ~14%, so the same language mechanism may be the dominant failure here.** Verify everything on your own data.
- Activation IDs, episode IDs and candidate names below belong to the other project and are not accessible to you.
- Its image-token and attention captures were not available; you now have them.

---
## Its FINAL.md
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
- **a16h\*, a14h\*:** single-head ablations of action-expert attention in layers 16 and 14, built for causal analysis only. The GPU0 capture worker hung during this run, so I have no results.

## Why base
The one cleanly identified mechanism, the instruction mis-parse behind premature cube releases, has a ceiling of about +3 pp. None of my closed-form edits reproduced the parse fix selectively. The bigger failure groups are visual and physical (perched or wedged cube, banana dropped at the bowl edge), and I found no editable component that explains them. Submitting an edit that failed its own offline checks would risk regressions on held-out seeds, so I am submitting base.


---
## Its notebook (verbatim)
# Notebook

(The editor keeps this file current: findings, hypotheses, candidates, results, current best.)

## Environment notes
- Bash is restricted: only bare `owsp ...` commands are allowed (no pipes, redirection, loops, mkdir, python). Write scripts with the Write tool into `scripts/` and run `owsp py NAME scripts/x.py`. Specs in `specs/`.
- Frames are written to <agent dir>/frames/.
- Sandbox notes persisted in /work/notes: releases_base.json, grasps_base.json, items_release.json, a1_items.json, a2_items.json, a3_items.json.
- FK (Franka modified DH + 0.257 m flange/gripper offset) in scripts/releases.py gives EE xyz from joints.

## Scene geometry (fixed in every slot; EE coordinates from FK)
- Banana initial ~ (0.53, -0.055). Cube initial ~ (0.44, -0.065). Bowl ~ (0.455, 0.11).
- Gripper obs: holding the cube ~0.31-0.37; holding the banana 0.15-0.8 (depends on grasp); empty fully closed ~0.9+.
- Base policy goes for the banana first in essentially every episode (first grasp at banana location in ~127/128).

## Base practice (operator pre-run, slots s20000..s20127)
- 99/128 = 77.3% success.
- Failures (29): s20011 s20013 s20014 s20015 s20017 s20028 s20029 s20039 s20040 s20041 s20046 s20055 s20060 s20071 s20078 s20082 s20083 s20085 s20088 s20089 s20092 s20097 s20106 s20107 s20108 s20110 s20115 s20120 s20127
- Episode records in the sandbox have no events; events come only from `owsp show EID`.

### Failure taxonomy (final frames at step 885, events)
- A. Cube perched on the banana/bowl rim or right next to the bowl; policy then hovers idle ("thinks done"): s20040, s20078, s20082, s20106, s20015, s20120, s20039 (~7).
- B. Both objects in the bowl but cube placed BEFORE banana (after the banana was dropped mid-transport the policy switched to the cube), or banana on top: s20060, s20071, s20108, s20110, s20011 (~5). Sim shows "Completed 2/2" events but success=false.
- C. Cube phase runs out of time: repeated grasp/drop, premature releases, cube pushed away (bumped 0.1-0.2 m): s20013, s20017, s20028, s20029, s20041, s20046, s20083, s20085, s20089, s20092, s20097, s20115, s20127, s20055 (~13).
- D. Banana phase trouble: s20014, s20107 (~2-3).
- Successful episodes also often waste 100-500 steps on cube regrasps (e.g. s20000: cube 221->646).

### KEY FINDING 1: premature cube release is language-conditioned (acts a1-release-cf, a2-release-cf2)
- Premature release = the policy grasps the cube at its start location, lifts it a few cm, then on the very next replan commands the gripper open (cube still held, g~0.35), 15-25 cm from the bowl. ~33 events in 128 base eps; ~29% of first-location cube grasps end this way. Frames: s20000 222-255, s20003 225-262, s20020 195-232.
- a1 (29 premature-release replans, 19 carry replans with the cube held, recorded noise):
  - replay with the original prompt reproduces release in 28/29 (parity good, ~0.01 except near-boundary items).
  - prompt "put the cube in the bowl": release drops to 10/29 (34%). Carry items: 1/19 release.
- a2:
  - carry items (cube held) + prompt "put the banana in the bowl": 7/20 release (vs 0/20 original).
  - premature items + "put the cube then the banana in the bowl": 29/29 release.
  - banana-carry items (holding the banana in phase 1): original 0/20 release; with "put the cube in the bowl" 7/20 release.
  - initial approach (k=2) with "put the cube in the bowl": arm heads to the cube (x~0.41-0.46) instead of the banana (x~0.53).
- Interpretation: the policy computes a target object from language and scene; when the held object mismatches the target it tends to drop it ("wrong object -> release"). In phase 2 the target belief is sometimes still "banana", which produces premature cube releases. A crude cube bias would break phase 1 (approach cube first / drop banana).
- Action-expert residual (mean over tokens), denoise step 0: the prompt-driven difference is tiny through L12 and appears abruptly at L13 (|d| 2.3 -> 11.8), growing to L17. At L13-17 the orig-minus-cubeonly direction also separates premature from carry items under the original prompt (cos with the natural contrast 0.4-0.55; projection gap ~1.6-1.8 sd). Early-layer contrasts are dominated by prompt-length artifacts.

### Finding 2: release signal is unified in the action expert (a3-bowl-release)
- Direct-effect attribution on the gripper readout at denoise t=1: release push is written by attention in action layers L14 and L16 (plus a bit of L13 attn, L16 MLP). Groups: bowl release BR +0.26/+0.20 (L14/L16 attn), banana bowl release BBR +0.22/+0.11, premature P +0.08/-0.06, cube carry C -0.08/-0.27, banana carry B -0.10/-0.31.
- Attention-output difference vectors (P-Pc language-induced, Cb-C, Bc-B, P-C natural, BR-C bowl release) all have cosine 0.5-0.93 at L13/14/16. One shared "release" signal; mismatch and over-bowl evidence are integrated upstream (VLM).
- a4-idle-cf: in idle/perched-cube states, "put the cube in the bowl" does NOT make the robot act; "pick up the cube" makes it descend. Idle = completion misjudgement, not target confusion.
- a5-vlm-target: at the last prompt token, VLM L14-16, premature items look like the banana-prompt condition and carry items look like the cube-prompt condition ((C-P)/sep ~1.0). The decision is largely formed in the VLM, so it is input-driven, not just flow noise. Visually, premature releases tend to come from shallow grasps (cube high in the fingertips; frames s20004 225, s20009 240 vs carry s20005, s20019 225-240).
- Release-position analysis: good final cube releases center (0.456, 0.103), sd 1.6 cm. 16/29 failures contain a premature or edge release (cube or banana >5 cm from center), e.g. banana edge drops s20071 k8, s20110 k11, s20014 k14 leading to wrong-order failures.
- Rough ceiling: P(fail | premature cube release) = 9/27 vs 20/101 otherwise, so fixing premature releases alone gains ~+3pp at most.

## Candidates
- hold1: rank-1 edit of action.final_norm.Dense_0.kernel shift block = time-dependent gripper velocity bias of -1.0 at t=1.0, 0.9 and -0.5 at t=0.8 (0 at later steps), via the pseudo-inverse of action_out_proj so only dim 7 is affected. ~1.01M values (0.03%).
  - Hypothesis: mode selection happens early in denoising; weak (premature/edge) releases flip to hold, strong over-bowl releases survive.
  - Offline suite (acts hold1-s1, hold1-s2): P 12/29 suppressed, BR 9/30 suppressed (rest delayed), BBR 2/18, REOPEN after missed grasp 8/16 suppressed (bad), PRE-close 8/24 now close earlier, CLOSE earlier by 2-5 actions. NOT selective. Prediction failed. Rejected without a closed-loop eval. hold2 (beta -2) built but not evaluated (would be worse).

### Finding 3: the premature release is a parse error of the elliptical instruction (a6, a7, a8, a9)
- On the 29 premature-release replans (orig 97% release):
  - "put the banana then the rubiks cube in the bowl": 28% release.
  - "put the banana in the bowl then put the cube in the bowl" (V2, explicit sequence): 31% release.
  - Carry items: 0% release under both.
- Safety of rubiks phrasing (a7): banana carry B 3/20 now release (seesaw: moves the mismatch error into phase 1). BR 27/30, BBR 17/18 still release; INIT still heads for the banana.
- Safety of V2 (a8): B 0/20 release, BR 28/30, BBR 18/18, REOPEN 16/16, INIT unchanged. V2 fixes phase 2 with no phase-1 cost. So the model mis-parses the ellipsis "put the banana then the cube in the bowl".
- V2 on idle states and wrong-order decision states (a9): actions essentially identical to orig. Those failures are visual completion misjudgements, not language.
- So the language fix can only address premature cube releases (ceiling ~+3pp).

### seq1 (VLM L4 unit 8112 "then" detector, inject V2-orig delta at 'then')
- Unit 8112 in language.blocks.04: activation >=1.21 at 'then' in every item, <=0.15 at every other prompt slot (ratio 8.2). Delta(V2-orig) at 'then', L4 resid: |70| vs resid 612, consistency 0.96, cos(P items, B items) 0.95.
- Edit: language.blocks.04.mlp.linear[8112] += delta / mean_act (2047 values).
- Prediction: P release drops toward 31% with no side effects.
- Result (acts seq1-s1, seq1-s2): ZERO effect. P 29/29 still release; all other groups unchanged. Prediction failed: a single-token early injection does not carry the V2 effect.
- a11-vlm-seq-deep: at VLM L10-16 the V2-orig delta at content tokens (banana/cube/bowl) is item-dependent (cos(P-items, B-items) ~0 or negative, 60-90% unexplained by the mean); at end tokens it is small and inconsistent. V2's effect is an interaction with scene state, so a constant injection cannot emulate it. Abandoned the prompt-emulation route.

### sil1 (silence 81 VLM L12-16 MLP units that are premature-release-specific at the 5 end tokens)
- Selection used even-index P items: score = (mean_P - max over C/BR/BBR/B/PRE/CLOSE/REOPEN) / sd > 1.3 and mean_P > 0.5. Strongest: L16 unit 8894 at end-1 (P 6.45, C 0.85, BR -0.38, B 0.45), L16 6312, L14 3624, L15 13204. Zeroed their mlp.linear rows (81 x 2048 values).
- Prediction: P releases drop, others unchanged.
- Result (acts sil1-s1, sil1-s2): selection half 15/15 still release, held-out 12/14; all other groups unchanged. Prediction failed: these units are correlates, not causes. The decision probably reaches the expert via image-token features (not capturable).

### Head-ablation causal analysis (in progress)
- Candidates a16h0..a16h7: zero action.blocks.16.attn.attn_vec_einsum_1.w[h] (one head each). Built via the queue script scripts/abl_head.py (pops from /work/notes/head_queue.json; remaining queue: L14 heads 0-7).
- Probe items (/work/notes/abl_items.json): P 16, BR 12, C 6, B 8. Specs specs/a16h*.json; analysis scripts/an_abl.py.
- Goal: find a head whose ablation removes premature (language-mismatch) releases but keeps bowl releases and carries. If found, partially scale it and test closed-loop.

- STATUS: the a16h0-abl capture (and a 1-item ping on base) hung on the GPU0 worker for 30+ min; acts_calls stuck at 19. The background job running the L16 head probes 0-3 was killed at its time limit. The ping was still pending and acts_calls still 19 afterwards. Not restarted: a new capture would queue behind the hung one. Head-ablation results unavailable.

## Budget log
- Episodes used: 0 / 1600. Acts items: 1640 counted (plus the hung call).

## Current best
base (no edit). FINAL.md written with base and the full reasoning. Update it if the head ablations come back with a selective head.
