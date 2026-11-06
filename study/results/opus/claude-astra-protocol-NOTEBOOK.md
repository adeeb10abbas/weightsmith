# Notebook — StackYellowOnRedTask sparse repair (Claude editor, adaptive mode)

## 0. Setup / budget
- Base (original policy) on dev slots 20032–20063: **12/32 success** (37.5%).
- Budget: 3 candidate evals (32 eps each), 96 activation items, 7200 s.
- Only editable tensors: `action.blocks.LL.mlp_1.linear` [4096,1024]; ≤128 changed rows over ≤2 layers.
- Gate: strictly > 12 successes AND ≤ 2 newly broken base-success slots.

## 1. Evidence: base episode events (`owsp show`, all 32 read)
Note: `WRONG_OBJECT_GRABBED_FAILURE: 'yellow_block'` fires in *successful* episodes too
(target-group naming artifact) → ignored. Real distractors: `rubiks_cube_1`, `birdhouse`, `lizard_figurine_01`, (red_block).

First object grabbed (step) — S=success:
| group | slots | successes |
|---|---|---|
| R: rubiks_cube_1 grabbed first | 35,38,39,40,43,44,47,48,53,55,58,62,63 | 3/13 (40,53,58) |
| Y: yellow first (or red/multi first then yellow: 59, 52) | 32,33,34,36,37,41,42,46,49,50,51,52,54,56,59,61 (+59 D, 52 M) | 9/17 |
| N: never grasped any target | 45,57,60 | 0/3 |

Failure taxonomy (20 failures):
- **Distractor grasp (10)**: 35,38,39,43,44,47,48,55,62,63. rubiks grabbed at steps 55–195; 8 of them never grab yellow (score 0), 35/39 grab yellow late (393/644) and fail to stack.
- **No grasp / long hesitation (3+)**: 45 (no events at all), 57 (only empty closes), 60 (birdhouse bump, empty closes); also 34 (first yellow grab @794), 46 (@897).
- **Grasped yellow, not stacked (5)**: 33 (many regrasps), 36 (dropped @198, idle to 825), 49 (repeated empty closes), 52 (multi-grab then regrasps), 61 (dropped @470).

## 2. Evidence: frames (`owsp frames`)
- `frames/base.base.s20042_0-60-120-135-165-197.jpg` (fast success): arm goes over the middle of the table (red block,
  rubiks cubes) at step 60, then right/forward to the yellow block (front, next to green block & dinosaur), grasps @135, stacks @197.
- `frames/base.base.s20044_0-30-45-59-90-150.jpg` (R failure): by step 30 wrist cam is centered on `rubiks_cube_1`, which has a
  **yellow top face + blue sides**; from above it looks like a yellow block. Arm descends and closes on it @59.
- Interpretation: distractor = visually "yellow cube" seen top-down; it lies on the reach path, closer than the yellow block.
- `frames/base.base.s20045_30-60-120-240-480-840.jpg` (N failure, no events): steps 30–60 centered over the yellow-topped rubiks
  (gripper channel 0.7 at k=3 but no grab), step 240 over the **birdhouse (yellow walls)**, then wanders raised until 900.
  → the "never grasped" mode is the same yellow-distractor confusion, just without a successful grab of the distractor.

## 3. Evidence: obs()/action summaries (scripts/traj1.py, traj2.py; no activation budget)
- Requests every 15 steps (k=0 at step 0). Initial joint state and scene identical across all 32 slots → k=0 observation identical;
  only the policy noise differs.
- Grasp configs: yellow j=[.50,.34,.34,-2.24,-.05,2.48,.32] vs rubiks [.22,.16,.20,-2.50,.06,2.63,-.11] (rubiks tight sd).
- R-first episodes start closing the gripper at **k=3–4 (steps 45–60)**; Y-first ones keep moving (j0 base yaw grows to 0.4–0.5 by k=5–6 in several).
- **k=0 chunk (identical obs, noise-only)** already predicts outcome: chunk-end Y−R diff in j6 (wrist roll) = +0.19 rad (d=+1.64);
  j0 −0.04 (d=−.78), j1 −0.06 (d=−.71), j2 +.03 (d=+.5). Success−failure at k=0: j6 +0.15 (d=+1.23), j2 d=+.75.
- k=2..5: Y−R j0 d=+0.5→+1.2, j2 d=+0.8→+1.2, j1 d≈−0.6 (yellow-goers rotate base further & differ in arm posture).
- => target choice is a noise-driven mode selection at k=0–2 that is committed at k=3–4.

## 4. Gate-risk reasoning (before any edit)
- Episodes are closed-loop and contact-rich; an edit that perturbs actions everywhere will partly re-randomize outcomes.
  With 12 base successes, full re-randomization at p≈0.4–0.5 would break ~6 of them → gate fails (≤2 allowed).
- So the edit must be (a) small, (b) concentrated in units that matter in the early reach / target-choice phase.

## 5. Activation captures
- **C1 `early-k0k2`** (spec `specs/early_k0k2.json`): all 32 base episodes × requests k=0 (step 0, identical obs) and k=2 (step 30,
  before the rubiks commitment at k=3–4) = **64 items**; all 18 layers; denoise steps [0,3,6,9]; action tokens = mean.
  Purpose: phase-matched success−failure contrast in the target-choice window. 32 items kept in reserve.
  (Capture was slow: >10 min wall-clock.)
- Editable tensor row norms (base): ~0.50–0.60 per row in all layers (L17 0.50, L16 max 1.15).
- The client was killed at its background time limit, but the capture completed server-side (`acts: [early-k0k2]`;
  status still shows acts_items 0). **I count it as 64 items used → 32 remaining.**
- Sanity: replayed actions vs recorded chunks max |diff| = 0.035 (fp16 capture) → capture = the recorded requests.

## 6. Analysis of C1 (scripts/ana1.py, ana2.py, ana3.py)
**Null / negative findings (important):**
- Plain success−failure (12 vs 20) residual direction, averaged over k=0,2: split-half cos(even,odd) ≈ −0.1…−0.2 at all
  layers → *not reproducible*. In-sample d≈2 everywhere = high-dim overfitting.
- S−F directions at k=0 vs k=2 are anti-aligned (cos ≈ −0.2), per-unit SMD corr(k0,k2) ≈ −0.1…−0.2: the k=0 first-chunk
  difference (extra wrist roll) is partly *undone* by k=2 (Y−R chunk-end j6: +.19 → +.14 → +.05), so pooling k's mixes opposite signals.
- Per-unit SMD tails only mildly above chance (|SMD|>1.3: 8–12 units in L12–L17 vs ~1.5 expected under normal null; heavy tails likely).
- At k=2 no contrast (S−F or Y−R) reproduces across halves.

**Reproducible signal:** at **k=0** (identical observations, noise-only), the **yellow-first minus rubiks-first** (Y=17, R=13; N excluded)
direction is reproducible across both splits: cos(even/odd, first/second half) = +0.44/+0.59 at L17 t=0.1, +0.46/+0.60 at L16,
+0.27/+0.34 at t=1.0; S−F direction at L17 is stable across denoise steps (cos 0.82–0.88 adjacent steps).
I treat this as a success-minus-failure direction on the target-choice subtask (first grasp = instructed object vs distractor).
Action-level k=0 S−F split-half cos +0.24/+0.94; k=2 −0.46/−0.44 (again: only the first chunk carries a stable signal).

**Unit contributions (ana3):** L17 MLP units contribute only 0.07 (L16: 0.28) of the Y−R gap along u (gap 11.0, pooled sd 3.67 at L17):
the gap is carried in the residual from the noisy action input/earlier layers, not written by L16/L17 MLP units.
→ project-out "repair" of L16/L17 units would move R episodes by ≲1 unit (and my first sign-naive selection would even move them the
wrong way). So a direct steer is the only lever with a meaningful size at these layers.

## 7. Candidate p1 (created from base) — FROZEN before eval
- Note: "Steer L17 residual along k=0 yellow-first-minus-rubiks-first direction (+1 pooled SD) via 64 large sign-consistent L17 units".
- **Operation** (`scripts/build_p1.py`): layer 17 only. u = unit(mean_Y − mean_R) of `resid_out` L17 at k=0, avg over denoise
  steps t∈{1.0,0.7,0.4,0.1}. Rows = the 64 units with largest |mean activation| at k=0 among the 2716 units whose activation sign
  is identical in all 64 captured items × 4 denoise steps. `row_add_direction` with factor **+f** on the 33 positive-activation
  rows and **−f** on the 31 negative-activation rows, f = 0.004576 = 1.0 × pooled sd(3.665) / Σ|a| (800.9), so every episode's
  L17 residual moves **+1 pooled SD along u** (realized post-bf16: k0 +3.67 for Y and R alike, k2 +3.82).
- rows+ : 79,148,536,548,597,708,752,955,1161,1228,1426,1519,1523,1698,1795,1978,2182,2186,2210,2300,2315,2317,2356,2418,2518,2642,2999,3095,3154,3426,3728,3965,4004
- rows− : 254,266,387,573,589,713,816,852,902,907,1006,1046,1087,1205,1581,1685,1785,2063,2100,2153,2328,2474,2494,2622,2772,3247,3383,3530,3611,3621,3743
- `owsp diff p1`: 1 tensor (`action.blocks.17.mlp_1.linear`), 64 rows, 46,718 elements changed, **L2 = 0.03709**, max |Δ| 8.9e-4, rel L2 0.0011.
- **Mechanism hypothesis:** target choice is made by the noise-driven first chunk; the yellow-going mode has a distinct L17
  residual signature (→ extra +wrist roll, smaller j0/j1 in chunk 1). A constant push of 1 SD toward that signature should tilt
  borderline noise samples into the yellow-going mode, so fewer episodes commit to the yellow-topped rubiks cube at k=3–4.
- **Prediction:** rubiks-first episodes 13 → ~8–10; successes 12 → 13–15. Rough self-estimate P(pass gate) ≈ 0.3.
- **Possible breakages:** (i) it is a *constant* push, so later phases also get a small joint-target bias (~⅓ of the k=0 Y−R action
  difference ≈ +0.06 rad j6, ≈ −0.02 rad j1), which may hurt grasp/stack precision; (ii) every trajectory changes from step 0, so
  late/fragile base successes (41, 53, 54, 58, 40) may be re-randomized → >2 broken would fail the gate.
- **Pre-registered follow-up rules (written while p1 eval runs):** causal check = per-slot k=0 chunk difference p1−base (same obs,
  same seed ⇒ pure edit effect); expected +j6 at chunk end.
  1. p1 passes gate → keep as incumbent; any p2 must start from p1 or be a strictly-justified variant.
  2. k=0 effect as predicted, fewer rubiks-first, but >2 broken → p2 = same construction at 0.5 SD from base.
  3. k=0 effect negligible and outcomes ~unchanged → p2 = stronger push (2 SD).
  4. k=0 effect as predicted but rubiks-first not reduced → lever falsified; do not escalate it.

### p1 result (`owsp eval p1 --n 32 --offset 32`, fp=42f14e2042903e51, 11.3 min)
- **16/32** (base 12). Paired: **fixed 8** (35,38,39,43,48,63 — six of the ten base rubiks-first failures — plus 49,52),
  **broke 4** (37,40,50,54). **Gate: NOT eligible** (4 newly broken > 2).
- Causal k=0 check (`scripts/cmp_cand.py`; same obs + same seed ⇒ pure edit effect): chunk-end Δ(p1−base) mean over slots
  = [+.005, +.014, −.005, −.005, +.013, +.024, −.012] rad (sd over slots .013–.054). Predicted +j6/−j0/−j1 (yellow-going);
  observed is small (~7% of the Y−R j6 gap) and **mostly opposite in sign** → the "push toward yellow-going first chunk"
  mechanism is **not supported**. (Plausible reason: L17 residual encodes the flow *velocity* (noise−action); pushing toward the
  Y-group velocity signature need not move the integrated action toward Y actions.)
- Trajectories diverge (>0.02 rad) at k=1–2 in essentially every slot ⇒ outcomes largely re-rolled.
- Outcome classes: never-grabbed-yellow 11 → 11 (unchanged); grabbed-yellow-not-stacked 9 → 5; success 12 → 16.
  The gain is in post-grasp conversion (12/21 → 16/21), which I did not predict. With binomial sd ≈ 2.8 the +4 is ~1.4 sd:
  cannot distinguish a real effect from re-roll luck. Breaks were late/fragile successes (37,40,54 hit distractors; 50 grabbed but no stack).
- Verdict: prediction (fewer distractor-first failures) **not supported**; success gain unexplained; gate failed on breakage.

## 8. Candidate p2 (from base) — decision
- Rule 2's *action* (dose reduction) is the only follow-up that addresses the gate failure (breakage) without post-hoc retuning of
  the direction; its mechanistic premise failed, so p2 is explicitly a **dose test**: same direction/rows rule, **0.5 SD**.
  If p1's gain is edit-driven, half the push should keep part of it while re-rolling fewer base successes.
- **p2 — FROZEN before eval** (`scripts/build_p2.py`, created from base): identical rows/direction/sign rule to p1
  (layer 17; same 33 rows+ and 31 rows−), factor **f = 0.002288** (= 0.5 × 3.665 / 800.9); realized post-bf16 shift along u
  k0 +1.80 (Y and R alike), k2 +1.88.
- `owsp diff p2`: 1 tensor (`action.blocks.17.mlp_1.linear`), 64 rows, 35,210 elements changed, **L2 = 0.01886**, max |Δ| 4.9e-4.
- **Prediction:** k=0 chunk Δ ≈ half of p1's (≈ −0.006 rad j6). Trajectories still diverge, but later/less often →
  broken ≈ 1–3, fixed ≈ 2–5, successes 12–15. Self-estimate P(pass gate) ≈ 0.3. If successes ≤ 12, p1's gain was most likely re-roll luck.
- Possible breakages: same as p1 (re-roll of late/fragile successes; small persistent joint bias), at half size.

### p2 result (`owsp eval p2 --n 32 --offset 32`, fp=52815bd51936a899, 13.1 min)
- **11/32** (base 12). Paired: **fixed 6** (34,35,36,38,49,52), **broke 7** (37,40,41,53,54,58,59). **Gate: NOT eligible**
  (fewer successes, 7 broken). Score classes base {0:11, .5:9, 1:12} → p2 {0:13, .5:8, 1:11}.
- Prediction (12–15 successes, 1–3 broken) **not supported**. Pre-registered reading applies: successes ≤ 12 ⇒ p1's 16/32 was
  most likely re-roll luck. Dose–response is non-monotonic (1 SD → 16 with 8/4 fixed/broke; 0.5 SD → 11 with 6/7), both scattered
  around the base rate (37.5%, binomial sd ≈ 2.7).

### Why outcomes re-roll for any candidate (`scripts/cmp_k0.py`) — important correction
- Same seed at k=0 in base/p1/p2 (8012800 for s20032), **but the k=0 camera images are not bit-identical across runs**
  (`base_img`/`wrist_img` differ; renderer is noisy/grainy). So my "same obs + same seed ⇒ pure edit effect" check was
  confounded: k=0 chunk max|Δ| per slot, median: p1−base 0.044, p2−base 0.042, **p1−p2 0.013**; and p2's mean k=0 Δ vs base
  ([+.006,+.017,−.005,−.004,+.014,+.024,−.016]) ≈ p1's — i.e. the candidate−base first-chunk difference is dominated by
  run-to-run variation (rendering/lane numerics; the C1 capture replay also differed from the recorded chunks by up to 0.035),
  not by the edit. Trajectories then diverge at k=1–2 (median diverge k = 1 for p2).
- Consequence: against the fixed base recording, *any* candidate's outcomes are largely re-drawn; with 12 base successes and a
  ~40% per-slot success rate, ≤2 broken requires a genuinely large reliability gain, not a small steer.

## 9. Decision: no p3; final = base
- Remaining wall budget ≈ 20 min; a p3 eval takes ~13 min. More importantly, the only lever I had evidence for (k=0 Y−R
  direction steer) is falsified at the action level and its outcome differences are indistinguishable from re-roll noise; another
  dose would be a lottery draw on the same lever (contrary to the "no sweeps over arbitrary trials" rule) and would not be expected
  to generalize to unseen slots. No new evidence-backed hypothesis could be captured/analysed in the remaining time
  (a 64-item capture took ~20 min). 32 activation items and 1 candidate eval left unused.
- **Final: base / no edit.** p1 (16/32, broke 4) and p2 (11/32, broke 7) are both gate-ineligible.

## 10. Summary of what was learned
- Dominant base failure = yellow-distractor confusion (yellow-topped rubiks cube on the reach path; birdhouse), decided by the
  noise-driven first chunks and committed at k=3–4; secondary = post-grasp drops/regrasp loops.
- The k=0 yellow-first-vs-rubiks-first residual direction is reproducible in late action-expert layers (split-half cos ≈ +0.5),
  but L16/L17 MLP units barely write it (Σ contribution 0.07–0.28 of an 11-unit gap); it is carried in the residual from the
  noisy action input/earlier layers. A constant L17 steer along it did not move the first chunk toward yellow-going actions
  beyond run-to-run noise and did not reduce distractor-first failures (never-grabbed-yellow 11 → 11 in p1, 13 in p2).
- Plain success-minus-failure contrasts (12 vs 20) in the early window were not reproducible (split-half cos ≈ −0.1…−0.2).
- Methodological lesson: renderer noise makes base-vs-candidate pairing re-roll most slots; the "≤2 newly broken" gate therefore
  rejects small steering edits unless they confer a large, robust reliability gain.

