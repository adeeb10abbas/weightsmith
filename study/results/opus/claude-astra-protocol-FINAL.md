# FINAL — StackYellowOnRedTask sparse repair (Claude editor, adaptive mode)

## Selected: **base (no edit)**

Neither evaluated candidate passed the operator gate (strictly more than 12/32 successes AND at most 2 newly broken
base-success slots), so the gate-selected answer is the original checkpoint.

| candidate | edit | dev successes (base 12) | fixed / broke | gate |
|---|---|---|---|---|
| p1 | L17 `mlp_1.linear`, 64 rows, `row_add_direction` ±f·u, f = 0.004576 (1.0 pooled SD), post-bf16 L2 = 0.03709 | 16/32 | 8 / **4** | not eligible (broke > 2) |
| p2 | same rows/direction, f = 0.002288 (0.5 SD), post-bf16 L2 = 0.01886 | 11/32 | 6 / 7 | not eligible |
| p3 | not created (see below) | — | — | — |

Exact edits of the rejected candidates (both from base, layer 17 only, u = unit(mean yellow-first − mean rubiks-first)
of L17 `resid_out` at request k=0, averaged over denoise steps t∈{1.0,0.7,0.4,0.1}, capture `early-k0k2`):
- rows with +f·u: 79,148,536,548,597,708,752,955,1161,1228,1426,1519,1523,1698,1795,1978,2182,2186,2210,2300,2315,2317,2356,2418,2518,2642,2999,3095,3154,3426,3728,3965,4004
- rows with −f·u: 254,266,387,573,589,713,816,852,902,907,1006,1046,1087,1205,1581,1685,1785,2063,2100,2153,2328,2474,2494,2622,2772,3247,3383,3530,3611,3621,3743

## Mechanism account (short)
- **Failure mechanism (supported by events, frames and action traces):** the main failure is target confusion with
  yellow-looking distractors. The `rubiks_cube_1` has a yellow top face and sits on the reach path, and the birdhouse has yellow
  walls. In 13/32 base episodes the first grasp was the rubiks cube (3/13 succeed vs 9/17 when yellow was first).
  The choice is noise-driven in the first action chunks: at k=0 the chunk's wrist roll predicts yellow-first (d = +1.64).
  It is committed at steps 45–60.
- **Edit hypothesis tested (p1/p2):** steer the last action-expert layer along the reproducible k=0 yellow-first − rubiks-first
  residual direction (split-half cos ≈ +0.5), so that borderline noise samples fall into the yellow-going mode.
- **Outcome:** not supported.
  - The steer did not move the first chunk toward yellow-going actions beyond run-to-run variation.
  - Distractor failures did not decrease: never-grabbed-yellow went 11 → 11 (p1) and 13 (p2).
  - Success counts (16, 11) scatter around the base rate and are non-monotonic in dose.
  - Renderer noise makes the k=0 images differ between runs, so every candidate's trajectories diverge from the base recording
    within 1–2 requests. Paired outcomes are largely re-drawn, and the "≤2 broken" gate rejects edits without a large, robust gain.
- I did not run a third candidate. It would have been another dose of a falsified lever, effectively a lottery draw, with no
  expected generalization. No new evidence-backed hypothesis could be captured and analysed in the remaining time budget.

Budget used: 2 of 3 candidate evaluations; 64 of 96 activation items (one capture, `early-k0k2`); about 7,000 of 7,200 s.
Details: `NOTEBOOK.md`.
