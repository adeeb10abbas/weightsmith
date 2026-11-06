# New result record: Claude Opus cohort on the weightsmith runtime

Filled from `continuation/RESULT_TEMPLATE.md`. Supports `blog/failure-levels.md`. Nothing here was imported into `paper/`.

- **Study/cohort name and date:** `opus-weightsmith-v1`, Nov 4–5, 2026. Separate from the Astra original and fixed16 cohorts; not pooled.
- **Research question and hypothesis fixed before evaluation:** Which failure levels of a zero-shot generalist VLA can sparse, gradient-free weight edits repair? Pre-stated hypothesis: local motor failures are editable, perceptual target confusion is not. Each candidate's hypothesis is in its editor notebook, written before evaluation.
- **Policy family, exact architecture, checkpoint and prior task training:** OpenPI `pi05_droid_jointpos` (PaliGemma VLM, 18×2048, plus a Gemma action expert, 18×1024, 10-step flow matching; ~3.35B parameters). DROID-trained. **No RoboLab task training.**
- **Simulator/hardware, source revision, runtime and checkpoint hashes:** RoboLab on Isaac Sim, `performance` preset, 16 envs per eval. One RTX 3090 (the second card is power-limited and unused for evals). weightsmith top-level runtime; code files identical to commit `f503fb3`. Checkpoint per `third_party/pi05_droid_jointpos.sha256`.
- **Task, observation/action interface, horizon and success predicate:** 16 RoboLab tasks (list in `blog/failure-levels.md` §4). Two 224×224 cameras plus joint/gripper state in; 15×8 action chunks out (7 joint deltas, 1 gripper binarised at 0.5). Horizon 450 or 900 steps by task. Success is RoboLab's boolean terminal flag.
- **Failure type and independent evidence for that classification:** RoboLab events and recorded gripper commands (`scripts/classify_failures.py`). The wrong-object event is known to produce false positives (Astra audit). Level assignments for the five case-study tasks rest on editor diagnoses from kinematics, frames and activation captures. No independent rater.
- **Editor provider/model/version, effort, tools, editable tensors and budget:**
  - Editor: Claude Opus through Claude Code 2.1.x, `--effort xhigh` (stacking) or `max` (green box), headless.
  - Tools: Bash restricted to `owsp *` plus file tools; all 730 tensors editable.
  - Budgets: 1,200 (stacking) or 800 (green box) practice episodes and 6,000 activation items.
  - Some edits were built by the operator session (also Claude Opus) under the same rules: `gb*`, `wr*`, `gthr04`, `nf-*`.
- **Exposure to other editors' results or held-out evidence:** Editors received a `PRIOR_FINDINGS.md` of general lessons from earlier editors on other tasks. No Astra outcomes. No held-out evidence (none exists).
- **Probe parity, semantic annotation validation and repeated unchanged-policy controls:** Activation captures check action parity against recorded base actions (typical error 0.03–0.04). Annotation validation was not performed. Unchanged-weight evidence: two same-weight runs of `combo1` agree on 46/64; near-neutral `gthr04` flips 70/192 pairs on headroom tasks.
- **Discovery/selection versus held-out split and evaluation order:** Practice slots only (stacking 20000–20063; green box and mustard 60000–60063; `thr-*` 60000–60031). No candidate qualified for held-out scoring (slots ≥ 80000).
- **Every proposal:** listed in `results/opus/comparisons.json`, with candidate notes in each project's `cands/*/meta.json` on the GPU host.
  - Two mustard neuron-flip candidates (`nf-l1415`, 131,072 values; `nf-l1316-256`, 1,048,576 values) were stopped by offline checks and never rolled out.
  - Stacking: `kp*`, `kyb*`, `v1a3`, `v1am1`, `dk2`, `dv2`, `dkvm1` and `hr13b10` were built but never evaluated closed-loop.
- **Selected incumbent and independent final score:** None selected. Every closed-loop candidate was ≤ base or within noise.
- **Retention tasks, failure cases, and unavailable comparisons:**
  - `gthr04` retention: cube into bowl 16/16, banana onto plate 16/16, butter onto raisin box 14/16, all unchanged.
  - No physical robot. No task-fine-tuned checkpoint. No fixed16 runtime run (needs two GPUs).
- **Fixed-input action effects versus closed-loop behavioral effects:**
  - `gthr04`: +0.109 mean gripper command; closed rows 674 → 781 of 1440; closed-loop 117/240 vs 117/240.
  - `nf-l1415`: 0.0005 rad mean joint change and 0/96 decision flips, so not run.
  - `wr2gb4`: graspable wrist pose 9/16 → 16/16 in closed loop, but 0 successes.
- **Full discovery cost:**
  - Recorded editor API spend: about $799 (stacking editor $632; first project $94; green-box editors $73).
  - Operator sessions and GPU time were not metered.
  - About 1,400 unique simulated episodes.
- **Artifact paths, hashes and exact plot-generation command:**
  - `python3 results/scripts/plot_opus_forest.py`
  - `python3 results/scripts/make_mustard_strip.py <owsp frames sheet s60000> <owsp frames sheet s60013>`
  - Raw data: `~/vibes/opus-weight-surgery/projects/{stack-yellow-red, cube-then-banana, green-*, mustard-*, thr-*, claude-rq123}` on the GPU host.
- **Interpretation:**
  - **Measured:** no sparse weight edit improved closed-loop success at levels L1, L3, L4 or L5 on this checkpoint and runtime. Unchanged-weight variability is about 28–36% discordant pairs.
  - **Scope limits:** one zero-shot generalist checkpoint, simulation only, practice slots.
  - **Unresolved hypotheses:** L2 (motor decision within a learned skill) and L6 (language binding) may be editable; neither was tested closed-loop here.

## Addendum (Nov 5–6, night): language-binding edit `v2mi-f`, RubiksCubeThenBananaTask

- **Hypothesis fixed before evaluation:** the premature cube release is caused by the VLM binding "in the bowl" to the adjacent noun. Reproducing, inside the weights, the residual changes that the explicit instruction ("Put the cube in the bowl then put the banana in the bowl") produces should keep the cube, without changing phase-2 behaviour.
- **Edit:** 17 VLM MLP down-projections (`language.blocks.00–16.mlp.linear`), each updated with `dW = K^T (K K^T + lam I)^-1 V`.
  - Keys: 72 per layer. These are mean original-prompt MLP activations at 40 aligned prompt/state slots plus 32 image cells (2 cameras × 16), from 6 recorded premature-release decisions (capture `c2-multi`).
  - Values: per-layer residual increments under the explicit prompt minus the original.
  - `lam = 1e-3·tr(KKᵀ)/72`, alpha 1. 450,468,837 scalars changed (13.4%); rank ≤ 72 per layer.
  - Built by the operator session (Claude Opus), not by an autonomous editor.
- **Offline attempts before it (all on recorded decisions, no rollouts):** `st11a5` and `st1016a4` (axis steering, 21/21 still release); `v2s-b` (mean state shift, 6/6 release); `v2m-h`, `v2m-f` and `v2m-2` (40-slot per-token shift: 15, 13 and 8 of 21 release); `v2t-f` (with text slots: 13/21); `v2mi-f` (with image cells: **0/21**; carries 0/9 and all 19 phase-2 decisions unchanged).
- **Practice, closed loop:** slots 20000–20031, 25/32 vs base 3/32 (22 fixed, 0 broke, exact McNemar p ≈ 4.8×10⁻⁷).
- **Held-out (operator, `weightsmith/heldout.py`, slots 90000–90127, order alternating by half):**
  - base 27/128 (Wilson 0.149–0.290), edit 106/128 (0.753–0.884);
  - 81 fixed, 2 broke, exact McNemar p = 7.2×10⁻²²;
  - paired bootstrap difference +61.7 pp [+52.3, +70.3];
  - mean steps to success 412 → 300.
- **Transfer, PickGlassesTask:**
  - practice: 22/32 vs 12/32 (15 / 5), then on fresh slots 24/32 vs 14/32 (13 / 3);
  - held-out (slots 90000–90063): **44/64 vs 22/64**, 31 fixed / 9 broke, p = 6.8×10⁻⁴, Wilson 0.566–0.788 vs 0.239–0.466.
- **Other tasks (edit vs earlier base on the same slots):** cube into bowl 16/16 vs 16/16; banana onto plate 16/16 vs 16/16; stack yellow on red 17/32 vs 14/32; mustard 12/32 vs 7/32; bowls 21/32 vs 19/32.
  - Butter onto raisin box: 10/16 vs 14/16. The *unedited* rerun on the same slots scored 8/16 (0 / 6 vs the first base), so the apparent loss is within identical-weight noise.
- **Preservation variant `v2p-f`:**
  - Construction: image-cell keys only, plus 160 preservation keys (32 cells × 5 other tasks, mapped to zero), layers 0–16.
  - Offline: 1/21 release.
  - Closed loop: 24/32 vs 3/32 (21 / 0); butter 10/16. Not needed, given the identity result.
- **GPU lanes:** success by lane across tonight's runs was 249/448 (lane 0) vs 199/368 (lane 1); no lane effect.

- **Minimal edit `abl-imgl8`** (image cells only, layers 8–16, rank ≤32; 5.3M factored parameters):
  - practice 25/32 vs 3/32 (22 / 0);
  - held-out slots 90000–90127: **102/128 vs 19/128**, 86 fixed / 3 broke, p = 3.8×10⁻²².
- **Transfer map (32 paired slots each, base then edit in the same session):** cube left of bowl 22 vs 3 (20 / 1); right of bowl 0 vs 0; stack three cubes 19 vs 14; block tower 14 vs 15; yogurt 2 vs 3; bagel 1 vs 0; marker 1 vs 0; flat green box 0 vs 0.
- **Cube left of bowl, held-out (slots 90000–90063):** **36/64 vs 8/64**, 33 / 5, p = 4.3×10⁻⁶, Wilson 0.441–0.677 vs 0.065–0.228.
