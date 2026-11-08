# New result record: where to put the distilled edit (VLM, action expert, action head)

Filled from `study/continuation/RESULT_TEMPLATE.md`. Supports `README.md` in this directory. Nothing here was imported
into `paper/`.

- **Study/cohort name and date:** `opus-where-v1`, 2026-11-07. Operator session (Claude Opus) on the weightsmith
  runtime; a separate cohort from the paper's runs, the Astra study and the blind-editor study, not pooled with them.
- **Research question and hypothesis fixed before evaluation:** Does the location of the same closed-form instruction
  distillation (same 6 decisions, same teacher wording) matter? Hypothesis given by the study owner and committed before
  any closed-loop run (`README.md`, commit `e69ef19`): VLM edits transfer, action-side edits do not; six numbered
  predictions are in the README.
- **Policy family, exact architecture, checkpoint and prior task training:** OpenPI `pi05_droid_jointpos` (PaliGemma
  VLM 18 x 2048, Gemma action expert 18 x 1024 with adaRMS time conditioning, 10-step flow matching), DROID-trained, no
  RoboLab task training.
- **Simulator/hardware, source revision, runtime and checkpoint hashes:** RoboLab on Isaac Sim, `performance` preset,
  16 environments per process, two RTX 3090s shared with other projects through the GPU locks. Runtime: the GPU host's
  `~/weightsmith` (HEAD `a0cc1cd` plus the uncommitted runtime files used for the paper's runs; not modified here).
  Checkpoint per `third_party/pi05_droid_jointpos.sha256`.
- **Task, observation/action interface, horizon and success predicate:** RubiksCubeThenBananaTask ("Put the cube then
  the banana in the bowl", 900 steps); transfer PickGlassesTask and RubiksCubeLeftOfBowlTask, retention RubiksCubeTask
  (450 steps). Two 224 x 224 cameras plus joint and gripper state in; 15 x 8 action chunks out, gripper binarized at
  0.5. Success is RoboLab's terminal flag.
- **Failure type and independent evidence for that classification:** Language binding (paper Section 2 and
  Appendix C): 21 recorded premature releases, flipped by rewording. No independent rater.
- **Editor provider/model/version, effort, tools, editable tensors and budget:** No autonomous editor. The operator
  session built A1, H1 and VA (and the offline variants) in the `owsp py` sandbox from activation captures, in closed
  form, no gradients; V1-full, V1-0.16% and P1 are byte copies of existing candidates (fingerprints verified).
- **Exposure to other editors' results or held-out evidence:** The operator knew the paper's results and the blind
  editor's held-out result before the study. Slots 94016-94079 had never been used.
- **Probe parity, semantic annotation validation and repeated unchanged-policy controls:** Base replay of the 49
  recorded decisions: maximum absolute action error 0.084 against the recorded actions. The action-expert time
  conditioning was checked by rebuilding the captured head input from the layer-17 residual (relative error 0.3%). No
  unchanged-policy rerun was made in this study; the paper's noise floor (about 30% discordant pairs) applies.
- **Discovery/selection versus held-out split and evaluation order:** Construction used only the 6 recorded decisions
  and the 49-decision offline check (practice episodes 20000-20063). No arm was selected or tuned on closed-loop
  outcomes; every pre-registered arm was run on held-out slots 94016-94079 regardless of its offline result, in two
  32-slot chunks with the arm order reversed in the second (`ctl_run.py`). 12 of 14 chunks ran on one GPU lane; the
  pre-specified sensitivity analysis on the 32 slots unaffected by the lane count agrees with the main analysis.
- **Every proposal:** `README.md` tables. Closed loop: V1-full, V1-0.16%, A1, H1, P1, VA. Offline only: `a1-nogate`
  (Eq. 1 ignoring the adaRMS gate, no effect), `a1-late` (layers 13-17), `h1-grip` (gripper column only).
  Values changed and ranks: `results/diff_*.json`, `results/build_*.json`.
- **Selected incumbent and independent final score:** No selection; this is an ablation. Held-out on 64 slots against
  one unedited run (5/64): V1-full 55, VA 43, V1-0.16% 41, P1 30, A1 20, H1 0 (`results/heldout_analysis.json`, with
  Wilson intervals, paired bootstrap intervals and exact McNemar p).
- **Retention tasks, failure cases, and unavailable comparisons:** Retention (cube into bowl, 32 slots): 32, 32, 31
  (A1) of 32 against 32. Transfer: V1-full raised eyeglasses (25 vs 14) and cube left of bowl (20 vs 4); P1 and A1 did
  not. Not run: H1 transfer (H1 fixed nothing), VA and V1-0.16% transfer (GPU budget), the explicit-wording arm on
  these slots.
- **Fixed-input action effects versus closed-loop behavioral effects:** Offline, A1 flipped 20 of 21 releases with 2 of
  19 phase-two flips, yet in closed loop its first action chunk closed the gripper in about half of the episodes; H1
  flipped 19 of 21 with 6 phase-two flips and closed the gripper from the first chunk in 94% of the episodes; VA matched
  the teacher's action-expert state best at the 6 decisions but did worse than V1-full in closed loop. Offline checks
  on the 49 decisions did not include initial-approach states, where the action-side edits leaked.
- **Full discovery cost:** 5.90 GPU-hours on the two-GPU host (closed loop 5.59, activation captures at most 0.31),
  about 3.0 hours of the whole host (`results/gpu_time.json`); wall clock 14:47-20:56 with queueing. Operator session
  cost not metered.
- **Artifact paths, hashes and exact plot-generation command:** Projects `where-to-edit`, `where-glasses`,
  `where-left`, `where-cube` under `~/vibes/opus-weight-surgery/projects/` on the GPU host (captures, candidates,
  episodes, frozen copies). Fingerprints: `v1-full` 63db0c52a51306c9, `v1-img816` 5bf8c4329c1e416a, `p1-blind`
  4ca8023b0592b79f, `a1` 09b7c9f5bfe26912, `h1` d965cd706aed9dec, `va` 5ee25e8d2e78a555. Tables: `python3
  make_tables.py`; results section: `python3 write_results.py`; data refresh: `./collect_results.sh`.
- **Interpretation:** Measured: on this task and teacher, the VLM edit beats every action-side and stacked variant on
  held-out slots and is the only one that transfers. Mechanistic support (offline, 6 decisions): the VLM edit reproduces
  the teacher's action-expert computation with a 1.7% median weight change per layer, while the same copy inside the
  action expert needs |dW|/|W| near 1 and the least-norm head copy cannot separate states. Not shown: that no
  action-side construction can match the VLM edit (gated or preservation-keyed action-side edits were not built here),
  or that the pattern holds on other tasks, wordings or checkpoints.
