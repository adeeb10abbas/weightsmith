# Same Goal, Different Words → weightsmith (sgdw)

**Summary.**
- π0.5 shows the study's wording gap, sharply, on the goal where the world-action models showed it most: mustard
  bottle right of the raisin box (S4-R), TF 27/32 vs RF 0/32 on the same practice slots (27 TF-only pairs, exact
  McNemar p = 1.5 × 10⁻⁸). On S5-LR (left bowl on right bowl) it shows none (5/32 vs 3/32).
- TF is a known-good input for RF here: under RF the policy neither takes the bottle nor carries it right, and
  replaying the same inputs with the TF wording fixes both decisions.
- Distilling TF into the weights closes part of the gap on unseen slots. The paper's recipe (image-cell keys
  averaged over all decisions, VLM layers 8–16) fixed approach decisions but not carrying decisions offline; keeping
  keys separate per decision phase fixed both. `s4r-grp2` (2 phases, 64 keys/layer, 0.32% of parameters as factors):
  held-out 19/64 vs 0/64 (19 fixed, 0 broke, p = 3.8 × 10⁻⁶). `s4r-grp3` (adds a grasp phase, 96 keys/layer,
  0.48%): held-out 27/64 vs 0/64 (27 fixed, 0 broke, p = 1.5 × 10⁻⁸), against TF's 27/32 on practice slots. A
  random-direction update with `s4r-grp3`'s layers, rank and singular values scored 0/64.
- Under the TF wording the edits cost a net 4 of 27 practice successes (23/32 vs 27/32, p ≥ 0.34): not detectable at 32
  pairs, not excluded either.
- Phase C: Cosmos3 Nano's instruction reaches its actions only through text keys/values in each of 36 layers, and the
  TF and RF prompts align token for token, so the same edit is well posed; the checkpoint and environment are gone
  from ali-ms and evaluation with CPU offload costs about 21 minutes per episode, so a test is a multi-day job here.

## Question

"Same Goal, Different Words" (Abbas, CoRL 2026 workshop, arXiv 2610.16242) found that describing a placement from the
reference object's side (reference-first, RF) instead of the moved object's side (target-first, TF) lowers stable
placement by 12.2 points across three world-action models (Cosmos3 Nano −14.6, Cosmos3 Edge −8.1, FLUX 3 Action
−13.8). Weightsmith's closed-form instruction distillation copies a known-good wording's internal effect into the
weights. If TF is a known-good input for RF, distilling TF into the weights should close the gap. We test this on
π0.5 (the weightsmith runtime) and plan it for Cosmos3 Nano.

## The study's setup (Phase A)

Source: `adeeb10abbas/world-action-model-steerability`, branch `codex/nano-stock-workstation-20260926` at
`9a6fcde` (`experiments/robolab_workshop/`, `docs/robolab-workshop-20260926/`; study id RWS-20260926). It ran on a
cluster (`/data/users/ali/rws-20260926`, RoboLab `0aef241`); none of it is checked out on ali-ms. The directories
`~/sgw-scene-design-20260923` and `~/projects/steerable*` hold the earlier VLA-vs-WAM direction study and the
clean-scene (SGW) design, not this study.

- **Scenes and goals.** Four stock RoboLab scenes, twelve goals, robot frame (x forward, y left):
  S1 `rubiks_cube_banana_bowl.usda` (base task `RubiksCubeLeftOfBowlTask`, 30 s): cube left / right / in front of /
  behind the bowl; S3 `butter_raisin_box.usda` (`ButterAboveRaisinTask`, 40 s): butter box left / right / on top of
  the raisin box; S4 `mustard_raisin_box.usda` (`MustardAboveRaisinTask`, 40 s): mustard bottle left / right / on top
  of the raisin box; S5 `bowls_2_table.usda` (`BowlStackingLeftOnRightTask`, 20 s): left bowl on right, right on left.
  S1-R, S3-R and S4-L are already true at reset (maintenance goals). A planned S2 (two bins) was dropped.
- **Wordings** (`planned_episodes.jsonl`, codes D/S/I): DIR (direct or native RoboLab wording), TF ("Move the
  mustard bottle on the table so that the mustard bottle is to the right of the raisin box."), RF ("Move the mustard
  bottle on the table so that the raisin box is to the left of the mustard bottle.").
- **Design.** 3 models × 12 goals × 8 starting states (C01–C08) × 3 wordings = 864 rollouts, 15 Hz, policy seed 6100,
  32-action chunks.
- **Score.** Stable placement: the relation holds (45° cone in the robot frame for left/right/front/behind), the moved
  object touches its support, the gripper does not touch it, and it moves slower than 2 cm/s, for one continuous
  second (`scoring.py`, `sim_env.study_predicate`). It checks the relation, not which object moved.
- **Largest TF − RF gaps** (stable placement at any time, pooled over the three models, of 24; site data
  `episodes.json`): S4-R 20 vs 2, S5-LR 8 vs 1, S1-L 8 vs 4, S3-TOP 6 vs 2; S1-B goes the other way (1 vs 7).
- **π0-family.** No π0-family policy was run on these goals ("There is no pi baseline in the primary study",
  `EXPERIMENT_SPEC.md`; the TMLR plan proposes π0.5-DROID as a later phase). Earlier, separate studies ran π0.5 and
  π0-FAST on cube left/right of the bowl with direct, paraphrased, declarative and contrastive wordings (π0.5 3/40 left
  vs 22/40 right; π0-FAST 1/10 vs 10/10), but never a reference-first wording. So there was no π0-family TF/RF gap
  before this work.
- **World-action-model checkpoints** (none is on ali-ms now; see Phase C): Cosmos3 Nano Policy DROID
  (`nvidia/Cosmos3-Nano-Policy-DROID@6706d76`, 33 GB, bf16 diffusers safetensors, 15.2B-parameter transformer),
  Cosmos3 Edge Policy DROID (`@a7c7288`, 4B, about 9 GB), FLUX 3 Action DROID (7B, bf16; cluster only). In the
  paper all three were served on the cluster (Nano on a B200 node) behind `experiments/robolab_workshop/policy_server.py`.

## Making the goals runnable for π0.5 (Phase B setup)

- **Task modules.** [`tasks/mustard_right_of_raisin_rf.py`](tasks/mustard_right_of_raisin_rf.py) (S4-R) and
  [`tasks/bowl_left_on_right_rf.py`](tasks/bowl_left_on_right_rf.py) (S5-LR) reuse the stock scenes, objects and
  horizons and replace the success term with the study's stable placement (relation, support, detached, < 2 cm/s
  for 1 s, evaluated once per control tick). Their default instruction is the RF wording, so the task's own
  instruction is the one being repaired; TF goes through weightsmith prompt overrides.
- **Least invasive registration.** Nothing in the shared RoboLab checkout changed. A separate clone of this branch
  on ali-ms (`~/sgdw/weightsmith`) adds `WSP_TASK_DIRS` to `weightsmith/run_batched.py`: a task named by class is
  registered from a module in those directories (RoboLab accepts task file paths) and selected by class name as
  usual. The clone's `wsp.env` is `~/weightsmith/wsp.env` with its own log directory, policy-server port 8400 (the
  other agent's servers use 8300/8301), GPU 0 as the primary lane and GPU 1 as the optional second lane, and the same
  shared GPU lock directory.
- **Projects** `~/vibes/opus-weight-surgery/projects/sgdw-s4r` and `sgdw-s5lr` (`prompt_overrides: true`,
  practice slots 20000–20255, 16 environments).
- **What differs from the study.** π0.5 sees its usual two 224 × 224 images (one exterior, one wrist) instead of the
  WAMs' three-view composite, it replans every 15 actions, and every slot starts from the stock scene's default
  layout (the study used eight different starting layouts per scene), so slots differ by policy noise and simulator
  nondeterminism only, as in all weightsmith projects.

## Results

### π0.5 shows the gap, and more sharply than the world-action models (practice slots 20000–20031)

| Goal | TF | RF | TF only / RF only | exact McNemar p |
|---|---|---|---|---|
| S4-R mustard right of the raisin box | **27/32** (Wilson 68–93%) | **0/32** (0–11%) | 27 / 0 | 1.5 × 10⁻⁸ |
| S5-LR left bowl on the right bowl | 5/32 | 3/32 | 5 / 3 | 0.73 |

Same 32 slots, unedited policy, wording changed through the prompt override only, success = the study's stable placement
([`results/screen_s4r_tf_vs_rf.json`](results/screen_s4r_tf_vs_rf.json),
[`results/screen_s5lr_tf_vs_rf.json`](results/screen_s5lr_tf_vs_rf.json)). On S4-R the RF − TF difference is
−84.4 points (95% bootstrap −96.9 to −71.9); the world-action models' pooled gap on this goal was 20/24 vs 2/24.
S5-LR is near the floor under both long wordings and shows no gap, so for π0.5 the gap is goal-specific; we distil
only S4-R.

**What RF gets wrong** (proprioception and replays; [`results/traj_s4r.json`](results/traj_s4r.json),
[`results/diverge_s4r.json`](results/diverge_s4r.json), [`results/diverge2_s4r_tf_episodes.json`](results/diverge2_s4r_tf_episodes.json)).
Under TF the hand goes left and down to the mustard (mean hand position at replan 4: y = +21.8 cm, z = 15.1 cm in
the robot base frame, y pointing left), closes at replan 4 (median) and carries the bottle right; all 32 TF episodes
grasp the bottle. Under RF the hand stays higher and nearer the middle (y = +9.7 cm, z = 20.1 cm), closes later
(median replan 13) or never (8/32), only 7/32 episodes ever grasp the bottle, and none places it. Replaying recorded
inputs with only the wording changed isolates two wrong decisions. On 96 replans (0–15) of six RF failures, the TF
replay's chunk-end hand target lies left of the RF replay's in 90 (by 8.0 cm on average over replans 2–8). On the 20
replans of four TF episodes where the hand holds the bottle and both wordings keep the gripper closed, the TF replay
carries it further right in 19 (by 8.9 cm on average). RF misreads both which object to take and which way to carry it.

### Building the edit (offline, practice decisions only)

Decisions, by a rule fixed before any edit was built ([`runs/s4r/select_items.py`](runs/s4r/select_items.py)):
6 *approach* decisions, one per RF-failure episode s20000–s20003, s20007, s20008 (the replan in 2–6 where the TF
and RF replays of the same input disagree most), and 6 *transport* decisions from TF episodes while the hand holds
the bottle (measured gripper 0.3–0.9, both wordings command closed; top 6 by the same disagreement, at most 2 per
episode). A third set, added after the first closed-loop run, holds 6 *grasp* decisions: the first replan whose chunk
closes the gripper in six other TF episodes (s20012, s20014, s20016–s20019), where the RF replay keeps the gripper open
in 5 of 6. Captures `s4r-ox` and `s4r-grasp-ox` replay these decisions under RF and TF with VLM residuals and MLP inputs
at the 2 × 16 image cells of layers 7–16. Every edit applies Eq. (1) of the paper to the MLP down-projections of VLM
layers 8–16 with alpha 1 and λ = 10⁻³·tr(KKᵀ)/n, rounded to bf16:

- `s4r-img8`: the paper's image-cell recipe (`weightsmith.distill`, unchanged): keys averaged over the 12 approach and
  transport decisions, 32 keys per layer, rank ≤ 32, 5.3M values as factors (0.16% of the parameters).
- `s4r-grp2`: the same recipe with keys and values averaged within each decision group (approach, transport) and the
  groups stacked in one solve: 64 keys per layer, rank ≤ 64, 10.6M values as factors (0.32%)
  ([`make_scripts.py`](make_scripts.py) `groups`).
- `s4r-grp3`: three groups (approach, grasp, transport): 96 keys per layer, rank ≤ 96, 15.9M values as factors (0.48%).
- `s4r-ind`: one key group per decision, 384 keys per layer, rank ≤ 384 (|ΔW|/|W| up to 0.30, against 0.05 for
  `s4r-img8`, 0.12 for `s4r-grp2` and 0.14 for `s4r-grp3`).

Offline replay with images, state and noise fixed ([`results/offline_checks_s4r.json`](results/offline_checks_s4r.json)).
Distance is the mean chunk-end end-effector distance to the unedited policy's TF replay of the same input.

| Decisions | n | unedited RF | `s4r-img8` RF | `s4r-grp2` RF | `s4r-grp3` RF | `s4r-ind` RF |
|---|---|---|---|---|---|---|
| build: approach | 6 | 15.1 cm | 2.9 cm | 1.6 cm | – | – |
| build: transport | 6 | 19.4 cm | 21.2 cm | 3.7 cm | – | – |
| build: approach + transport | 12 | 17.3 cm | 12.0 cm (7/12 closer) | 2.6 cm (12/12) | 2.7 cm (12/12) | 3.3 cm (12/12) |
| held-out approach (replans 2–6 of 6 other RF failures) | 30 | 10.7 cm | 3.9 cm (27/30 closer) | 2.7 cm (27/30) | 1.3 cm (27/30) | 2.4 cm (25/30) |
| held-out transport (other holding decisions) | 14 | 6.4 cm | 10.9 cm (2/14 closer) | 4.4 cm (8/14) | 3.9 cm (10/14) | 3.5 cm (9/14) |
| grasp replans that close the gripper (TF 6/6): set A (in `s4r-grp3`'s build) / set B (unused) | 6 / 6 | 1 / 2 | – | 3 / 5 | 6 / 6 | – |
| TF no-harm: TF replay moved by the edit (5 other TF episodes) | 40 | – | 2.4 cm, 0 gripper flips | 2.2 cm, 3 flips | 2.1 cm, 1 flip | 2.2 cm, 3 flips |

The standard recipe fixes the approach but moves transport the wrong way: averaging the keys over both phases gives
one value per image cell, and here that value follows the approach shift. Keeping the phases as separate key groups
fixes both offline; individual keys do about as well with a weight change more than twice as large. The `s4r-grp2`
TF flips close the gripper where the unedited TF replay opens it (twice while holding the bottle, once with an empty
hand).
`s4r-grp2` went to closed loop first; `s4r-grp3` was built after its practice run showed many episodes that reached
the bottle without grasping it. Both designs used practice decisions only. `s4r-img8` and `s4r-ind` were not run in
closed loop.

### Closed loop on practice slots 20000–20031

| Comparison | Edited | Unedited | Fixed / broke | exact McNemar p |
|---|---|---|---|---|
| RF + `s4r-grp2` vs RF | **8/32** | 0/32 | 8 / 0 | 0.0078 |
| TF + `s4r-grp2` vs TF (no harm) | 23/32 | 27/32 | 3 / 7 | 0.34 |
| RF + `s4r-grp3` vs RF | **16/32** | 0/32 | 16 / 0 | 3.1 × 10⁻⁵ |
| TF + `s4r-grp3` vs TF (no harm) | 23/32 | 27/32 | 5 / 9 | 0.42 |

([`results/practice_rf_edit_vs_base.json`](results/practice_rf_edit_vs_base.json),
[`results/practice_tf_edit_vs_base.json`](results/practice_tf_edit_vs_base.json),
[`results/practice_rf_grp3_vs_base.json`](results/practice_rf_grp3_vs_base.json),
[`results/practice_tf_grp3_vs_base.json`](results/practice_tf_grp3_vs_base.json),
[`results/outcomes_practice_s4r.json`](results/outcomes_practice_s4r.json).) Against TF's 27/32 on the same slots,
`s4r-grp2` recovers 8 and `s4r-grp3` 16 of the 27 episodes the wording loses. Under TF, each edit costs a net 4
episodes (−12.5 points; 95% bootstrap −31.2 to +6.2 for `s4r-grp2`, −34.4 to +9.4 for `s4r-grp3`), within the
run-to-run noise of 32 pairs (the paper measured about 30% discordant pairs for near-neutral edits; here 10 and 14 of
32). `s4r-grp2` makes the RF-worded policy grasp the bottle in 24/32 episodes (unedited RF 7/32, TF 32/32) and carry it to the right of the box in 13/32 (unedited RF 0/32):
8 stable placements, plus 5 that end right of the box without one (2 tipped over, 3 upright that never met the
stable-placement condition within 40 s); one lifted and tipped the bottle on the wrong side, and 18 never lifted it
2 cm. From the recorded object poses: the bottle starts 8.3 cm left of the box, and the edited successes leave it
15–34 cm to its right. The hand now goes left (y = +18.8 cm at replan 4, against +9.7 unedited and +21.8 under TF)
but stays higher than under TF (z = 19.6 against 15.1 cm) and closes at replan 7 (median; TF 4, unedited RF 13),
which is what the grasp group of `s4r-grp3` targets. With `s4r-grp3` the approach and grasp match TF: all 32 episodes
grasp the bottle, the hand is at y = +21.2, z = 15.3 cm at replan 4 (TF +21.8, 15.1) and closes at replan 4.5
(median; TF 4). What remains is the carry and placement: 16 stable placements, 8 that end right of the box without
one (4 tipped over), 7 grasps that never lift the bottle 2 cm, and 1 on the wrong side.

### Held-out slots 92000–92063 (never used for any choice)

`weightsmith/heldout.py` froze `s4r-grp2` (fingerprint `6af4d73a9fa52bf3`) and ran it against the unedited policy on
64 unseen slots, base and edit alternating order by half, both under the task's RF wording
([`results/heldout_grp2_92000-64.json`](results/heldout_grp2_92000-64.json)):

| Arm | Success | Wilson 95% | vs unedited: fixed / broke | difference (95% bootstrap) | exact McNemar p |
|---|---|---|---|---|---|
| unedited, RF | 0/64 | 0–5.7% | – | – | – |
| `s4r-grp2`, RF | **19/64** | 19.9–41.8% | 19 / 0 | +29.7 points (18.8 to 40.6) | 3.8 × 10⁻⁶ |
| `s4r-grp3`, RF | **27/64** | 30.9–54.4% | 27 / 0 | +42.2 points (29.7 to 54.7) | 1.5 × 10⁻⁸ |
| random-direction control of `s4r-grp3`, RF | 0/64 | 0–5.7% | 0 / 0 | 0 | 1 |

`s4r-grp3` (fingerprint `9e6f2378e28be6a3`) and its control `ctl-rand3` (`fbc8349de4182099`) were frozen and run
afterwards on the same slots by [`ctl_run_sgdw.py`](ctl_run_sgdw.py), in ABBA order over two 32-slot chunks, and
paired with the unedited arm above, as the paper did for its control
([`results/heldout_ctl_92000-64.json`](results/heldout_ctl_92000-64.json),
[`results/heldout_pairs_92000-64.json`](results/heldout_pairs_92000-64.json)). The control has the edit's layers
(8–16), rank (96 per layer) and singular values, so the same Frobenius norm, with random orthonormal directions
(numpy `default_rng(0)`); it reproduces `s4r-grp3`'s stored tensors exactly before the directions are replaced
([`make_scripts.py`](make_scripts.py) `ctl`). `s4r-grp3` against `s4r-grp2`: 27 vs 19 (21 / 13, p = 0.23). The
TF arm on held-out slots was dropped to stay near the GPU budget, so the held-out TF rate is not measured.

Where the bottle ended ([`results/outcomes_heldout_s4r.json`](results/outcomes_heldout_s4r.json)): unedited RF 63/64
never lift it and 11/64 push the raisin box more than 2 cm (they go for the wrong object); the random control
behaves the same (64 no lift, 9 box). `s4r-grp2`: 19 placed, 10 right of the box without a stable placement (4
tipped over), 2 carried to the wrong side, 33 never lifted. `s4r-grp3`: 27 placed, 16 right of the box without a
stable placement (12 tipped over), 8 carried to the wrong side, 13 never lifted. The edits fix taking the bottle more
than carrying and setting it down. `s4r-grp2`'s successful held-out episodes took 295 steps on average (of 600).

## Caveats

- **One goal, one model, one start.** The edit was built and tested on S4-R only, for π0.5 only. Every slot starts
  from the stock scene's default layout; the study varied the layout over eight starts. Slots differ by policy noise
  and simulator nondeterminism, which the paper shows is large (about 30% discordant pairs for near-neutral edits).
- **Success criterion.** Our task module re-implements the study's stable placement inside RoboLab (relation in a 45°
  cone in the robot frame, mustard touching the table, not touching the gripper, < 2 cm/s for 1 s). It credits the
  relation regardless of which object moved. The raisin box moved more than 2 cm in 1/32 practice and 11/64
  held-out episodes of the unedited RF policy (it reached for the wrong object), in 9/64 of the random control, once
  with `s4r-grp3` (a failed episode), and never under TF or `s4r-grp2`, so no success here comes from moving the box.
- **Edit design used practice data adaptively.** `s4r-grp2` (two key groups) was built after seeing the paper's recipe
  (`s4r-img8`, one group) fail offline on transport decisions, and `s4r-grp3` after seeing `s4r-grp2`'s practice
  episodes reach the bottle without grasping it. The approach and transport selection rule was fixed before any edit
  was built; the grasp set was chosen afterwards by a fixed rule (first closing replan). Held-out slots (92000+) were
  never used for any choice, and both edits were frozen before their held-out runs.
- **Partial fix.** The best edit places the bottle in 27/64 held-out episodes; TF on practice slots placed it in 27/32.
  Most remaining failures tip the bottle over or carry it the wrong way after a correct grasp.
- **Held-out pairing across runs.** `s4r-grp2` and its unedited arm ran in one `heldout.py` run with alternating
  order (18:03–19:18); `s4r-grp3` and its control ran in a separate run on the same slots (19:26–20:30), paired with that
  unedited arm (the unedited policy scored 0/64, 0/32 on practice and never above 0 under RF).
- **TF no-harm** is measured on 32 pairs only; a decrement of the observed size (−4 episodes) cannot be excluded.
- **Runtime.** A separate clone (`~/sgdw/weightsmith`, this branch) ran everything, with its own policy-server port
  and GPU 0 as the primary lane; GPU locks were shared with the other agents on the host, so wall-clock times include
  queueing.

## GPU time

3.2 GPU-hours (194 GPU-minutes; [`results/gpu_ledger.json`](results/gpu_ledger.json), [`gpu_ledger.py`](gpu_ledger.py)),
slightly over the 3-hour budget: 84 minutes of practice evaluations (8 runs of 32 episodes), 96 minutes of held-out
and control runs (8 halves of 32), and 14 activation captures counted at 1 minute each. A first S4-R launch, stopped
after about 3 minutes to move this runtime off the other agent's policy-server port, is not in the ledger. Most runs
used one GPU because other agents held the second; wall-clock time, 14:51 to 20:31, was mostly spent waiting in the
shared GPU lock queue.

## Files

| Path | What |
|---|---|
| `../../../weightsmith/run_batched.py` | `WSP_TASK_DIRS`: register task modules kept outside the RoboLab checkout (the only runtime change) |
| `tasks/` | S4-R and S5-LR as RoboLab tasks with the study's stable-placement success term, RF wording by default |
| `paired.py`, `ho_compare.py` | paired statistics (exact McNemar, Wilson, bootstrap) on practice and held-out records |
| `make_specs.py`, `runs/s4r/select_items.py`, `runs/s4r/specs/` | capture specs and the fixed decision-selection rule |
| `make_scripts.py`, `runs/s4r/build_*.json` | `owsp py` build scripts rendered with `weightsmith.distill` (`build`, `groups`, and the random-direction `ctl`) |
| `sb_pairs.py`, `sb_diverge.py`, `sb_check.py`, `runs/s4r/sb_*.py` | sandbox analyses: paired slots, TF-vs-RF replay disagreement, offline checks |
| `outcomes.py` | object outcomes (where the bottle ended, tipping, box disturbance) from RoboLab's state recordings |
| `ctl_run_sgdw.py` | `study/results/opus/controls/ctl_run.py` importing this checkout instead of `~/weightsmith` |
| `gpu_ledger.py` | GPU-minutes per pool from the runtime's logs |
| `results/` | every number in this README as JSON, plus the practice episode records (`episodes_*.jsonl`); `runs/s4r/*.txt` are the raw sandbox outputs (`*.report.txt`: per-layer build and control reports with tensor digests) |

Host state (ali-ms): runtime clone `~/sgdw/weightsmith` (this branch plus a `wsp.env`), run scripts and logs in
`~/sgdw/runs` and `~/sgdw/logs`, projects `~/vibes/opus-weight-surgery/projects/sgdw-s4r` (candidates `s4r-img8`,
`s4r-grp2`, `s4r-grp3`, `s4r-ind`, `ctl-rand3`, and `ctl-rand`, the control of `s4r-grp2`, built but never run;
held-out runs under `heldout/`) and `sgdw-s5lr`.

## Phase C: the same edit on Cosmos3 Nano (feasibility plan)

Facts below come from the cosmos-framework copy that served Nano on ali-ms (`~/wam-nano-stock-20260926/cosmos-framework`,
commit 411d25b; module paths below are relative to its `cosmos_framework/`), the Sept 26 Nano server records in
`~/wam-nano-stock-20260926/`, and the study's `MODEL_CONFIGS.json`. Nothing was run on a GPU for this section.

**Where the instruction enters.** Nano's 15.2B-parameter transformer (`OmniMoTModel` → `Cosmos3VFMNetwork` →
`Qwen3VLTextForCausalLM`, `model/vfm/mot/`) is a Qwen3-VL-8B-shaped mixture of transformers:
36 layers, width 4096, 32 query and 8 key/value heads of 128, MLP width 12,288. Each layer has two weight sets: a
text pathway (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`) and a generation pathway for video and action
tokens (the same names with `_moe_gen`). In `two_way_attention` (`mot/attention.py`) text tokens attend causally to
text only, while video and action tokens attend to all text keys/values plus their own, in every layer. So the
instruction reaches the actions only through each layer's text keys and values, a cross-attention in all 36
layers. There is no AdaLN; the timestep embedding is added to the noisy tokens. The prompt is the instruction plus a
fixed 74-token metadata suffix in the Qwen chat template, recomputed in every forward pass; classifier-free guidance
uses an empty caption through the same text weights. Each decision costs 4 UniPC steps × 2 guidance branches = 8
forward passes (33 action tokens denoised jointly with 9 latent video frames; `llm2action` reads the actions out).

**Matched positions come for free.** With the Qwen tokenizer (byte-identical copy in
`~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B`), the TF and RF S4-R instructions are both 22
tokens: positions 0–9 ("Move the mustard bottle on the table so that the") and 21 (".") are identical and 10–20
differ, so the 104-token prompts align one to one. The video and action tokens, the analogue of π0.5's image cells,
sit at the same positions under both wordings.

**What to edit, in order of cost.**
1. *Generation-pathway MLPs at video/action tokens (the direct analogue of the π0.5 image-cell edit).* Keys: the
   input of `layers.L.mlp_moe_gen.down_proj` (12,288-d) under RF at pooled latent cells and the action tokens, per
   denoising step, averaged over recorded decisions; values: the TF − RF change of that layer's residual increment at
   the same positions; ΔW = Kᵀ(KKᵀ + λI)⁻¹V per layer, mid-to-late layers first. Add the unconditional branch's
   keys with zero values as preservation keys, since the edit would otherwise also move the empty-caption pass and
   change the guidance difference.
2. *Text-pathway values.* Because text states do not depend on the image, noise or timestep, keys
   (post-norm inputs of `self_attn.v_proj`, 4096-d) and values (TF − RF `v_proj` outputs at the aligned positions)
   come from one text-only forward pass per wording, with no robot data. Value edits are linear; key edits pass
   through per-head RMSNorm and rotary embeddings. Only layer 35 is local: editing an earlier layer's keys/values also
   changes later text states, which must be re-captured layer by layer (edit L, re-run, edit L+1), as in MEMIT.
   This is cheap but risks memorising one prompt pair; transfer to the other RF goals (S1-L, S3-TOP, S5-LR) is
   the test.

**Weights: load, edit, save.** The release is a diffusers folder of bf16 safetensors (transformer in 7 shards,
30.3 GB; VAE 1.4 GB; vision encoder 1.15 GB; 33 GB in all). `inference/model.py` (`from_pretrained_dcp`) loads
either this layout or torch DCP shards; `scripts/export_model.py` and `convert_model_to_dcp.py` convert between
them. An edit is a few tensors (`v_proj` is 1024 × 4096; a generation-pathway `down_proj` is 4096 × 12,288): read
them with `safetensors` on CPU, add ΔW in float64, round to bf16, write a copy of the folder that replaces only
the touched shard(s), and serve it with the existing `--checkpoint-path`. Editing in process is harder because
FSDP shards the parameters. The exact released tensor names (`to_v` / `add_v_proj` in diffusers naming, per the
loader's rename table) are not verified, because the index file is no longer on the host.

**Capturing activations.** No capture code exists for Cosmos (the only forward hook is a training norm monitor).
Plan: forward hooks on the chosen modules in the RoboLab policy server (`cosmos_framework/scripts/action_policy_server_robolab.py`
or the workstation server), replaying recorded requests (`observation.npz`, prompt, seed) with the RF and TF prompt
and the same noise; pool the 3.6k video tokens to coarse cells on device before saving. Recorded inputs on the host
cover the cube/bowl scene with D/S/I left/right prompts (`~/wam-nano-stock-20260926/attempt-006/server/rank-0/request-*`,
90 requests) and 19 mustard/raisin requests under the stock prompt (`spatial-suite-001`); the study's own S4-R
observations are on the cluster.

**Memory and compute on 2 × 24 GB.** The bf16 transformer alone is 30 GB, so one 3090 cannot hold it. On ali-ms the
study served Nano with `torchrun` on both GPUs and FSDP CPU offload (`--offload-fsdp`): about 2 minutes to load, and
requests arrived every 67 s (median spacing of the recorded requests: 90 in `attempt-006`, 119 in
`spatial-suite-001`; the server logs put inference near 60 s). Without offload each rank held 15.45 GiB and the
simulator's 7.66 GiB did not fit beside it. A 600-action S4 episode is 19 requests, about 21 minutes, so the
study's 8 TF + 8 RF episodes take about 5.6 hours of both GPUs per weight variant, and base plus edit about 11 hours
(22 GPU-hours). Text-pathway capture (about 7.6B parameters, 15 GB) fits on one GPU; generation-pathway capture needs
the two-GPU path. One 80 GB GPU (the study used a B200) holds the model without offload and should be several times
faster (not measured here).

**State of the host.** The Nano checkpoint folder (`~/wam-nano-stock-20260926/checkpoints/nano`), the cosmos Python
environments, the Edge cache and the git-tracked `cosmos_framework/checkpoint/*` files are gone from ali-ms, and
the official container needs CUDA 13, above the host driver (535). Restoring means downloading 33 GB and rebuilding
the environment that ran before (torch 2.10 + cu128, Python 3.13).

**Work estimate.** Restore checkpoint and environment, reproduce one recorded request bit for bit: 0.5–1 day.
Capture hooks and matched TF/RF capture on recorded decisions: 1 day. Build and offline replay check (do RF + edit
actions move toward TF's on held-out decisions): 0.5 day. Closed-loop S4-R base vs edit under RF and TF at the
study's size: about 11 hours of both GPUs here. Random-direction control and transfer goals: as much again. In total
3–4 working days on ali-ms, most of it waiting on the offloaded model; on the cluster about 2 days.

**A cheaper first world-action model.** Cosmos3 Edge Policy DROID (4B) ran on a single 3090 in the August study at
14,435 MiB and 5.5 s per warm request (`~/sgw-scene-design-20260923/steerable/docs/VLA_VS_WAM_STEERABILITY_STUDY.md`),
next to a simulator on the other GPU, about ten times faster per decision than Nano with offload here. Its S4-R
gap is TF 4/8 vs RF 0/8 in the study. At 19 requests per S4 episode, 96 episodes (base, edit and control under both
wordings, 16 each) would take roughly 4 GPU-hours on ali-ms (estimate); its checkpoint (9.2 GB) is also gone from the host.

**Prototype.** Not built: the checkpoint and environment are gone from the host, so even a one-decision capture needs
the 33 GB download and an environment rebuild first, which does not fit this session's budget.
