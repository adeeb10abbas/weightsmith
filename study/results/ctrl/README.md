# Controlled study: seven blind editors, one task per failure kind (2026-11-06)

**Question.** With identical tools, budget and protocol, can a blind LLM editor find a weight edit that improves a zero-shot π0.5 policy, and does it survive held-out scoring? One RoboLab task per failure kind.

**Result.** One of seven submissions was accepted: on the two-step instruction task (cube then banana), a 1,024-value edit took held-out success from **11/64 to 27/64** (22 fixed, 6 broken; exact McNemar p = 0.0037, Holm-adjusted over the seven tasks p = 0.026; paired difference +25.0 pp [+9.4, +39.1]). Two more submitted edits stayed within noise on held-out slots. Four editors submitted the unedited policy.

| Task | Failure kind | Submission | Edit | Dev, slots 70000–70031: base → cand (fixed / broke) | Held-out, slots 91008–91071: base → edit (fixed / broke), p |
|---|---|---|---|---|---|
| Cube then banana | Two-step instruction | `p1` | rank-1 gated change to the gripper column of `action.action_out_proj.kernel`; **1,024 values** | 4/32 → 10/32 (8 / 2) | **11/64 → 27/64 (22 / 6), p = 0.0037: accepted** |
| Pick up eyeglasses | Thin-object pickup | `p1` | one repurposed MLP unit in `action.blocks.17` (gate, up, down); 3,072 values | 12/32 → 18/32 (9 / 3) | 18/64 → 27/64 (18 / 9), p = 0.12 |
| Mustard into left bin | Grasp and carry | `p1` | two rows of `language.blocks.00.mlp.linear` that strengthen the token "left"; 4,094 values | 6/32 → 9/32 (7 / 4) | 8/64 → 11/64 (9 / 6), p = 0.61 |
| Cube left of bowl | Spatial relation | `base` | (p1: rank-1 on 64 rows of `language.blocks.08.mlp.linear`, 130,764 values, not submitted) | 5/32 → 3/32 (3 / 5) | not scored |
| Stack three cubes | Assembly (stacking) | `base` | (a gated gripper edit failed a split-half check offline; no candidate ran) | 16/32 | not scored |
| Stack yellow on red | Lookalike perception | `base` | (a steering edit had no offline effect; not run) | 13/32 | not scored |
| Pick up flat box | Flat-object pickup | `base` | (p1: rank-1 gripper readout change, 1,021 values, not submitted) | 0/32 → 0/32 (0 / 0) | not scored |

Held-out runs alternate order by half (first half base first, second half edit first). Cube then banana improves in both halves (7 → 10 and 4 → 17). Wilson 95% intervals for cube then banana: 9.9–28.2% unedited, 30.9–54.4% edited. No slot crashed. Edit sizes are from `owsp diff` on the frozen candidates.

## Design

- **Editors.** Seven fresh Claude Opus sessions (Claude Code, extra-high effort), one per task. Each received the same brief apart from the task line (`<task>/BRIEF.md`) and the same general hints from earlier editors on other tasks (`PRIOR_FINDINGS.md`).
- **Blind.** Bash only for `./owsp ...`; file tools only inside the editor's own directory. No access to other projects, other editors' notes, the repository, the paper or the web.
- **Protocol.** Run the unedited policy on 32 development slots. Diagnose with activation captures on recorded decisions; instruction overrides are allowed for diagnosis only. Build at most two candidates in closed form (no gradients, training, fitted predictors or search), each evaluated once on the development slots after a written prediction. Submit one candidate or `base` with `FINAL.md`.
- **Budget.** 96 practice episodes and 192 activation items, enforced by the tool.
- **Scoring.** The operator scored each submitted candidate with `weightsmith/heldout.py` (frozen copy, fingerprint check, paired slots, order alternating by half, exact McNemar, Wilson intervals) on 64 held-out slots that no editor saw. Acceptance: exact McNemar p < 0.05 on at least 64 held-out pairs (BENCHMARK.md).

## Deviations from the plan

1. **Budget cut.** About an hour in, before any editor had evaluated a candidate, the budget was cut from three candidates to two. The cut was one identical message to all seven and was enforced in each project's `config.json` (96 episodes).
2. **Interruption.** All seven sessions were interrupted once by the operator and resumed in fresh sessions from their own `NOTEBOOK.md`. Three base evaluations queued at that moment (stack three cubes, yellow on red, eyeglasses) were lost before completing. They were rerun and never charged.
3. **Wait command.** A blocking `./owsp wait` command (`operator/wait_proj.sh`) was added so that editors could wait on the shared GPU queue without ending their session.
4. **Rule break.** The eyeglasses editor ran one `ls` of its own directory outside the allowed commands, and said so in its report.
5. **One candidate only.** GPU queue waits of 25–30 minutes per job led four editors (cube then banana, mustard, eyeglasses, cube left of bowl) to stop after one candidate.
6. **Failed first launch.** The first held-out launch at slot 91000 stopped before any episode ran, because slots must form whole groups of 16. Scoring used slots 91008–91071; the log of the failed launch is not part of any result.

## Round 2: stacking, extended time

After the study, the stacking editor was given more time with the same budget (two candidates) to build the gate it had proposed. Its `p1` (3,072 values: one repurposed unit in `action.blocks.17` that gates the gripper open at the stack) removed the targeted failure, with closes on empty air at the stack falling from 73 to 5 replans, but success did not change: development 17/32 vs 16/32 (6 fixed / 5 broken); held-out slots 91008–91071, 39/64 vs 35/64 (17 fixed / 13 broken, exact p = 0.585). The remaining bottleneck is placing the third cube, a motor-precision problem the edit does not touch. Records: `stack3/FINAL2.md`, `stack3/NOTEBOOK.md`, `stack3/scripts_round2/`, `heldout/ctrl-stack3__*.json`.

## Findings worth noting

- **Same diagnosis, smaller edit.** The cube-then-banana editor independently found the misreading behind the paper's main result. Replaying 16 premature releases under "Put the cube in the bowl" turned all 16 into holds. It then wrote a different edit, 1,024 values gated to premature-release states. That edit recovers part of the effect (held-out 42% against 17%); the paper's operator-built distilled edit reaches 83% on this task.
- **The same two units, found twice.** Two editors working on different tasks (cube left of bowl, mustard) independently found the same two layer-0 MLP units, 3097 and 8023, as detectors of the token "left".
- **No edit without a known-good input.** On lookalike perception, rewording the instruction did not move the arm, so there was nothing to distil, and no edit was submitted. On flat-box pickup, an edit changed the decision it targeted (grasps stayed closed, and the box was lifted 10–19 cm in 6 episodes) without producing a success.

## Files

- `<task>/`: `BRIEF.md`, `FINAL.md`, `NOTEBOOK.md`, `scripts/`, `specs/` for each editor; `PRIOR_FINDINGS.md` is shared by all seven.
- `heldout/`: the operator's held-out results (`results.json` format of `heldout.py`, with per-slot outcomes) and plans.
- `summary.json`: development results paired by slot, candidate notes, `owsp diff` sizes, budgets and held-out summaries, from `operator/ctrl_collect.py`.
- `operator/`: the scoring launcher, the collector, the held-out tally and the wait script.
