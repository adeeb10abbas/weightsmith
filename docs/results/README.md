# Results so far

All numbers come from the run logs and editor files of the reference setup (packaged on
2026-11-04). Policy: OpenPI `pi05_droid_jointpos`, unedited unless stated. RoboLab, Isaac Sim
`performance` render preset, 900-step limit, success = RoboLab's boolean terminal flag.

## Base policy: practice sets (operator pre-runs, editor-visible)

| task | instruction | slots | base success |
|---|---|---|---|
| `BananaThenRubiksCubeTask` | "put the banana then the cube in the bowl" | s20000..s20127 | **99/128 (77.3%)** |
| `RubiksCubeThenBananaTask` | "Put the cube then the banana in the bowl" | s20000..s20063 | **9/64 (14.1%)** |
| `StackYellowOnRedTask` | "Stack the yellow block on the red block" | s20000..s20063 | running at packaging time |

## Base policy: task surveys (operator only, 16 episodes each, slots 60000..60015)

Used to pick tasks with room to improve. Top failure reasons are RoboLab's terminal messages.

| task | success | main failure reasons (count) |
|---|---|---|
| `RubiksCubeThenBananaTask` | 2/16 | cube never grabbed in stage 1 (7); banana never grabbed in stage 1 (6); cube not in bowl (1) |
| `RubiksCubeLeftOfBowlTask` | 1/16 | cube not left of bowl (15) |
| `RubiksCubeRightOfBowlTask` | 0/16 | cube not right of bowl (16) |
| `StackYellowOnRedTask` | 6/16 | not stacked (7); yellow block never grabbed and not stacked (3) |
| `MarkerInMugTask` | 1/16 | marker never grabbed (15) |
| `Stack3RubiksCubeTask` | 11/16 | not stacked (5) |
| `BlockStackingOrderAgnosticTask` | 11/16 | not stacked (3); no object completed (2) |

## Editor runs

| run | editor | task | status | submitted | held-out |
|---|---|---|---|---|---|
| 1 | Claude Opus (Claude Code, effort xhigh) | banana-then-cube | finished | `base` (no edit) | not run (nothing to test) |
| 2 | Claude Opus (Claude Code, effort xhigh) | cube-then-banana | stopped by operator before any candidate | none | not run |
| 3 | Claude Opus (Claude Code, effort xhigh) | stack-yellow-red (capability, fixed instruction) | in progress (base practice running; editor queued to start after it) | none yet | not run |
| - | GPT-6 Astra (Codex CLI) | - | not run yet | - | - |

### Run 1: banana-then-cube, honest negative ([FINAL.md](claude-opus-run1-banana-then-cube/FINAL.md), [NOTEBOOK.md](claude-opus-run1-banana-then-cube/NOTEBOOK.md))
- **Mechanism found: an elliptical-instruction mis-parse.** After grasping the cube, the policy
  sometimes opens the gripper on the next replan, 15-25 cm from the bowl. This happens on about
  29% of first cube grasps.
  - Replaying the recorded inputs reproduces it in 28 of 29 cases.
  - The explicit prompt "put the banana in the bowl then put the cube in the bowl" cuts it to 9/29.
  - In stage 2 the policy still partly treats the banana as the target, so it drops the held cube as the "wrong object".
  - The release decision is already formed in VLM layers 14-16, and action-expert attention in layers 14 and 16 writes it.
- **No edit submitted.** Three closed-form candidates were built and each was rejected offline, on replayed decision points:
  - a rank-1 gripper bias: not selective;
  - a single "then"-detector unit injection: no effect;
  - silencing 81 correlated VLM units: correlates, not causes.
  - Head ablations showed the release push is spread across heads.
  - The mechanism's estimated ceiling was about +3 pp, below what practice evals can resolve. So the editor submitted `base`.
  - It used 0 of 1600 practice episodes and about 1640 activation items.

### Run 2: cube-then-banana, stopped ([NOTEBOOK.md](claude-opus-run2-cube-then-banana/NOTEBOOK.md))
- The same mis-parse, stronger on the mirrored instruction.
  - 21/21 first-phase premature releases replay as releases under the original prompt, and 2/21 under the explicit two-clause prompt.
  - The notebook's rule: "the hold/release target is the noun bound to 'in the bowl'".
- Diagnostic closed-loop **prompt override (not a weight edit)**: 22/32 with the explicit prompt vs 3/32 for base on the same slots (fixed 20, broke 1).
  - This bounds what a language-side weight fix could achieve on this task.
- The run was stopped before any weight candidate was built.

### Run 3: stack yellow on red (capability), in progress
- At packaging time the 64-episode base practice set was running; the editor starts after it.
- The instruction is fixed and prompt overrides are disabled (`"prompt_overrides": false` in
  [configs/stack-yellow-red.json](../../configs/stack-yellow-red.json)), so only weights can change.
- Brief: [briefs/stack-yellow-red/BRIEF.md](../../briefs/stack-yellow-red/BRIEF.md). No results yet.
