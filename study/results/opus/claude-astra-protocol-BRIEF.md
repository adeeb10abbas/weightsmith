# RQ1-RQ3: sparse repair of a manipulation policy (Claude editor, weightsmith runtime)

This is one independently initialized editor session in **adaptive** proposal mode. It mirrors the protocol given to the earlier Astra editor (same task, evidence, budgets, edit interface and gate), adapted to this runtime's `owsp` command. Work only through `owsp` plus the Read/Write/Edit file tools in this directory. Do not use the internet, subagents, operator files, or other runs. Do not change this brief or request additional resources.

## Task and fixed policy

Improve closed-loop success for `StackYellowOnRedTask`, instruction **Stack the yellow block on the red block**, using direct sparse edits to the pi0.5 DROID joint-position policy. The task has a fixed scene, 900 control steps at 15 Hz, and 16 parallel environments. An action chunk has 15 time steps and 8 used channels: 7 joint targets and 1 gripper channel. The gripper threshold is fixed at 0.5. Do not edit prompts, inference code, wrappers, thresholds, task predicates, or the simulator. Partial task scores do not count as success.

The action expert is an 18-layer transformer. **The only tensors you may edit** are `action.blocks.LL.mlp_1.linear`, where LL is 00 through 17, shape [4096, 1024]. Row r is the 1024-dimensional residual direction written by hidden unit r. Action-expert hidden activations have 4096 features; residual activations have 1024. Flow matching uses 10 denoising steps. Activations can be captured at selected layers, denoising steps, and action-token positions.

**Cumulative support per candidate:** at most 128 changed rows across at most 2 layers, measured against the original checkpoint after bfloat16 rounding. A row with the same index in two layers counts twice. Check with `owsp diff`.

## Evidence, budget, and outcome

You receive exactly 32 original-policy development episodes on slots 20032–20063 (`owsp episodes --cand base -v`). They use the same scene and different policy-noise streams. Read their events (`owsp show`), actual camera inputs (`owsp frames`) and action summaries before proposing a mechanism. These data and the declared tools are your only empirical evidence.

You have at most:
- **Three candidate evaluations**, each on these same 32 slots: `owsp eval pN --n 32 --offset 32`. Exactly once per candidate. 96 candidate episodes total.
- **96 activation items** in total, counting repeated and failed captures.
- **7200 elapsed seconds** of your working time, including tool execution and queue waits. This is a cap, not a requirement.

A fixed operator gate compares each candidate with the original 32-episode base: a candidate is eligible only if it has **strictly more successes and at most 2 newly broken base-success pairs** (paired by slot, as printed by `owsp eval`). Eligible candidates rank by more successes, then smaller actual post-bfloat16 delta L2, then earlier proposal. A proposal with zero changed entries is identity/no-edit and cannot qualify. Base/no-edit is a valid final answer. The operator later scores frozen candidates on unseen slots; those outcomes must never reach you.

## Candidate lifecycle (adaptive mode)

Create candidates in order `p1`, `p2`, `p3` (`owsp new pN --note "<initial hypothesis>"`). Build the edit, inspect `owsp diff`, and write the final mechanism and prediction in `NOTEBOOK.md` **before** evaluating. After that, do not change the candidate. Evaluate p1, inspect its outcomes and the gate, and only then create p2, starting from base or from the currently accepted incumbent (`--from`). Repeat for p3.

Keep a concise notebook with episode/capture IDs, failure descriptions, why the selected units/direction might cause the behavior, the expected repair and possible breakages, all operations and factors, and whether results supported the prediction. Correlation between an activation and failure is not proof that editing it will repair the policy. Null findings and rejected candidates matter.

Finish by writing `FINAL.md` with the gate-selected candidate (or base), the exact edit, and a short mechanism account.

## Tools

| command | effect |
|---|---|
| `owsp status` | Budget used, candidates, captures. |
| `owsp tensors [REGEX]` | Names, shapes, statistics. |
| `owsp episodes [--cand NAME] [-v]` | Recorded practice rows. |
| `owsp show EID` | Event timeline and terminal result. |
| `owsp frames EID --steps a,b,c` | Renders actual camera frames to a local JPG (open with Read). |
| `owsp new NAME [--from OTHER] [--note TEXT]` | Create a candidate. |
| `owsp diff NAME` | Validated actual changes. |
| `owsp eval NAME --n 32 --offset 32` | The 32 development episodes for a candidate. |
| `owsp acts spec.json` | Activation capture (spec format below). |
| `owsp py NAME script.py` | CPU numpy sandbox for the **allowed operations only** (below). |

Bash runs only bare `owsp ...` commands. Write scripts and specs with the Write tool into `scripts/` and `specs/`.

**Activation spec:**
```json
{"id":"inspect-grasp","cand":"base",
 "items":[{"episode":"base.base.s20032","k":8},{"episode":"base.base.s20033","step":120}],
 "action_layers":[0,1],"denoise_steps":[0],"action_tokens":"mean"}
```
`k` is a request index; `step` selects the nearest recorded request at or before that step. Outputs in `acts(aid)[1]`: `action.mlp_hidden` [N,S,La,T,4096], `action.resid_out` [N,S,La,T,1024], `actions` [N,15,8]. Prompt overrides and image/attention extensions are not part of this protocol.

**Sandbox helpers** (`owsp py`): `load(name)`, `base(name)`, `save(name, bf16(array))`, `episodes(cand)`, `obs(eid)` (exact inputs and action chunks per request), `acts(aid)`; `/work/notes` persists JSON.

**Allowed operations — use only these, as explicit NumPy:**
- Descriptive statistics and contrasts: per-feature success-minus-failure standardized mean differences of captured activations (episode-averaged); a unit success-minus-failure residual direction at a captured layer; row statistics, norms, and projections of rows onto such a direction.
- `row_scale`: multiply selected rows by a factor (0 silences, -1 flips).
- `row_add_direction`: add `factor * unit_direction` to selected rows.
- `row_project_out`: `w - factor * dot(w,u) * u` for selected rows and a same-layer unit direction u.

No gradient optimizer, fitted regression, random or evolutionary search, model training, or sweeps over arbitrary trials. Selection of rows and factors must follow your stated hypothesis.
