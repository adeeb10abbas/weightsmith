# Closed-form instruction distillation (`weightsmith distill`, Experimental)

Turn a prompt fix into a weight fix. This is the method behind the paper's edit: cube then banana, held-out 27/128 → 106/128 ([paper](https://ali-adeeb.com/weightsmith/paper.pdf), Section 2).

## When to use it

Use it when the policy fails under its own instruction, an explicit rewording makes the *same weights* succeed, and the instruction cannot be changed where the policy runs. The rewording is the known-good input whose internal effect gets copied into the weights. Distillation cannot help when no wording works: in the paper, edits aimed at perception, a missing skill, contact physics or the gripper threshold gave no gain.

## The update

At each edited VLM layer, the keys K are the original wording's mean MLP inputs at aligned positions (prompt and robot-state tokens, image cells, or both), averaged over a few recorded failure decisions, and the values V are the change the explicit wording makes to that layer's residual update at the same positions. The edit ΔW = Kᵀ(KKᵀ + λI)⁻¹V with λ = 10⁻³·tr(KKᵀ)/n is the minimum-norm update that maps every key to its value: one linear solve per layer, rank at most n, no gradients and no training data.

## Run it

[TUTORIAL.md](TUTORIAL.md) walks through a full run with real outputs. In short, on the GPU host, with base
practice episodes already in the project (`scripts/base_practice.sh 32`):

```bash
export WSP_ROOT=$PWD/projects/ctb                                            # your project
weightsmith distill run examples/cube_then_banana/distill.json --dry-run   # print every command, spec and script; touch nothing
weightsmith distill run examples/cube_then_banana/distill.json             # select -> capture -> build -> check, via bin/owsp
weightsmith distill render examples/cube_then_banana/distill.json --script build   # just the `owsp py` build script
```

`run` drives `bin/owsp` against the project in `WSP_ROOT` (or the config's `project`), so the usual `WSP_HOST`,
`WSP_REMOTE_REPO` and project budget apply, and it also works from a laptop:

| Step | What it does | Runs on |
|---|---|---|
| select | picks failure decisions from base practice episodes by rule (`early_release` or `grasp`), unless `items` are given; resolves rule-based `check_items` groups the same way, never reusing a build decision | CPU sandbox (`owsp py base`) |
| capture | replays those decisions under the original and the explicit wording (`owsp acts`) | GPU, charged to the activation budget |
| build | with `responsive_only`, keeps the decisions at which the two wordings disagree on the gripper; creates the candidate and writes ΔW into it | CPU sandbox (`owsp py`) |
| check | replays the edited policy on the same decisions and on the check items, then compares gripper decisions and end-effector targets offline | GPU capture, then CPU |

It never starts a closed-loop or held-out evaluation. It ends by printing the `owsp eval` command for practice
seeds and the operator's `weightsmith/heldout.py` command. Everything it wrote (specs, scripts, `items.json`,
`check_items.json`, `build.json`, `check.json`) is in `distill-runs/<name>/` or `--workdir`.

Reruns are safe. An existing capture is reused only when it holds exactly the decisions and wordings of this run,
and otherwise the run stops and says so. The candidate's check capture is named after its `owsp` fingerprint, so a
rebuilt edit is never checked against an older replay.

### The config

[`examples/cube_then_banana/distill.json`](../examples/cube_then_banana/distill.json) applies the recipe of the
paper's 0.16% edit (image cells, layers 8–16) to your own project.
[`distill-first-carry.json`](../examples/cube_then_banana/distill-first-carry.json) does the same from releases on
the first carry (`max_step`), which worked much better in the tutorial's project.

| Field | Meaning |
|---|---|
| `project` | project root on the GPU host; default `$WSP_ROOT` |
| `name`, `candidate` | prefix of the capture ids (`<name>-o`, `-x`, `-c-<fingerprint>`, `-cb`) and the candidate to create |
| `instruction`, `explicit_instruction` | the project's wording and the rewording that works |
| `items` or `select` | the decisions: explicit `{"episode", "k"}` items, or a rule (`early_release`: grasp, then let go within `max_hold` replans with the measured gripper in `[min_grip, max_grip]`; `grasp`: the first close of an open hand), plus `cand`, `success` (`false`: failed episodes, `null`: all), `first_only`, `max_step`, `limit`, `episodes`, `exclude` |
| `responsive_only` | build only from the decisions whose gripper decision changes with the wording; stop if none does |
| `check_items` | decisions NOT used to build the edit, grouped: `{group: [items]}` or `{group: rule}` |
| `edit` | `mode` (`image`, `state` or `both`), `layers` (0–16), `alpha`, `lam_rel` |
| `eval`, `heldout` | the sizes and slot start printed in the next steps |

[`reference-project.json`](../examples/cube_then_banana/reference-project.json) records the decisions and check
items behind the paper's edit. It only runs on the reference project, because the same episode names in another
project hold different decisions.

## Status

Experimental: it has run end to end on one task.

- **The build step rebuilds the paper's edits bit for bit.** Rebuilt from capture `c2-multi` in fresh candidates, both published edits come out identical, with the same `owsp` fingerprints: `v2mi-f` → `63db0c52a51306c9` and `abl-imgl8` → `5bf8c4329c1e416a`. [`tests/test_distill_reproduce.py`](../tests/test_distill_reproduce.py) repeats this on the GPU host when `WSP_DISTILL_PROJECT` is set.
- **`run` works end to end on a new project.** On 2026-11-07 it ran select → capture → build → check on a fresh cube-then-banana project with 32 base practice episodes ([TUTORIAL.md](TUTORIAL.md)). It selected 10 early releases, of which the rewording changes 8, and built `ctb-imgl8` (fingerprint `72e042a97074dddc`). Offline, the edit makes the explicit wording's gripper call at 10/10 of those decisions and keeps the gripper closed at 11 of 16 unseen early releases (the unedited replay: 1); 3 of 16 grasp decisions no longer close. In closed loop on the same 32 practice slots it scored 8/32 against 2/32 (8 fixed, 2 broken, exact p = 0.11): a screen that passes, not evidence, and well short of the paper's version of this recipe (25/32 against 3/32 on its practice slots).
- **The decisions matter.** Rebuilt from releases on the first carry ([`distill-first-carry.json`](../examples/cube_then_banana/distill-first-carry.json), `ctb-first`, fingerprint `3047901c3502ffea`), the same recipe kept the gripper closed at 13 of 16 unseen releases, changed 1 of 16 grasps, and scored 19/32 against 2/32 on the same slots (18 fixed, 1 broken, exact p = 7.6×10⁻⁵). Both runs are practice screens: neither edit has a held-out score yet.
- **Not yet shown:** other tasks, and selection rules beyond gripper decisions (`early_release`, `grasp`).
- The CPU tests run in CI: [`tests/test_distill.py`](../tests/test_distill.py) (the math and the rendered sandbox scripts) and [`tests/test_distill_pipeline.py`](../tests/test_distill_pipeline.py) (the whole `run` against a fake GPU host).

## Python

```python
from weightsmith import distill
dW = distill.closed_form_edit(K, V, lam_rel=1e-3)                  # [d_in, d_out] float64, for y = h @ W
script = distill.render_sandbox_script(distill.BuildConfig(
    orig_capture="c2-multi", expl_capture="c2-multi", orig_items=list(range(6)), expl_items=list(range(12, 18)),
    mode="both", layers=list(range(17))))                         # the v2mi-f build, as an `owsp py` script
```

The original 22-line script, every failed attempt before it and the offline checks are in [study/results/opus/ctb/](../study/results/opus/ctb/README.md).
