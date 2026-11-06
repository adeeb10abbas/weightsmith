# Closed-form instruction distillation (`weightsmith distill`, Experimental)

Turn a prompt fix into a weight fix. This is the method behind the paper's edit: cube then banana, held-out 27/128 → 106/128 ([paper](https://adeeb10abbas.github.io/weightsmith/paper.pdf), Section 2).

## When to use it

Use it when the policy fails under its own instruction, an explicit rewording makes the *same weights* succeed, and the instruction cannot be changed where the policy runs. The rewording is the known-good input whose internal effect gets copied into the weights. Distillation cannot help when no wording works: in the paper, edits aimed at perception, a missing skill, contact physics or the gripper threshold gave no gain.

## The update

At each edited VLM layer, the keys K are the original wording's mean MLP inputs at aligned positions (prompt and robot-state tokens, image cells, or both), averaged over a few recorded failure decisions, and the values V are the change the explicit wording makes to that layer's residual update at the same positions. The edit ΔW = Kᵀ(KKᵀ + λI)⁻¹V with λ = 10⁻³·tr(KKᵀ)/n is the minimum-norm update that maps every key to its value: one linear solve per layer, rank at most n, no gradients and no training data.

## Run it

```bash
weightsmith distill run examples/cube_then_banana/distill.json --dry-run   # print every command, spec and script; touch nothing
weightsmith distill run examples/cube_then_banana/distill.json             # select -> capture -> build -> check, via bin/owsp
weightsmith distill render examples/cube_then_banana/distill.json --script build   # just the `owsp py` build script
```

`run` drives `bin/owsp` against the project in the config, so the usual `WSP_HOST`, `WSP_REMOTE_REPO` and project budget apply:

| Step | What it does | Runs on |
|---|---|---|
| select | picks failure decisions from base practice episodes by rule (`early_release` or `grasp`), unless `items` are given | CPU sandbox (`owsp py base`) |
| capture | replays those decisions under the original and the explicit wording (`owsp acts`) | GPU, charged to the activation budget |
| build | creates the candidate and writes ΔW into it | CPU sandbox (`owsp py`) |
| check | replays the edited policy on the same decisions and on `check_items` (other failures, correct carries, phase-two decisions) and compares gripper decisions and end-effector targets offline | GPU capture, then CPU |

It never starts a closed-loop or held-out evaluation. It ends by printing the `owsp eval` command for practice seeds and the operator's `weightsmith/heldout.py` command.

[`examples/cube_then_banana/distill.json`](../examples/cube_then_banana/distill.json) is a worked config for the 0.16% edit (image cells, layers 8–16). Its fields are `project`, `name` and `candidate`, `instruction` and `explicit_instruction`, the decisions (`items`, or a `select` rule when `items` is absent), `check_items`, `edit` (`mode`: `image`, `state` or `both`; `layers` in 0–16; `alpha`; `lam_rel`), and the `eval` and `heldout` sizes to print. Its `items` name episodes of the reference project; delete them to select decisions from your own.

## Status

- **Proven: the build step.** Rebuilt from capture `c2-multi` in fresh candidates, both published edits come out bit for bit identical, with the same `owsp` fingerprints: `v2mi-f` → `63db0c52a51306c9` and `abl-imgl8` → `5bf8c4329c1e416a`. [`tests/test_distill_reproduce.py`](../tests/test_distill_reproduce.py) repeats this on the GPU host when `WSP_DISTILL_PROJECT` is set.
- **Only dry-run: select, capture and check.** They are covered by [`tests/test_distill.py`](../tests/test_distill.py) on synthetic captures run through the real sandbox prelude, but have not yet run end to end on a new task.
- The CPU tests run in CI.

## Python

```python
from weightsmith import distill
dW = distill.closed_form_edit(K, V, lam_rel=1e-3)                  # [d_in, d_out] float64, for y = h @ W
script = distill.render_sandbox_script(distill.BuildConfig(
    orig_capture="c2-multi", expl_capture="c2-multi", orig_items=list(range(6)), expl_items=list(range(12, 18)),
    mode="both", layers=list(range(17))))                         # the v2mi-f build, as an `owsp py` script
```

The original 22-line script, every failed attempt before it and the offline checks are in [study/results/opus/ctb/](../study/results/opus/ctb/README.md).
