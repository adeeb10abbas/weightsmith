# Goal and layout inputs

This module defines three historically informed pi0.5 repair goals: ordered
cube/banana placement, cube behind a bowl, and mustard right of a raisin box.
Each goal keeps an original request, a fixed clarification, an unfamiliar
wording, and a competing request. All source collection and qualification are
fresh. Historical success does not automatically qualify a goal.

`get_goal(goal_id)` returns prompts, selected and competing task/goal IDs, the
existing runtime directory/case, horizon, and source-state rule. Runtime task
classes, goal scoring, fixed16 serving, and layout reset checks are reused from
the validated earlier studies without changing their files. A worker must use
the selected runtime directory consistently for task registration, simulation
entry point, `layouts.py`, and `scoring.py`; run different runtime families in
separate processes to avoid the legacy modules' bare `tasks`/`layouts` imports.

`build_layouts(goal_id, split, n=None, seed=None)` returns JSON suitable for that
runtime. Stages are `collection` (48 layouts), `development` (16), and `heldout`
(16). The first two are marked development data. Seeds and IDs are distinct.
Held-out layouts swap relevant object starts in half the layouts and change
reference placement or common scene offset. They change arrangements of known
objects, not object identities or scenes. `write_layouts` refuses replacement
with different contents. Before any learned-policy execution, the caller must
run `preflight.py` for every used layout set and verify its completed summary.
Generation and these CPU tests do not establish physical reset validity.

The temporal selector retains the fixed first-early-release heuristic. Each
spatial selector uses the middle recorded decision of the failed original
episode: `floor((decision_count - 1) / 2)`. The corresponding successful
clarified episode qualifies the source pair; no successful-trajectory state or
action target enters this selector. No per-source search or rollout feedback
chooses the fitting index. The root study protocol defines collection caps,
eligibility, triplet assignment, and test counts before launch.

Policy-free preflight, inside the normal Isaac/RoboLab environment:

```sh
python goals/preflight.py --goal cube_behind --layouts /absolute/layouts.json \
  --output /absolute/fresh-preflight-directory --device cuda:0 --headless
```

The launcher owns the GPU lock, working directory, environment, and output log.
The script tests every layout twice in complete groups of sixteen, logs the
existing runtime's reset evidence, forbids policy imports and action steps, and
stops on invalid initial goals or unstable placements. An incomplete preflight
must remain intact; a separately declared new attempt uses a fresh directory.

CPU checks (these are software tests, not robot-policy results):

```sh
python3 -m unittest discover \
  -s study/iros2027/v2/rss_few_pair_validation_20261120/goals -p 'test_*.py' -v
```
