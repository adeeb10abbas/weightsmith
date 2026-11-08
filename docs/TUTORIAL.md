# Your first weight edit

In this tutorial you take π0.5 on one RoboLab task and find the one decision where it goes wrong. You then show
that a reworded instruction fixes that decision, write the rewording's effect into the weights with
`weightsmith distill` (closed form, no training), and measure the edited policy against the unedited one on the
same simulator seeds. In the run below, the first edit took practice success from 2/32 to 8/32, and one
iteration guided by the offline check reached 19/32.

Every output below comes from real runs on 2026-11-07 on the reference box (2× RTX 3090), trimmed for length,
with the project path shortened to `projects/ctb`. Your numbers will differ: the simulator is not
bit-deterministic, which is the subject of [step 7](#7-read-a-paired-result).

**You need** a GPU host where `scripts/smoke_test.sh` passes ([SETUP.md](SETUP.md)), with this repo checked out
and its environment active (`source .venv/bin/activate`). Run every command from the repo root on the GPU host.
To drive the host from a laptop instead, see [SETUP.md](SETUP.md#drive-the-gpu-host-from-a-laptop); the
commands are the same.

| Step | What you do | GPU time here |
|---|---|---|
| [1](#1-make-a-project) | Make a project and look at the weights | none |
| [2](#2-run-the-unedited-policy) | Run the unedited policy on 32 practice seeds: the paired baseline | 11 min |
| [3](#3-find-the-decision-that-fails) | Find the decision that fails | none |
| [4](#4-replay-that-decision-under-two-wordings) | Replay that decision under two wordings | under 1 min |
| [5](#5-distil-the-rewording-into-the-weights) | Distil the rewording into the weights | about 2 min (4 captures) |
| [6](#6-run-the-edited-policy-on-the-same-seeds) | Run the edited policy on the same 32 seeds | 11–17 min |
| [7](#7-read-a-paired-result) | Read a paired result, then iterate: a second edit, checked and evaluated | about 18 min |
| [8](#8-the-held-out-score-operator-only) | The held-out score (operator only) | about 45 min; 8/64 → 36/64 |

GPU jobs queue behind each other through per-GPU locks, so on a shared host the wall time can be much longer.
In this run a 17-second capture waited 13 minutes behind other people's evals.

## The task

*"Put the cube then the banana in the bowl"* (RoboLab's `RubiksCubeThenBananaTask`). The released policy
usually fails it. It grasps the cube, lifts it, and lets go short of the bowl. The paper traced this to how it
reads the instruction: "in the bowl" binds to the nearest noun, so while carrying the cube the policy concludes
it holds the wrong object. Worded as *"Put the cube in the bowl then put the banana in the bowl"*, the same
weights succeed most of the time.

That is the situation closed-form instruction distillation is for: the skill is in the weights, the policy
misreads the instruction it is given, and some rewording that it reads correctly exists.

## 1. Make a project

A project is one folder. Its `config.json` names the task, the instruction, the practice seeds and the budget;
`owsp` adds the candidates, recorded episodes and activation captures as you go.

```bash
export WSP_ROOT=$PWD/projects/ctb
mkdir -p $WSP_ROOT && cp configs/cube-then-banana.json $WSP_ROOT/config.json
bin/owsp status
```

```json
{
 "budget_used": {"episodes": 0, "acts_calls": 0, "acts_items": 0, "evals": 0},
 "limits": {"practice_episodes": 1200, "acts_items": 6000, "acts_items_per_call": 96},
 "task": "RubiksCubeThenBananaTask",
 "instruction": "Put the cube then the banana in the bowl",
 "candidates": [],
 "acts": []
}
```

`bin/owsp` is the only command an editor gets, whether that editor is you or an LLM. Every sub-command works on
the project in `WSP_ROOT`. List the weights it can edit:

```bash
bin/owsp tensors 'language.blocks.08'
```

```
language.blocks.08.attn.attn_vec_einsum.w               [8, 256, 2048]         mean=+8.462e-06 std=0.03395 ...
language.blocks.08.attn.kv_einsum.w                     [2, 1, 2048, 256]      mean=-6.424e-06 std=0.03699 ...
language.blocks.08.attn.q_einsum.w                      [8, 2048, 256]         mean=-3.103e-05 std=0.03727 ...
language.blocks.08.mlp.gating_einsum                    [2, 2048, 16384]       mean=+1.949e-05 std=0.03584 ...
language.blocks.08.mlp.linear                           [16384, 2048]          mean=+7.116e-06 std=0.03643 ...
language.blocks.08.pre_attention_norm.scale             [2048]                 mean=+1.336 std=1.118 ...
language.blocks.08.pre_ffw_norm.scale                   [2048]                 mean=+1.594 std=1.231 ...
[7 tensors]
```

These are layer 8 of the vision-language model (VLM). The `language.blocks.LL.mlp.linear` matrices, the MLP
down-projections, are what this tutorial's edit changes. Without a pattern, `bin/owsp tensors` lists all 730
tensors: the VLM, the action expert and the action projections.

## 2. Run the unedited policy

```bash
scripts/base_practice.sh 32
```

```
[eval using extra GPU lane(s): [0]]
[eval base fp=base n=32 success=2/32 (6.2%) wall=11.2 min]
  base.base.s20000                   FAIL steps=900 Condition not satisfied: object_grabbed( | WRONG_GRABBED_FAILURE@103,...
  base.base.s20001                   FAIL steps=900 Condition not satisfied: object_grabbed( | GRABBED@119,TARGET_DROPPED@123,...
  ...
  base.base.s20011                   OK   steps=242 Completed subtask 'pick_and_place' 2/2 | GRABBED@52,TARGET_DROPPED@106,...
  ...
```

This ran 32 episodes of the unedited policy (`base`) in closed loop, 16 in parallel per GPU. Each episode is a
**slot**: `s20000` to `s20031`. A slot fixes the policy's noise seed and the scene, so any later candidate run on
the same slots is **paired** with these episodes, seed by seed. Episodes last up to 900 simulator steps, so even
with the second GPU as an extra lane (the first line) this takes 11 minutes; on one GPU expect about twice that.
`base_practice.sh` then resets the budget, so this baseline costs the editor nothing.

2 of 32 is low, even for this task: the paper's unedited policy scored between 3/32 and 27/128 on other slot
blocks. A 32-episode rate is that noisy; step 7 comes back to this.

## 3. Find the decision that fails

Every recorded episode keeps its events and the exact policy inputs at every replan. Look at a failure:

```bash
bin/owsp show base.base.s20001
```

```
 "reason": "Condition not satisfied: object_grabbed(object=rubiks_cube) (step 1/2)",
 "events": [
  {"step": 119, "name": "OBJECT_GRABBED_SUCCESS", "info": "success: object_grabbed(object=rubiks_cube). ..."},
  {"step": 123, "name": "TARGET_OBJECT_DROPPED",  "info": "Target object dropped during transport"},
  {"step": 185, "name": "OBJECT_GRABBED_SUCCESS", ...},
  {"step": 241, "name": "TARGET_OBJECT_DROPPED",  ...},
  {"step": 295, "name": "OBJECT_GRABBED_SUCCESS", ...},
  {"step": 316, "name": "TARGET_OBJECT_DROPPED",  ...},
  {"step": 350, "name": "WRONG_OBJECT_GRABBED_FAILURE", "info": "Wrong object grabbed: 'banana' (target objects: ['rubiks_cube'])"},
```

The robot picks up the cube three times and drops it three times on the way. Which decision drops it? The policy
replans every 15 simulator steps: it sees the two camera images and the robot state, then outputs a chunk of 15
actions, 7 joint targets plus a gripper command. `bin/owsp py` runs a NumPy script in a sandbox that can read the
recorded episodes. Run [gripper_timeline.py](../examples/cube_then_banana/gripper_timeline.py), which prints the
gripper command of every replan:

```bash
bin/owsp py base examples/cube_then_banana/gripper_timeline.py
```

```
k=19 step=285  measured 0.00  commanded 0.39 open
k=20 step=300  measured 0.31  commanded 0.90 CLOSE
k=21 step=315  measured 0.31  commanded 0.01 open   <- opens while holding something
k=22 step=330  measured 0.00  commanded 0.03 open
```

At replan `k=20` the gripper closes on the cube (measured opening 0.31: something is in the hand). At `k=21`,
step 315, the policy commands it open, one step before the drop event at 316. That is the decision to fix:
episode `base.base.s20001`, replan 21. (`bin/owsp py base` runs against the unedited weights read-only; set
`EID` at the top of the script to look at another episode.)

## 4. Replay that decision under two wordings

An activation capture (`bin/owsp acts`) replays recorded decisions on the GPU with the camera images, robot
state and noise seed held fixed. Each item can carry its own `prompt`, so one capture can ask the same decision
under two wordings. [two_wordings.json](../examples/cube_then_banana/two_wordings.json):

```json
{
 "id": "two-wordings",
 "cand": "base",
 "items": [
  {"episode": "base.base.s20001", "k": 21, "prompt": "Put the cube then the banana in the bowl"},
  {"episode": "base.base.s20001", "k": 21, "prompt": "Put the cube in the bowl then put the banana in the bowl"}
 ],
 "action_layers": [17], "action_tokens": [0], "denoise_steps": [0]
}
```

`action_layers`, `action_tokens` and `denoise_steps` keep the capture small, since only the actions matter
here. The run below asked the same question at the first 6 decisions that step 5's selection rule picks, 12
items in one capture:

```bash
bin/owsp acts examples/cube_then_banana/two_wordings.json
bin/owsp py base examples/cube_then_banana/compare_wordings.py
```

```
{"id": "two-wordings", "items": 12, "shapes": {...}, "parity_err": 0.011301781789053611, "sec": 8.0}

base.base.s20001 k=21  gripper 0.01 open   (live run: 0.01)  'Put the cube then the banana in the bowl'
base.base.s20001 k=21  gripper 0.85 CLOSE  (live run: 0.01)  'Put the cube in the bowl then put the banana in the bowl'
base.base.s20000 k=10  gripper 0.00 open   (live run: 0.00)  'Put the cube then the banana in the bowl'
base.base.s20000 k=10  gripper 0.00 open   (live run: 0.00)  'Put the cube in the bowl then put the banana in the bowl'
base.base.s20002 k=27  gripper 0.25 open   (live run: 0.25)  'Put the cube then the banana in the bowl'
base.base.s20002 k=27  gripper 0.81 CLOSE  (live run: 0.25)  'Put the cube in the bowl then put the banana in the bowl'
base.base.s20003 k= 7  gripper 0.01 open   (live run: 0.01)  'Put the cube then the banana in the bowl'
base.base.s20003 k= 7  gripper 0.01 open   (live run: 0.01)  'Put the cube in the bowl then put the banana in the bowl'
base.base.s20004 k=34  gripper 0.00 open   (live run: 0.00)  'Put the cube then the banana in the bowl'
base.base.s20004 k=34  gripper 0.81 CLOSE  (live run: 0.00)  'Put the cube in the bowl then put the banana in the bowl'
base.base.s20005 k=22  gripper 0.00 open   (live run: 0.00)  'Put the cube then the banana in the bowl'
base.base.s20005 k=22  gripper 0.72 CLOSE  (live run: 0.00)  'Put the cube in the bowl then put the banana in the bowl'
```

Two things to read here:

- **The replay is faithful.** Under the original wording each replay gives the gripper command the policy
  produced live, and `parity_err` 0.011, the largest action difference, is bf16 noise. Changing only the words
  is therefore a clean experiment.
- **The rewording fixes 4 of the 6 decisions** with nothing else changed: same images, same state, same weights.
  At `s20000 k=10` and `s20003 k=7` both wordings let go. In both episodes the simulator had just logged
  `WRONG_OBJECT_GRABBED_FAILURE` for the banana, so the hand most likely held the banana, and letting go of the
  wrong object is the right call.

So the information is already in the network, and the original wording just fails to route it. Step 5 copies
what the explicit wording does inside the network into the weights.

Capture ids are unique within a project. To capture again, change `"id"`.

## 5. Distil the rewording into the weights

Step 4 showed that the explicit wording carries the right answer. `weightsmith distill` writes what that wording
does inside the network into the weights, so that the original wording produces it too. For each edited layer:

- **keys** `K`: the original wording's activations entering the layer's MLP down-projection, averaged over the
  recorded decisions, at 32 positions that look the same under both wordings (2 cameras × 16 image regions);
- **values** `V`: how much more that layer adds to the residual stream at those positions under the explicit
  wording than under the original one;
- **edit**: `ΔW = Kᵀ(KKᵀ + λI)⁻¹V`, the smallest change to the weight matrix that maps each key to its value.

That is one small linear solve per layer: no gradients, no training data and no search. [DISTILL.md](DISTILL.md)
has the details. [distill.json](../examples/cube_then_banana/distill.json) applies the recipe of the paper's
smallest edit: image regions in VLM layers 8–16, 0.16% of the model's parameters as low-rank factors. Look first,
then run:

```bash
weightsmith distill run examples/cube_then_banana/distill.json --dry-run   # prints every command; runs nothing
weightsmith distill run examples/cube_then_banana/distill.json
```

The output, step by step (commands shortened). **Select** picks failure decisions from your base episodes by
rule, the first early release in each failed episode (grasp, then let go within 2 replans with something in the
hand), and the check groups the same way, never reusing a build decision:

```
project projects/ctb (WSP_ROOT), candidate ctb-imgl8, work dir distill-runs/ctb-imgl8
$ bin/owsp py base distill-runs/ctb-imgl8/select.py
scanned 30 episodes, selected 10 decisions
  base.base.s20000 k=10 step=150 gripper=0.106
  base.base.s20001 k=21 step=315 gripper=0.311
  ...
  base.base.s20009 k=29 step=435 gripper=0.292
10 decision items
check group other_early_releases: 16 decisions selected by rule 'early_release'
check group grasps: 16 decisions selected by rule 'grasp'
```

**Capture** replays the 10 decisions under each wording, recording the VLM's MLP activations and residual
stream at layers 7–16:

```
$ bin/owsp acts distill-runs/ctb-imgl8/ctb-imgl8-o.json
{"id": "ctb-imgl8-o", "items": 10, "shapes": {..., "vlm.img_mlp_hidden": [10, 10, 2, 17, 16384], ...}, "parity_err": 0.0171, "sec": 10.7}
$ bin/owsp acts distill-runs/ctb-imgl8/ctb-imgl8-x.json
{"id": "ctb-imgl8-x", "items": 10, "shapes": {...}, "parity_err": 0.0, "sec": 7.0}
```

(`parity_err` is only measured for replays under the episode's own wording, hence 0.0 for the explicit one.)
**Build** keeps the decisions the wording actually changes (`"responsive_only": true`), the same comparison
as step 4, then writes the edit into a new candidate:

```
decision                     original  explicit
base.base.s20000 k=10        open      open
base.base.s20001 k=21        open      CLOSE    <- the wording changes the decision
base.base.s20002 k=27        open      CLOSE    <- the wording changes the decision
base.base.s20003 k=7         open      open
base.base.s20004 k=34        open      CLOSE    <- the wording changes the decision
...
8 of 10 decisions change with the wording
the edit is built from the 8 decisions that change with the wording
$ bin/owsp new ctb-imgl8 --note 'distill ctb-imgl8: {"alpha": 1.0, "lam_rel": 0.001, "layers": [8, ..., 16], "mode": "image"}'
$ bin/owsp py ctb-imgl8 distill-runs/ctb-imgl8/build.py
L 8 language.blocks.08.mlp.linear      keys  32  fit-rel-err 2.68e-03  |dW|/|W| 0.0169  changed 82.4%
L 9 language.blocks.09.mlp.linear      keys  32  fit-rel-err 2.99e-03  |dW|/|W| 0.0207  changed 83.9%
...
L15 language.blocks.15.mlp.linear      keys  32  fit-rel-err 8.95e-02  |dW|/|W| 0.0787  changed 84.8%
L16 language.blocks.16.mlp.linear      keys  32  fit-rel-err 1.69e-01  |dW|/|W| 0.0566  changed 77.7%
candidate ctb-imgl8 fingerprint 72e042a97074dddc
```

How to read the build lines: `fit-rel-err` is how far `K·ΔW` lands from `V`. It is small, and grows in the last
layers, where the ridge `λ` keeps nearly parallel keys from blowing up the solution. `|dW|/|W|` says each matrix
moves by 2–8%. `changed` counts stored values that differ after rounding to bfloat16: the edit is dense, but it
has rank 32 per layer, so it fits in 5.3 million numbers. The fingerprint identifies these exact weights; every
later eval of the candidate prints it.

**Check** replays the edited policy offline on three groups of recorded decisions: the 10 it was built from,
16 other early releases it never saw, and 16 grasps that it should leave alone:

```
$ bin/owsp acts distill-runs/ctb-imgl8/ctb-imgl8-c-72e042.json
$ bin/owsp acts distill-runs/ctb-imgl8/ctb-imgl8-cb.json
$ bin/owsp py base distill-runs/ctb-imgl8/check.py
decisions (n=10): gripper closed  base 0 | explicit 8 | edit 8   flips vs base 8   edit agrees with explicit 10/10 (base 2/10)
  chunk-end end effector: base->explicit 10.7 cm, edit->explicit 6.7 cm, edit moved 11.7 cm
check grasps                     (n=16): gripper closed base 16 | edit 13   flips  3   edit moved 1.1 cm
check other_early_releases       (n=16): gripper closed base  1 | edit 11   flips 10   edit moved 9.1 cm

next steps
  1. closed loop on practice slots, ORIGINAL instruction, paired with base on the same slots:
       WSP_ROOT=projects/ctb bin/owsp eval ctb-imgl8 --n 32 --offset 0
  ...
```

Read it top to bottom:

- **The decisions it was built from:** under the original wording, the edited policy now makes the explicit
  wording's call at all 10, holding at the 8 that the rewording changes and still letting go of the banana at
  the other 2. Its arm targets also move toward the explicit wording's (6.7 cm away instead of 10.7).
- **Releases it never saw:** it keeps the gripper closed at 11 of 16, where the unedited policy holds at 1.
  So the edit is not just memorising its 8 snapshots.
- **Grasps:** 3 of 16 grasp decisions no longer close. That is a side effect to watch in closed loop, where the
  policy gets another chance to close 15 steps later.

The whole run took 63 minutes of wall time on the shared host. Most of it was waiting for GPU locks: the four
captures used about 2 minutes of GPU (19, 19, 66 and 16 seconds). Everything it wrote is in
`distill-runs/ctb-imgl8/`. Rerunning the same command took 30 seconds and no GPU: it checked that each existing
capture holds the same decisions (`capture ctb-imgl8-o exists with the same 10 decisions: reusing it`) and rebuilt
the candidate bit for bit (same fingerprint). Had the selection changed, it would have stopped instead of reusing
stale captures.

## 6. Run the edited policy on the same seeds

```bash
bin/owsp eval ctb-imgl8 --n 32
```

```
[eval ctb-imgl8 fp=72e042a97074dddc n=32 success=8/32 (25.0%) wall=16.7 min]
[paired vs base on 32 shared slots: base 2.0 vs cand 8 | fixed 8, broke 2]
  ctb-imgl8.72e042.s20000            OK   steps=774 Completed subtask 'pick_and_place' 2/2 | WRONG_GRABBED_FAILURE@120,...
  ctb-imgl8.72e042.s20001            OK   steps=641 Completed subtask 'pick_and_place' 2/2 | GRABBED@46,TARGET_DROPPED@50,...
  ctb-imgl8.72e042.s20002            FAIL steps=900 Condition not satisfied: object_grabbed( | WRONG_GRABBED_FAILURE@120,...
  ...
  ctb-imgl8.72e042.s20011            FAIL steps=900 Condition not satisfied: object_grabbed( | GRABBED@54,TARGET_DROPPED@145,...
  ...
```

Same 32 slots, same original instruction: the change is in the weights only. Episode ids now start with the
candidate and its fingerprint (`ctb-imgl8.72e042.s20000`), so every result stays tied to the exact weights that
produced it, and `bin/owsp show` and `gripper_timeline.py` work on these episodes too. No GPU was free for a
second lane this time (no `extra GPU lane` line), so the run took 16.7 minutes instead of 11.

The edited policy succeeded 8 times against 2: **8 fixed, 2 broken**. Step 7 explains whether that means
anything.

## 7. Read a paired result

Every slot ran once under each policy, so compare seed by seed:

| | edit succeeds | edit fails |
|---|---|---|
| **base succeeds** | both | **broke** |
| **base fails** | **fixed** | neither |

Only the two off-diagonal cells say anything about the edit. `weightsmith stats` does the arithmetic. Export
each slot's outcomes from the project with [paired_csv.py](../examples/cube_then_banana/paired_csv.py), then
compare the two arms:

```bash
bin/owsp py base examples/cube_then_banana/paired_csv.py > paired.csv
weightsmith stats paired.csv --arms base,ctb-imgl8
```

```
base vs ctb-imgl8: 32 paired slots (20000-20031)
  base         2/32 =   6.2%   Wilson 95%: 1.7% to 20.1%
  ctb-imgl8    8/32 =  25.0%   Wilson 95%: 13.3% to 42.1%
  pairs     both 0, fixed 8, broke 2, neither 22  (31.2% of pairs differ)
  change    +18.8 points   paired bootstrap 95%: 0.0 to +37.5
  McNemar   exact p = 0.11   (8 fixed vs 2 broke)
  detects   about 27 points with 32 pairs (80% power, alpha 0.05, 30% of pairs differing)
  verdict   no detectable difference at alpha 0.05; the observed change is smaller than what this many pairs can detect
  note      fewer than 64 pairs: treat this as a screen, not evidence (BENCHMARK.md scores on >= 64 held-out pairs)
```

Line by line:

- **Wilson 95%**: the range each success rate could plausibly have, given only 32 episodes. The two ranges
  overlap.
- **pairs**: the four outcome types. The exact McNemar test asks how likely a split at least as lopsided as
  8 fixed against 2 broke would be if every changed pair were a coin flip: p = 0.11.
- **change**: 18.8 points, with a paired bootstrap interval that reaches down to 0.
- **detects**: the smallest change 32 pairs can reliably see is about 27 points, so a real 19-point gain could
  easily hide in this run.

Two things to keep in mind before you trust any practice number.

**Identical reruns flip about 30% of pairs.** Run the same weights twice on the same slots and many episodes
come out differently. GPU physics and rendering are not bit-deterministic, and the policy executes 15 actions
open loop between looks, so a tiny difference in one contact grows into a different episode. Two runs with
identical weights and seeds agreed on only 46 of 64 episodes, and the unedited policy once scored 14/16 and then
8/16 on the same seeds ([BENCHMARK.md](../BENCHMARK.md#9-noise-floor-and-sample-size)). Any single *fixed* pair
may be luck. The signal is the balance of fixed against broke over many pairs.

**32 episodes is a filter, not evidence.** With that much churn, 32 pairs reliably detect only changes of about
27 percentage points (64 pairs: 19, 128: 14). And these are practice slots: you picked decisions from them, and
the edit was built from their recordings. Use a 32-episode run to drop candidates that do nothing or break the
task. Confirm a survivor on fresh practice slots (`bin/owsp eval base --n 64 --offset 32`, then the candidate
with the same arguments), and claim nothing before an operator has scored it on held-out slots.

**What this run says.** 8 fixed against 2 broken passes the filter ("worth a closer look") and proves nothing
yet. The success bit is also a coarse signal; the recorded events say more.
[carries.py](../examples/cube_then_banana/carries.py) measures how long the robot holds the cube:

```bash
bin/owsp py base examples/cube_then_banana/carries.py
```

```
base        2/32 succeeded   object in the bowl in  3   170 carries, median  8 steps before the drop
ctb-imgl8   8/32 succeeded   object in the bowl in 14   177 carries, median 31 steps before the drop
```

The edit does what it was built to do: the robot now carries the cube about four times longer before letting go,
and gets an object into the bowl in 14 episodes instead of 3. Most episodes still fail later. Two more things
stand out:

- Both broken slots are the two where the unedited policy succeeded. One of them, `s20011`, opens with a grasp
  that the offline check flagged: the edit no longer closes there at replan 3. That could be the side effect at
  work, or noise; 32 pairs cannot tell.
- The paper's version of this recipe scored 25/32 against 3/32 on its own practice slots, so this edit is
  clearly weaker. One suspect is the selection: the rule took the first release that fit it in each failed
  episode, and here those came mostly after many retries (replans 18 to 50). The paper built its edit from
  releases on the first carry.

### Iterate

When a screen passes but falls short, let the offline check and the events pick the next try, not held-out
slots. Here the next try is to build from first-carry releases.
[distill-first-carry.json](../examples/cube_then_banana/distill-first-carry.json) keeps decisions up to sim step
150 (`max_step`), allows carries of up to 4 replans (`max_hold`), and takes every such release in an episode
(`first_only: false`). It has its own `name`, so it captures afresh instead of reusing the first edit's captures:

```bash
weightsmith distill run examples/cube_then_banana/distill-first-carry.json
bin/owsp eval ctb-first --n 32
bin/owsp py base examples/cube_then_banana/paired_csv.py > paired.csv
weightsmith stats paired.csv --arms base,ctb-first
```

The selection now finds 10 releases between replans 4 and 10, and the rewording changes 7 of them. The offline
check of the new candidate (`ctb-first`, fingerprint `3047901c3502ffea`), next to the first edit's:

```
decisions (n=10): gripper closed  base 0 | explicit 7 | edit 7   flips vs base 7   edit agrees with explicit 10/10 (base 3/10)
  chunk-end end effector: base->explicit 9.0 cm, edit->explicit 4.0 cm, edit moved 7.0 cm
check grasps                     (n=16): gripper closed base 16 | edit 15   flips  1   edit moved 0.8 cm
check other_early_releases       (n=16): gripper closed base  1 | edit 13   flips 12   edit moved 4.8 cm
```

| Offline check | `ctb-imgl8` (first edit) | `ctb-first` |
|---|---|---|
| early releases it never saw, gripper kept closed (unedited replay: 1 of 16 each) | 11 of 16 | 13 of 16 |
| grasps that no longer close | 3 of 16 | 1 of 16 |
| arm targets, distance to the explicit wording's | 6.7 cm (unedited 10.7) | 4.0 cm (unedited 9.0) |

Better on every line, so it earns a closed-loop run:

```
[eval ctb-first fp=3047901c3502ffea n=32 success=19/32 (59.4%) wall=15.8 min]
[paired vs base on 32 shared slots: base 2.0 vs cand 19 | fixed 18, broke 1]
```

```
base vs ctb-first: 32 paired slots (20000-20031)
  base         2/32 =   6.2%   Wilson 95%: 1.7% to 20.1%
  ctb-first   19/32 =  59.4%   Wilson 95%: 42.3% to 74.5%
  pairs     both 1, fixed 18, broke 1, neither 12  (59.4% of pairs differ)
  change    +53.1 points   paired bootstrap 95%: +34.4 to +71.9
  McNemar   exact p = 7.6e-05   (18 fixed vs 1 broke)
  detects   about 27 points with 32 pairs (80% power, alpha 0.05, 30% of pairs differing)
  verdict   ctb-first better than base (p < 0.05)
  note      fewer than 64 pairs: treat this as a screen, not evidence (BENCHMARK.md scores on >= 64 held-out pairs)
```

```
base        2/32 succeeded   object in the bowl in  3   170 carries, median  8 steps before the drop
ctb-imgl8   8/32 succeeded   object in the bowl in 14   177 carries, median 31 steps before the drop
ctb-first  19/32 succeeded   object in the bowl in 25   122 carries, median 34 steps before the drop
```

19 against 2, with 18 fixed and 1 broken: a gain of 53 points, twice what 32 pairs can detect, and close to the
paper's 25/32 against 3/32 for this recipe. Head to head on the same slots, `ctb-first` beats the first edit too
(`weightsmith stats paired.csv --arms ctb-imgl8,ctb-first`: 14 fixed, 3 broken, p = 0.013). The decisions an edit
is distilled from matter: the same recipe, built from the failure you actually care about, more than doubled the
success rate. The offline check predicted the ranking, and it cost 2 minutes of GPU instead of 16.

Comparing candidates on the same 32 practice slots is fine for choosing between them. Choosing on practice slots
is also exactly why the winner then needs a held-out score.

## 8. The held-out score (operator only)

Practice slots are for the editor. The score that counts comes from slots nobody looked at while building the
edit, run by an operator: base and candidate on the same unseen slots, in alternating order, one shot per slot
block. On the GPU host:

```bash
WSP_ROOT=$PWD/projects/ctb nohup setsid python3 weightsmith/heldout.py --cand ctb-first --slot-start 95008 --n 64 \
    > heldout-ctb-first.log 2>&1 < /dev/null &
tail -f heldout-ctb-first.log
```

- `--slot-start`: at least 80000, a multiple of 16, and never used for this candidate before. heldout.py
  rejects anything else before it starts.
- `--n`: a multiple of 32, at least 64 pairs, and 128 for a headline claim.
- It first freezes a copy of the candidate and checks its fingerprint, then runs 4 batches of 32 episodes. That
  is about 45 minutes on two free GPUs and 90 on one, so start it detached as above.
- `heldout/<tag>/` is never overwritten. A slot block gets one shot: running it again until the numbers look
  better would make them meaningless.

**The held-out score for this tutorial's edit.** The operator ran exactly the command above afterwards. `ctb-first`,
built end to end by `weightsmith distill` with no hand-picked decisions, held up on 64 slots nobody had seen:

```text
base 8/64 -> ctb-first 36/64   fixed 33, broke 5   exact McNemar p = 4.3e-6
Wilson 95%: 6.5-22.8% unedited, 44.1-67.7% edited   fingerprint 3047901c3502ffea   slots 95008-95071
```

That clears the benchmark's bar (at least 64 held-out pairs, p < 0.05). For reference, this is the summary heldout.py
printed for the paper's version of the same recipe (`abl-imgl8`, built from 6 hand-picked decisions; slots
90000–90127):

```json
{
 "cand": "abl-imgl8",
 "fingerprint": "5bf8c4329c1e416a",
 "paired_n": 128,
 "base_success": 19,
 "cand_success": 102,
 "base_wilson95": [0.097, 0.22],
 "cand_wilson95": [0.719, 0.857],
 "fixed": 86,
 "broke": 3,
 "mcnemar_exact_p": 3.7988915866351353e-22,
 "crashed_slots": 0
}
```

Read it like step 7: 86 fixed against 3 broken on 128 unseen pairs is far beyond the noise floor. The Wilson
intervals bound each success rate. `weightsmith stats $WSP_ROOT/heldout/<tag>/results.json --halves` prints the
same report as in step 7 from a held-out run, plus each half of the slot block (the two halves ran in opposite
order). The paper's full file is
[study/results/opus/heldout/](../study/results/opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json).

## Where next

- **Another task or wording.** Copy [distill-first-carry.json](../examples/cube_then_banana/distill-first-carry.json),
  change `name`, `instruction`, `explicit_instruction` and the `select` rule (`early_release` or `grasp`), and start
  with step 4: if no rewording changes the failing decisions, distillation has nothing to copy.
- **The method and every config field:** [DISTILL.md](DISTILL.md). The paper's full diagnosis and the attempts
  that failed first: [study/results/opus/ctb/](../study/results/opus/ctb/README.md).
- **Other edits and how to check them:** copy-paste recipes in [COOKBOOK.md](COOKBOOK.md), and the ideas behind
  them in [CONCEPTS.md](CONCEPTS.md).
- **Let an LLM do the editing:** the same tools as an MCP server ([INTEGRATIONS.md](INTEGRATIONS.md)), or a
  headless editor under a budget ([drivers/](../drivers/README.md), [BENCHMARK.md](../BENCHMARK.md)).
- **Share the edit as one file:** `weightsmith edit export $WSP_ROOT/cands/ctb-imgl8 -o ctb-imgl8.safetensors
  --rank auto`. For this edit that is a 272 MB file holding rank-32 factors per layer plus bf16 fix-ups, exact when
  applied, after about 5 minutes of CPU ([INTEGRATIONS.md](INTEGRATIONS.md#portable-edit-files)).

## Troubleshooting

| Symptom | What happened, and what to do |
|---|---|
| A command prints nothing for minutes | It waits for a GPU lock held by another job; it starts on its own. `ps aux \| grep -E 'owsp.py\|heldout.py'` shows who is ahead. Waiters are not served in order. |
| `weightsmith: command not found` | The environment you installed into is not active: `source .venv/bin/activate`. |
| `owsp: no project config at .../config.json` | `WSP_ROOT` is unset in this shell, or points somewhere else. |
| `bad or existing acts id two-wordings` | Capture ids are unique within a project. Change `"id"` in the spec. |
| `owsp py` prints `[stderr] Traceback ...` and `[exit 1]` | Your script failed inside the sandbox. owsp itself still exits 0, so read the output. |
| `capture 'ctb-imgl8-o' already exists but holds other decisions` | You changed the selection after an earlier run. Give the config a new `name`; capture ids derive from it. |
| `the select rule found no decisions in the project's episodes` | Run step 2 first, or relax the rule (`max_hold`, `max_step`, `success`). |
| `the explicit wording changes none of the selected decisions` | Distillation has nothing to copy. Check step 4 on more decisions, or try another rewording. |
| `prompt overrides are disabled in this project` | The project's config sets `"prompt_overrides": false`. Use `configs/cube-then-banana.json`, which allows them. |
| `--slot-start must be a multiple of envs = 16` | Held-out slots run in groups of 16 (95008, not 95000). |
| `.../heldout/<tag> exists; held-out runs are never repeated or overwritten` | That slot block was already scored for this candidate. Pick a fresh block: rerunning a block until the numbers look better would make the score meaningless. |
| Anything about `wsp.env`, GPUs or the simulator | See [SETUP.md](SETUP.md#troubleshooting). |
