# Cookbook: what to try, how, and how to tell whether it worked

Tested recipes for weight-space programming on π0.5. If *known-good input*, *paired slots* or *failure level* are
new words, read [CONCEPTS.md](CONCEPTS.md) first (two pages).

Each recipe is a short Python file in [`examples/recipes/`](../examples/recipes/) that runs in the `owsp py`
sandbox, on the GPU host's CPU. It starts with a docstring (when to use it, how to tell it worked, where the
pattern came from) and a `SETTINGS` block whose defaults point at the cube-then-banana project of the paper. We ran
every recipe on 2026-11-07 against the recorded episodes and activation captures of that project; each section
reports what it produced. [`tests/test_recipes.py`](../tests/test_recipes.py) runs all of them on synthetic data
in CI, so they keep working.

## The loop

```
 record the unedited policy ─▶ 1 find the failing decisions ─▶ 2 prompt swap: is there a known-good input?
                                                                  │ no  → the skill or signal is missing: train, don't edit
                                                                  ▼ yes
 3 capture activations ─▶ 4 build an edit ─▶ 5 offline check ─▶ 6 closed loop on practice slots, paired statistics
                                                                  ▼
                       4.5 random-direction control · operator's held-out run (≥ 64 pairs) · other tasks
```

| Step | Recipe | Run with | Spends |
|---|---|---|---|
| 1. Find the failing decisions | `01_find_decisions.py` | `owsp py base` | nothing |
| 2. Prompt-swap replay | `02_prompt_swap.py` + spec `prompt_swap.json` | `owsp acts`, then `owsp py base` | activation items |
| 3. Capture specs | `specs/*.json`, `check_spec.py` (local), `03_capture_info.py` | `owsp acts` | activation items |
| 4.1 Distil a known-good wording | `04_edit_distill.py` | `owsp py CAND` | nothing |
| 4.2 Gated rank-1 readout | `05_edit_gated_readout.py` | `owsp py CAND` | nothing |
| 4.3 Repurposed unit as a gate | `06_edit_repurpose_unit.py` | `owsp py CAND` | nothing |
| 4.4 Strengthen a lexical unit | `07_edit_lexical_unit.py` | `owsp py CAND` | nothing |
| 4.5 Random-direction control | `08_random_control.py` | `owsp py CAND` | nothing |
| 5. Offline validation | `09_offline_check.py` + spec `candidate_check.json` | `owsp acts`, then `owsp py base` | activation items |
| 6. Paired evaluation | `10_paired_outcomes.py`, `weightsmith stats` | `owsp eval`, then `owsp py base` | practice episodes |

## Before you start

- A GPU host set up with [SETUP.md](SETUP.md), a project root holding `config.json`, and `bin/owsp` pointed at it
  (`WSP_ROOT`, plus `WSP_HOST` and `WSP_REMOTE_REPO` when you work from another machine).
- The unedited policy recorded on practice slots: `scripts/base_practice.sh 64` or `bin/owsp eval base --n 64`.
  Every recorded episode keeps the exact policy inputs of every replan, so you can replay any decision later.
- Read-only recipes run as `bin/owsp py base examples/recipes/01_find_decisions.py`. Edit recipes write into a
  candidate: `bin/owsp new my-edit`, then `bin/owsp py my-edit examples/recipes/04_edit_distill.py`. Change the
  `SETTINGS` block in your own copy.
- The sandbox has no network and no GPU, about 100 GB of RAM and a 30-minute limit. It provides `load`, `base`,
  `save`, `bf16`, `saved` (weights) and `episodes`, `obs`, `acts` (data). Python state does not survive between
  calls; write what you need to keep to `/work/notes` (shared by the project) or `/work/scratch` (per candidate).

## 1. Find the failing decisions

**When.** You have watched some failures (`owsp show EID`, `owsp frames EID --steps ...`) and want the exact replans
where the policy goes wrong, plus replans that must not change, as capture items.

**Run.** `bin/owsp py base examples/recipes/01_find_decisions.py`

**What it does.** It splits every recorded episode into *holds*: runs of replans where the measured gripper says an
object is in the hand (0 is open, about 1 is closed on nothing, and the object's width sets the value in between:
about 0.35 for the cube and 0.58 for the banana here). A hold that ends with a chunk commanding "open" is a release
decision. Releases far from where successful episodes release are premature. It prints counts, a few episode
timelines, and JSON item lists for targets (premature releases) and preservation sets (releases at the goal, carries
and approaches in successful episodes).

**Check it worked.** The targets should be the failure you saw in the videos, and successful episodes should give
releases near the goal.

**On cube-then-banana.** Over 64 base episodes (9 successes) it placed the goal at (0.48, 0.06) m from 28
releases in successful episodes and found 22 premature releases, one per episode at most: all 21 decisions the
paper used plus one more (s20059, replan 10). It also listed 8 goal releases, 35 carry replans and 64 approach
replans as preservation candidates.

**Detail that matters.** Detect holds from the measured gripper, not from the chunk alone: many grasps here close
only in the last 3 of the chunk's 15 steps, so a majority-vote chunk looks "open" although the cube is in the hand at
the next replan. A first version that voted on chunks found only 12 of the 21.

## 2. Prompt-swap replay

**When.** Before any edit. This is the test for the known-good input rule: if some wording makes the same weights
decide correctly on the recorded failures, the failure is editable; if none does, look for a missing skill.

**Run.** Capture the target decisions under several wordings (per-item `"prompt"`), for example
[`specs/prompt_swap.json`](../examples/recipes/specs/prompt_swap.json) (60 items: 30 decisions under two wordings,
actions only, about 3 MB): `bin/owsp acts examples/recipes/specs/prompt_swap.json`. Then
`bin/owsp py base examples/recipes/02_prompt_swap.py` with your capture ids in `SETTINGS`.

**What it does.** It groups items by wording, pairs them by decision with the reference wording, and reports, per
wording, how many decisions keep the gripper closed, how many flip, and how far the chunk's end point moves
(forward kinematics of the 7 joint targets).

**Check it worked.** The reference row must show what the policy did live; for base captures the capture's
`parity_err` is about 0.01–0.1.

**On cube-then-banana** (the 21 premature releases, captures `a1-*`, `b1-parse`, `b2-*`), the paper's Fig. 2a:

| Wording | Hold | Release |
|---|---:|---:|
| Put the cube then the banana in the bowl *(given)* | 0 | 21 |
| Put the rubiks cube then the banana in the bowl | 0 | 21 |
| Put the cube, then the banana in the bowl | 0 | 21 |
| Put the banana in the bowl | 0 | 21 |
| Put the cube then the banana | 1 | 20 |
| Put the cube before the banana in the bowl | 10 | 11 |
| Put the cube and the banana in the bowl | 12 | 9 |
| Put the banana then the cube in the bowl | 18 | 3 |
| Put the cube in the bowl then the banana | 18 | 3 |
| Put the cube in the bowl then put the banana in the bowl | 19 | 2 |
| Put the cube in the bowl | 20 | 1 |

Renaming the cube changes nothing; putting "in the bowl" next to the cube flips 18–20 of 21. The policy binds "in
the bowl" to the noun next to it. The explicit wording is the known-good input (22/32 against 3/32 in closed loop,
`owsp eval base --prompt ...`, diagnosis only).

**When no wording works.** The blind yellow-on-red editor replayed 8 decisions under 8 wordings, even "stack the
*green* block", and the arm went to the same place every time. With nothing to distil, it submitted no edit.

## 3. Activation-capture specs

`owsp acts SPEC.json` replays recorded decisions through an instrumented copy of the policy and saves activations.
It charges one activation item per decision replayed, and each call takes about a minute to load the model plus
well under a second per item. Six specs in [`examples/recipes/specs/`](../examples/recipes/specs/) cover the
common needs; their items are real cube-then-banana decisions (the 21 premature releases, 9 cube carries and 19
banana-phase decisions of the paper's offline checks).

| Spec | Captures | Use it for | Size |
|---|---|---|---|
| `prompt_swap.json` | actions only, under two wordings | prompt-swap diagnosis (section 2) | 0.04 MB/item |
| `prompt_tokens.json` | VLM residual and MLP activations at prompt slots 0–63, all 18 layers | lexical units (4.4), prompt-position distillation (4.1) | 42.5 MB/item |
| `image_tokens.json` | VLM image cells (2 cameras × 4×4 + mean), layers 0–16 | image-cell distillation (4.1), perception | 22 MB/item |
| `attention.json` | action-expert attention mass and per-patch maps, denoising steps 0, 5, 9 | what the expert looks at | 0.7 MB/item |
| `action_readout.json` | expert's last hidden state and block-17 MLP at all 10 denoising steps | readouts (4.2) and gates (4.3) | 0.4 MB/item |
| `candidate_check.json` | actions of a *candidate* on target + preservation decisions | offline validation (section 5) | 0.04 MB/item |

The keys the capture worker reads:

| Key | Meaning |
|---|---|
| `items` | `[{"episode": EID, "k": replan}]` or `{"episode": EID, "step": sim_step}` (nearest replan at or before); optional per-item `"prompt"` |
| `cand` | `"base"` or a candidate name: whose weights replay the decisions |
| `vlm_layers`, `action_layers` | which of the 18 layers to keep (VLM default none; action default all) |
| `prompt_tokens` | prompt slots to keep, up to 64 (default 0–23); slot i is the i-th token of `<bos>Task: ..., State: ...;\nAction:` |
| `image_tokens` | `true` adds `vlm.img_mlp_hidden` / `vlm.img_resid_out` `[N, layers, camera, 17, F]` (cell 0 is the camera mean) |
| `action_attention` | `true` adds `action.attn_mass` (mass on base camera, wrist camera, padding, prompt+state, action tokens) and `action.attn_map` (16×16 per camera) |
| `denoise_steps` | which of the 10 flow steps to keep (step i is t = 1 − 0.1 i; default all) |
| `action_tokens` | `"mean"` over the 15 action tokens, or a list of token indices |

**Check a spec before spending items.** `python3 examples/recipes/check_spec.py my_spec.json` flags typos (the
worker silently ignores unknown keys, so `"vlm_layer"` would capture nothing and still charge the items), bad
ranges, more than 96 items per call, and estimates the size. To confirm the episodes and replans exist, paste the
items into `check_items` of `03_capture_info.py`.

**Inspect a capture.** `bin/owsp py base examples/recipes/03_capture_info.py` prints each capture's items per
wording, layers, steps, arrays with shapes and sizes (read from the file headers, without loading the data),
parity, and a spec that reproduces it. On cube-then-banana: `c2-multi` holds 48 items (6 decisions × 8 wordings),
3,150 MB, 65.6 MB per item, mostly VLM prompt-slot (1.8 GB) and image-cell (0.96 GB) activations; `ctb-ph` holds the
49 offline-check decisions (735 MB); `chk-v2mi-f` holds the same 49 replayed by the distilled edit (2 MB). All 49
items of the six example specs exist in the project.

**Parity.** For base captures under the recorded wording, `parity_err` compares the replay with what the policy did
live; about 0.01–0.1 is bf16 noise. It means nothing when the wording was changed or for candidate captures.

## 4. Edit patterns

Every pattern below comes from a real edit, credited in its section. Pick by what the diagnosis found:

| You found | Pattern | Size here |
|---|---|---|
| A wording that makes the same weights decide right (section 2) | 4.1 distil the known-good wording | 5.3M factors (rank 32 × 9 layers) |
| The failure lands on one output, and the expert's last hidden state separates the failing decisions | 4.2 gated rank-1 readout | 1,024 values |
| As 4.2, but a linear readout also fires on states that must not change | 4.3 repurpose an unused unit as a one-sided gate | 3,072 values |
| One word steers the decision, too weakly | 4.4 strengthen a lexical unit | 2 rows (4,094 values for mustard) |
| Any edit you want to believe | 4.5 random-direction control | same as the edit |

### 4.1 Distil a known-good wording

*Credit: the operator session's edit in the paper (Section 2; [`v2mi_edit.py`](../study/results/opus/ctb/v2mi_edit.py)),
using the multi-key closed form of ROME/MEMIT. `weightsmith distill` ([DISTILL.md](DISTILL.md)) packages the same build.*

**When.** Section 2 found a known-good wording, but the robot must keep receiving the original instruction.

**Run.** Capture the target decisions under both wordings with VLM activations (`image_tokens.json`, or
`prompt_tokens.json` for prompt positions), then `bin/owsp new my-distill` and
`bin/owsp py my-distill examples/recipes/04_edit_distill.py` with `capture`, `orig_items` and `expl_items` set to
your capture (in both example specs, items 0–5 carry the original wording and 6–11 the explicit one).

**How it works.** For each VLM layer L from a chosen set (0–16; layer 17's MLP output never reaches the actions) and
each position present in both wordings (an image cell, or a template token such as `State`):

- the key is the layer's MLP hidden activation there under the **original** wording, averaged over the decisions;
- the value is the change the **known-good** wording makes to what this layer adds to the residual stream there;
- the update ΔW = Kᵀ(KKᵀ + λI)⁻¹V is the smallest change to `language.blocks.LL.mlp.linear` that maps every key to
  its value (λ = 10⁻³ · tr(KKᵀ)/n). It has rank at most n, the number of keys.

**Check it worked.** Each layer maps its keys close to their values (fit error 3×10⁻³ in the middle layers, up to
0.2 in layers 15–16 where the keys are nearly collinear), and the printed fingerprint identifies the edit. Then
validate offline (section 5).

**On cube-then-banana.** With the defaults (32 image cells, layers 8–16, decisions 0–5 vs 12–17 of `c2-multi`), it
rebuilt the paper's 0.16% variant `abl-imgl8` bit for bit: fingerprint `5bf8c4329c1e416a`, 248,057,499 stored
values changed (7.4% of the model), 5.3M values as low-rank factors, in about 20 seconds. With
`prompt_slots: "pi05"` and layers 0–16 it rebuilt the full edit `v2mi-f` (`63db0c52a51306c9`, 450,468,837 values,
72 keys per layer). Offline the full edit flips 21/21 premature releases and no other gripper decision; on 128
held-out seeds it scored 106 against 27.

### 4.2 Gated rank-1 readout

*Credit: the blind cube-then-banana editor of the controlled study
([`ctb/`](../study/results/ctrl/ctb/FINAL.md), `scripts/build_p1b.py`). Its 1,024-value edit is the one autonomous
edit that passed held-out scoring: 11/64 → 27/64, p = 0.0037. It is in the [edit zoo](../edits/README.md) as
`edits/ctrl-ctb-p1.safetensors`, which `weightsmith edit apply` turns back into the scored candidate.*

**When.** The failure is one output channel (here the gripper opens too early) and the action expert's last hidden
state h (`action.head_input` in captures) tells the failing decisions apart from the ones that must not change.

**Run.** Capture targets and preservation decisions with `action.head_input` (`action_readout.json`), then
`bin/owsp new my-readout`, `bin/owsp py my-readout examples/recipes/05_edit_gated_readout.py`. The default groups
are the item ranges of that spec and of `ctb-ph`: 0–20 premature releases, 21–29 cube carries, 30–36, 37–41 and
42–48 banana approach, grasp and carry.

**How it works.** The action head computes the flow velocity v = h W + b. The edit adds `gain · a` to column 7 of
W (the gripper), so the gripper velocity moves by `gain · (h · a)` and nothing else changes. The key a is the
least-norm vector whose mean response is 1 on the targets and 0 on each preservation group. Before solving, it
projects out the directions in which h varies across the 15 tokens of one chunk: those carry each token's own noise
sample, not the robot's state (the blind editor's first version skipped this, and its key fired on noise).

**Check it worked.** (1) In-sample responses near 1 on targets and 0 on preservation groups. (2) The split-half
check: a key built from even-numbered episodes still separates the odd-numbered ones, and the reverse. The blind
stack-three-cubes editor's readout separated perfectly in-sample and failed this check, so it never ran.
(3) The first-order prediction flips targets only. It assumes the key is the same at all 10 denoising steps;
capture `action_readout.json` (all steps) for a real build, then validate by replay (section 5).

**On cube-then-banana** (`ctb-ph`, which holds denoising step 0 only): key response 1.00 on the 21 premature
releases (0.59 to 1.17), 0.71 on cube carries (left free: they already hold), and 0.00 on banana approach, grasp
and carry. Split-half: held-out targets average 0.91–1.01, held-out banana groups at most 0.26 (single decisions up
to 0.42). Predicted: 21/21 releases become holds, no preservation decision flips. The edit changes 1,024 values
with a norm of 3.41, four times the column's own norm (0.85); the blind editor's was 3.16.

### 4.3 Repurpose an unused unit as a gate

*Credit: the blind eyeglasses editor ([`glasses/`](../study/results/ctrl/glasses/FINAL.md), `scripts/build_p1b.py`;
3,072 values; development 12/32 → 18/32, held-out 18/64 → 27/64, p = 0.12). The stack-three-cubes editor proposed
the same design after its linear readout failed the split-half check, and built it in a second round
([`FINAL2.md`](../study/results/ctrl/stack3/FINAL2.md), `scripts_round2/build.py`), constraining the output row at all
10 denoising steps.*

**When.** A linear readout is not selective enough: its response goes negative or stays large on states that must
not change, so pushing harder breaks them. A GELU unit with a threshold is one-sided: zero below it.

**Run.** With the same kind of capture as 4.2 (it needs `action.mlp_hidden` and `action.resid_out` at block 17):
`bin/owsp new my-unit`, `bin/owsp py my-unit examples/recipes/06_edit_repurpose_unit.py`. It uses the first captured
denoising step.

**How it works.** It rebuilds the input of block 17's MLP from the captured residual and MLP activations (the
action expert's norms are adaptive: their scale, shift and gate come from the denoising time), picks the least-used
unit (mean activation times the norm of its output row), and rewires it: the gate column becomes a threshold on the
network's own target-vs-silent contrast, set so the pre-activation is at most −5 on every silent decision; the up
column makes the second factor about 1; the output row is the least-norm vector that, through the final norm and
the action head, moves the gripper velocity by `target_dv` at the median target activation and leaves the other 31
output dimensions alone.

**Check it worked.** The rebuilt activations match the captured ones (correlation about 0.99), the unit is zero on
every silent decision, in-sample and in the split-half check, and positive on most targets. The eyeglasses
editor's first version failed replay (pushing at every denoising step shifts the flow state, and the model
reacts nonlinearly: lifts ranged from −13 to +23 cm); its final version wrote only at the last step, which needs
captures of all 10 steps.

**On cube-then-banana** (`ctb-ph`, step 0): it picked unit 2437 (usage 0.0008 against a median of 0.29), the unit
both the eyeglasses editor and the stacking editor's second round repurposed on their own tasks. Rebuilt activations correlate 0.990 with the captured ones.
The gate fires on 18 of 21 premature releases and is exactly zero on all 28 silent decisions; split-half, it fires
on 6/9 and 7/12 held-out targets and stays zero on held-out silent ones. Predicted first order: 14 of 21 releases
become holds, nothing else changes (other output dims move by less than 10⁻¹⁴). 3,072 values changed. The output
row's norm is 259, eight times that of the whole `linear` tensor, because block 17's adaptive gate is about 0.008
at t = 1 (the eyeglasses editor's row had norm 1,247): when the unit fires, it dominates the block's output.

### 4.4 Strengthen a lexical unit

*Credit: the blind mustard editor ([`mustard/`](../study/results/ctrl/mustard/FINAL.md), `scripts/mk_p1.py`; 4,094
values). The cube-left-of-bowl editor independently found the same two "left" detector units, 3097 and 8023.*

**When.** A word steers the decision, but weakly. In the mustard task "left" sent only 8 of 32 replayed carries
left while "right" sent 27 of 32 right.

**Run.** Capture the decisions under the word and a contrast word at the same token position with prompt-slot
activations (`prompt_tokens.json`), then `bin/owsp new my-lexical`,
`bin/owsp py my-lexical examples/recipes/07_edit_lexical_unit.py`.

**How it works.** Row u of `language.blocks.LL.mlp.linear` is what unit u writes, scaled by its activation. A unit
that fires only on one token is a private write port into that token: adding γ·dr to its row adds β·dr to the
token and almost nothing elsewhere, with dr the mean residual difference between the two wordings at that position
and γ = β / (sum of the units' activations on the token).

**Check it worked.** The realized change at the token points along dr (cosine ±1, size |β|·|dr|) and the leakage onto
other tokens and image cells is small. Then check the decision **and** the outcome: the mustard edit sent 30/32
closed-loop carries left (13/32 before) and still moved held-out success only from 8/64 to 11/64 (p = 0.61), because
the cap grasp kept slipping.

**On cube-then-banana** (a demonstration with an untested hypothesis): "and" holds 12 of the 21 releases (section 2),
so the default moves the instruction's "then" toward "and" (β = −1). No layer-0 unit is selective for "cube", but two
are for "then" (units 9260 and 11175, firing 1.11 and 0.52 on it and at most 0.11 anywhere else). The realized change
has cosine −1.000 with dr at exactly |dr| = 104; leakage is at most 7.1 on the 2,324 other token positions (mean 1.6)
and 3.0 on image cells. Weakly firing units need large rows: the two rows change the tensor by 45% in norm. The
cube-left-of-bowl editor found that downstream layers ignored a similar layer-0 change, so replay it before trusting it.

### 4.5 Random-direction control

*Credit: the paper's control for the distilled edit (Appendix C; [`ctl_rand.py`](../study/results/opus/controls/ctl_rand.py)).
On held-out seeds it scored 9/64 against 13/64 unedited (p = 0.45) while the edit scored 50/64.*

**When.** Before believing any gain, and always when the edit is large. Score the control exactly like the edit.

**Run.** `bin/owsp new my-control --from my-edit`, then `bin/owsp py my-control examples/recipes/08_random_control.py`.
Run it once on a fresh copy: a second run would randomize the control itself.

**How it works.** For each edited tensor it replaces the change with a random one of the same size. Dense low-rank
changes (at least half the values change) keep their singular values, so rank, spectrum and Frobenius norm match,
with random singular vectors. Sparse changes (a column, a few rows) keep exactly which values change and their total
norm, with random values.

**Check it worked.** The control's norm and kept singular values match the edit's, and the dropped singular values
(bf16 rounding noise) sit well below the kept ones.

**On cube-then-banana.** On a copy of the 0.16% edit, the control kept 31–34 singular values per layer (the true rank
is 32; estimated from rounded values, so set `rank` when you know it) and matched each layer's norm within 1%; it
changes 29–32M values per layer against the edit's 26–29M, as the paper's control did (502M against 450M for the full
edit). On a copy of the readout edit it kept the same 1,024 values with the same norm.

## 5. Offline validation

**When.** After building a candidate, before any closed-loop episode. Five of the seven blind editors rejected at
least one candidate here and kept their episode budget, and the paper's operator stopped seven edits here before
the distilled one.

**Run.** Replay the target and preservation decisions with the candidate: `candidate_check.json` with your candidate
in `"cand"` (`bin/owsp acts`). Then `bin/owsp py base examples/recipes/09_offline_check.py` with the base capture,
the candidate captures, the item groups and, optionally, a replay of the known-good wording on the same decisions.

**What it does.** Because the replay reuses the recorded images, robot state and noise, any difference comes from
the weights. Per group it counts gripper decisions that flip, how far the chunk's end point moves, and how close the
edit lands to the known-good wording. A candidate passes when most targets flip and no preservation decision does.
Preservation chunks may still move when the known-good wording moves them too; set `max_shift_cm` to make motion a
criterion.

**On cube-then-banana** it recomputes the paper's Fig. 2b from existing replays:

| Candidate | Releases flipped (of 21) | Preservation flips (of 28) | Verdict |
|---|---:|---:|---|
| `v2mi-f` (distilled, full) | 21 | 0 | pass |
| `v2p-f` (with preservation keys) | 20 | 0 | pass |
| `abl-imgl8` (0.16% variant) | 19 | 0 | pass |
| `v2m-2` (prompt/state positions, strength 2) | 13 | 0 | fail |
| `v2m-f` (strength 1) | 8 | 0 | fail |
| `v2m-h` (strength 0.5) | 6 | 0 | fail |
| `st11a5`, `st1016a4` (rank-1 "target axis" steering) | 0 | 0 | fail |

The distilled edit lands 4.7 cm from the explicit wording's chunk ends on the premature releases, against 8.7 cm
for the unedited policy (4.5 against 8.1 cm over all 30 decisions, as in the paper).

**Passing is necessary, not sufficient.** The blind flat-box editor's readout passed replay (holding decisions
committed 11/16 instead of 6/16) and scored 0/32 in closed loop.

## 6. Paired evaluation and reading the statistics

**Practice runs.** Evaluate the candidate on the same practice slots as base (`bin/owsp eval my-edit --n 64`), export
the per-slot outcomes, and read them with `weightsmith stats`:

```bash
bin/owsp py base examples/recipes/10_paired_outcomes.py > practice.csv     # one column per arm, '#' comments
weightsmith stats practice.csv --arms base,v2mi-f
```

On cube-then-banana this gives 3/32 → 25/32 for the distilled edit (22 fixed, 0 broken, p = 4.8×10⁻⁷) and 3/32 →
22/32 for the explicit wording with unchanged weights (20 fixed, 1 broken), the paper's Section 2 numbers.

**Held-out runs.** The operator's `weightsmith/heldout.py` writes `results.json` with per-slot outcomes; read it
directly. The controlled study's accepted edit, from its published record:

```
$ weightsmith stats study/results/ctrl/heldout/ctrl-ctb__p1-RubiksCubeThenBananaTask-91008-64.json --halves
[study/results/ctrl/heldout/ctrl-ctb__p1-RubiksCubeThenBananaTask-91008-64.json]
base vs p1: 64 paired slots (91008-91071)
  base       11/64 =  17.2%   Wilson 95%: 9.9% to 28.2%
  p1         27/64 =  42.2%   Wilson 95%: 30.9% to 54.4%
  pairs    both 5, fixed 22, broke 6, neither 31  (43.8% of pairs differ)
  change   +25.0 points   paired bootstrap 95%: +9.4 to +40.6
  McNemar  exact p = 0.0037   (22 fixed vs 6 broke)
  detects  about 19 points with 64 pairs (80% power, alpha 0.05, 30% of pairs differing)
  verdict  p1 better than base (p < 0.05)
  first half (base ran first): 7/32 -> 10/32, fixed 7, broke 4, exact p = 0.55
  second half (p1 ran first): 4/32 -> 17/32, fixed 15, broke 2, exact p = 0.0023
  check: matches the file's own summary (n, successes, fixed, broke, p)
```

Reading it, line by line:

- **Rates and Wilson intervals** describe each arm alone. Overlapping intervals do not mean "no difference": the
  arms share slots, and the paired lines below are the comparison.
- **pairs** are the four outcome types. Only fixed and broke pairs say which policy is better; *both* and *neither*
  pairs carry no information about the difference.
- **change** is (fixed − broke)/n with a paired bootstrap interval (20,000 multinomial draws over the four types,
  seed 0, as in the paper). At n = 64 the bootstrap lives on a 1/64 = 1.6-point grid, so its ends can move by one
  step between methods (the paper's Appendix F, which resampled pairs, reports +9.4 to +39.1 for this run).
- **McNemar** is the exact two-sided test on fixed vs broke. The benchmark accepts an edit with more successes than
  base at p < 0.05 on at least 64 held-out pairs.
- **detects** is the smallest change this many pairs detects with 80% power when 30% of pairs flip by chance, the
  rate measured between identical reruns. A non-significant result below that line means "too small to see", not
  "no effect".
- **halves**: held-out runs alternate the run order by half; a real effect should point the same way in both.
- **check**: the numbers recomputed from the per-slot outcomes agree with the summary the file stores.

**Several tasks or edits.** Correct for multiple tests. Over the controlled study's three submitted edits, with the
four editors who submitted the unedited policy counted as p = 1:

```bash
weightsmith stats study/results/ctrl/heldout/ctrl-{ctb,glasses,mustard}__*.json --family-size 7
# Holm correction over 7 comparisons (4 not given here count as p = 1):
#   p = 0.0037  Holm 0.026  *  ctrl-ctb...     p = 0.12  Holm 0.73  ctrl-glasses...     p = 0.61  Holm 1  ctrl-mustard...
```

**Arms from different runs.** `--ref FILE:ARM --cand FILE:ARM` pairs arms by slot across files, as for the paper's
control (run later on the first 64 held-out slots):

```bash
weightsmith stats --ref study/results/opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json:base \
                  --cand study/results/opus/controls/ctl1-90000-64.results.json:rand
# base 13/64 vs rand 9/64: fixed 6, broke 10, exact p = 0.45; 64 slots in one arm only are left out
```

**Plan the run before spending episodes.**

| Pairs | Smallest detectable change (30% discordant) | Power to see a real 10-point gain |
|---:|---:|---:|
| 16 | 38 points (the exact test cannot reach 80% power at all) | 0.03 |
| 32 | 27 points | 0.10 |
| 64 | 19 points | 0.24 |
| 128 | 14 points | 0.49 |
| 256 | 10 points | 0.81 |

```python
from weightsmith import stats
stats.mde(64, 0.3)                       # 0.19: smallest detectable paired difference
stats.pairs_needed(0.10)                 # 236 pairs to detect a 10-point gain
stats.mcnemar_power(64, 0.20, 0.10)      # 0.24: exact power if 20% of pairs come out fixed and 10% broke
stats.validate_slots(91008, 64)          # raises for a start that is not a multiple of 16
c = stats.compare(base_outcomes, edit_outcomes)   # {slot: bool} each; c.report(), c.to_dict(), c.halves()
```

## 7. Pitfalls

**Noise.** Paired slots fix the scene and the noise seeds, but GPU simulation is not bit-deterministic: two runs of
identical weights flip 28–38% of paired outcomes. The unedited policy scored 14/16 and then 8/16 on the same butter
seeds, and two runs of one edited stacking policy agreed on only 46 of 64 episodes. Rerun base when a comparison
looks surprising, never trust a 16-episode screen, and score held-out runs with at least 64 pairs.

**Slot alignment to 16.** One simulator run steps 16 environments together and slot s lives in group s // 16, which
is what gives it the same seeds in every evaluation. A slot range must start at a multiple of 16 and cover whole
groups, and held-out runs (two halves) need n to be a multiple of 32. The controlled study's first held-out launch
at slot 91000 (16 × 5687.5) stopped before any episode ran; 91008 works. `stats.validate_slots` checks this.

**bf16 rounding.** Weights are stored in bfloat16, which keeps 8 significant bits, and `owsp diff` and `owsp eval`
refuse values that are not exactly representable: save `bf16(W + ΔW)`. Compute ΔW in float64, add it to the weights
and round the sum once. A change smaller than about 0.2–0.4% of a weight's own magnitude rounds back to the original
value, so tiny edits can vanish entirely: count the changed values (`owsp diff`). Rounding also makes a low-rank update
dense (the 0.16% edit changes 7.4% of stored values) and puts a noise floor under the singular values of a stored
change (section 4.5). To stack edits, start from `load(name)` (the candidate's current value), not `base(name)`.

**Moving the decision is not moving success.** Two blind editors built edits that did what they targeted and still
did not help. The flat-box editor's readout made the gripper commit (29/39 holding decisions against 20/42) and
lifted the box 10–19 cm in six episodes, but the box slipped every time: 0/32. The stack-three-cubes editor's
second-round gate cut closes on empty air at the stack from 73 to 5 replans, and success stayed put (17/32 against
16/32 in development, 39/64 against 35/64 held-out, p = 0.58): placing the third cube, which the edit does not
touch, was the bottleneck. The converse also happens: the stacking editor under the companion protocol scored 16/32
against 12/32 while its own check showed the targeted first-chunk shift had not occurred, which is noise. Measure the
targeted decision and the outcome, and believe a gain only when both move.

**In-sample separation.** A direction fitted on the same decisions it is tested on can separate them perfectly and
fail on new ones. Hold out episodes (recipes 5 and 6 split by episode parity).

**First-order predictions are not replays.** Recipes 5 and 6 predict the gripper decision by assuming the change is
the same at all 10 denoising steps. The flow model is nonlinear in its state: the eyeglasses editor's first gate
moved lifts anywhere from −13 to +23 cm on replay. Replay the candidate (section 5) before closed loop.

**Sandbox gotchas.** Recipes must not rebind the prelude's names (`CONFIG` is the workspace configuration that
`save` checks; `stats`, `load`, `DATA` and others are helpers); `tests/test_recipes.py` checks this. The prelude's
`acts(aid)` loads every array of a capture (3 GB for `c2-multi`); read only what you need with
`np.load(f"/work/data/acts/{aid}/acts.npz")[key]`, as the recipes do. Projects with `prompt_overrides` disabled refuse
any capture spec that carries a `"prompt"`, even one equal to the instruction.

**Side effects on other tasks.** An edit can touch every task the policy does. The distilled edit transferred to two
tasks nobody predicted (left of the bowl, but not right). On the butter task it first looked harmful (10/16 against
14/16), but the unedited policy itself scored 8/16 on a rerun, so the comparison could not tell. Run regression
checks on other tasks, with base rerun on the same slots.

## What each recipe produced on cube-then-banana

Runs of 2026-11-07 on the reference GPU host (CPU only, existing episodes and captures; no new captures or
closed-loop runs). Edit recipes wrote into scratch candidates named `cookbook-*`.

| Recipe | Result |
|---|---|
| `01_find_decisions.py` | 22 premature releases in 64 base episodes, including all 21 the paper used |
| `02_prompt_swap.py` | The rewording table of section 2 (Fig. 2a) |
| `03_capture_info.py` | Layouts and sizes of `c2-multi` (3,150 MB), `c1-attn`, `ctb-ph`, `chk-v2mi-f`; all spec items exist |
| `check_spec.py` | All six example specs valid; 0.04–42.5 MB per item |
| `04_edit_distill.py` | `abl-imgl8` and `v2mi-f` rebuilt bit for bit (same `owsp` fingerprints) |
| `05_edit_gated_readout.py` | 1,024-value edit; key 1.00 on targets, ~0 on preservation, holds out of sample; predicted 21/21 flips |
| `06_edit_repurpose_unit.py` | Unit 2437 (the eyeglasses editor's); zero on all silent decisions; predicted 14/21 flips; 3,072 values |
| `07_edit_lexical_unit.py` | Two "then" units; realized change exactly β·dr; leakage at most 7% of it; 4,096 values |
| `08_random_control.py` | Dense: spectrum and norm matched per layer; sparse: same 1,024 values, same norm |
| `09_offline_check.py` | Fig. 2b recomputed: passes the three edits that went on to closed loop, fails five stopped offline |
| `10_paired_outcomes.py` | CSV that `weightsmith stats` turns into 3/32 → 25/32, p = 4.8×10⁻⁷ |

None of the `cookbook-*` edits from recipes 5–7 has been replayed or run in closed loop; their predictions are first
order. Recipe 4's edit is the paper's, whose replay and held-out results are above.
