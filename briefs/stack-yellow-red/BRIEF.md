# Weight-space programming on pi0.5: editor brief

You are the editor in a weight-space programming experiment. It follows Takehi et al., *Weight-Space Programming* (MIT, Nov 2026). In that paper a frontier model read a network's weights and its activations, reasoned about what the parts compute, and then **edited the weights directly, with no gradient descent**, to fix a failure. Its headline example: tracing Qwen3.5-0.8B's reasoning loops to one residual direction written by two MLPs, then removing it from 0.016% of the weights.

Your target is a robot policy, **OpenPI pi0.5 (DROID joint-position checkpoint)**, running closed-loop in the RoboLab Isaac simulator.

## Task

`StackYellowOnRedTask`, instruction **"Stack the yellow block on the red block"** (fixed; it never changes). This is a single pick-and-stack manipulation task with a 900-step limit (60 s). The scene holds a yellow block and a red block on a table. Success is RoboLab's boolean terminal flag for the stacked condition. Partial scores do not count.

**Goal:** make pi0.5 *better at the manipulation itself*, so it stacks more reliably, by editing its weights. This is a capability repair, not a language problem: the instruction is fixed and prompt overrides are disabled in this project. Look for the computation that produces the failure, for example grasp closure timing, approach height/alignment over the red block, release timing, or post-release motion that knocks the stack over, and fix that computation in the weights. A separate operator will evaluate one final candidate against the unedited policy on held-out noise seeds you never see. Those seeds come from the same scene, and success there is the only thing that counts. Overfitting to practice seeds will show up.

## The network

pi0.5 is a PaliGemma VLM prefix plus a separate Gemma "action expert" that does flow matching.

- **Inputs.** Two 224×224 camera images (`base` over-shoulder and `wrist`) go through the SigLIP vision tower into 256 image tokens each. The prompt text and the discretized robot state are tokenized into the language prefix.
- **VLM:** `language.blocks.00..17`. Width 2048, MLP hidden 16384 (GeGLU), 8 query heads, 1 KV head, head dim 256.
  - `language.blocks.LL.mlp.gating_einsum` is `[2, 2048, 16384]`: `[0]` is the gate (GELU applied), `[1]` is the up projection.
  - `language.blocks.LL.mlp.linear` is `[16384, 2048]` and is the down projection. **Row r is the residual direction written by hidden unit r.**
  - `attn.q_einsum.w` is `[8,2048,256]`, `attn.kv_einsum.w` is `[2(k,v),1,2048,256]`, `attn.attn_vec_einsum.w` is `[8,256,2048]`.
- **Vision tower:** `vision.blocks.00..26` (SigLIP, width 1152). It is editable but expensive to reason about.
- **Action expert:** `action.blocks.00..17`. Width 1024, MLP hidden 4096, same attention layout with `_1` suffixes. Its 15 action tokens attend to the VLM's KV cache plus each other.
  - `action.blocks.LL.mlp_1.gating_einsum` is `[2,1024,4096]`.
  - `action.blocks.LL.mlp_1.linear` is `[4096,1024]`, the down projection. Row r is the direction written by hidden unit r.
  - Norms are **adaptive** (adaRMS): `pre_attention_norm_1.Dense_0` / `pre_ffw_norm_1.Dense_0` map the flow-time embedding `[1024]` to scale, shift and gate (3×1024).
- **Flow matching.** Noise x₁ ~ N(0, I) of shape `[15, 32]` is integrated over **10 Euler steps** (t = 1.0, 0.9, …, 0.1), each with x ← x − 0.1·v. Velocity v = `action.action_out_proj` applied to the expert's final hidden state (`[1024,32]` kernel). The first 8 of the 32 dims are used:
  - 7 joint-position targets (normalized; the output transform unnormalizes them).
  - 1 gripper dimension, **thresholded at 0.5**: above 0.5 closes, below opens.
  - `action_in_proj` embeds x_t, and `time_mlp_in`/`time_mlp_out` embed t.
- **Control loop.** The policy is queried every **15 sim steps**. The 15-action chunk is executed open-loop, so a 900-step episode is about 60 policy calls.
- There are about 3.35B parameters in all. Every tensor above is exported as layer-split bf16 arrays you can read and write.

## Your tools

You have exactly one command, `owsp` (on PATH), plus the Read/Write/Edit file tools in your working directory. `owsp` runs on a GPU box. Closed-loop evals run on GPU1, with 16 simulated robots in parallel. Activation captures run on GPU0, which is slow (clock-stuck) but adequate. Because they use different GPUs, you can run an `owsp eval` in the background (Bash `run_in_background`) and do `owsp acts`/`owsp py` analysis while it runs.

| command | what it does |
|---|---|
| `owsp status` | Budget used and limits, candidates, activation captures. |
| `owsp tensors [REGEX]` | Names, shapes and stats of editable tensors. |
| `owsp new NAME [--from OTHER] [--note TEXT]` | Create a candidate. It starts as the unedited policy, or as a copy of OTHER's edits. |
| `owsp py NAME script.py` | Run your Python file in a CPU numpy sandbox (no network, no GPU, about 100 GB RAM, 30 min limit). Pass `base` instead of NAME for read-only analysis. Helpers are below. |
| `owsp diff NAME` | Validate a candidate and report exactly which values it changes. |
| `owsp eval NAME --n N [--offset K]` | Closed-loop practice rollouts on practice slots K..K+N−1, out of 256 practice slots (N and K are multiples of 16). The 16 slots of a group run simultaneously, and a group takes up to ~8 min, since it lasts as long as its slowest episode, max 900 steps. Add ~2 min startup. So N=64 takes ~25–35 min. It prints success, a **paired comparison against base on the same slots**, and per-episode events. Every episode records the exact policy inputs at each replan, so you can capture activations on it afterwards. |
| `owsp episodes [--cand NAME] [-v]` | Success tallies per candidate and fingerprint. |
| `owsp show EID` | One episode's summary with its full event timeline: grabs, drops, objects in the bowl, wrong-object grabs, gripper closures, all with sim steps. |
| `owsp frames EID --steps 100,200,300` | Renders frames from the episode's sensor video (both policy cameras side by side; frame index ≈ sim step) to a local JPG and prints its path. Open it with the Read tool to see it. |
| `owsp acts spec.json` | GPU activation capture on exact recorded policy inputs. Format below. |

**Sandbox helpers in `owsp py`** (numpy is imported as `np`):
- Weights:
  - `load(name)`: current candidate value, as float32 holding exact bf16 values.
  - `base(name)`: the original value.
  - `save(name, array)`: writes the edit. Values **must** be bf16-representable, so wrap them in `bf16(x)`.
  - `drop_edit(name)`, `saved()`, `stats(name)`, `NAMES`.
- Data:
  - `episodes(cand=None, success=None)`: list of practice episode records.
  - `obs(eid)`: exact policy inputs at every replan (`base_img`, `wrist_img`, `joint`, `gripper`, `seed`, `step`, plus the `actions` chunk the policy output).
  - `acts(aid)`: returns `(index, arrays)` for a capture.
- Persistence: `/work/scratch` is per-candidate and writable, `/work/notes` is shared and writable. Nothing else persists, and Python variables do not survive between calls.

**`owsp acts` spec** (a JSON file):
```json
{"id": "a-grasp-vs-miss", "cand": "base",
 "prompt_tokens": [0,1,...,23],
 "items": [{"episode": "base.base.s20000e3", "step": 240}, {"episode": "...", "k": 17}],
 "action_layers": [0,1,2,...,17], "vlm_layers": [10, 15, 17],
 "denoise_steps": [0, 5, 9], "action_tokens": "mean"}
```
- Each item is a recorded replan, chosen either by replan index `k` or by the nearest replan at or before sim `step`.
- The worker replays the exact recorded inputs and noise seed, and checks parity with the recorded actions for `base`. Differences around 0.01 are bf16/batching noise.
- The language prefix is `<bos>Task: <the instruction>, State: <discretized joint/gripper tokens>;\nAction:`. Prompt slot i is the i-th prompt token. With the default slots 0..23, the task words sit at slots 3–11 and the state tokens start at slot 13. `index["prompt_token_pieces_item0"]` shows the decoded pieces, and you can request up to 64 slots with `"prompt_tokens"`.
- Speed is about 1–4 s per item on GPU0, plus ~30 s model load per call.
- **Image tokens and attention.** Add `"image_tokens": true` (needs `vlm_layers`) to get `vlm.img_mlp_hidden` / `vlm.img_resid_out` with shape `[N, Lv, cam, 17, F]`. cam 0 is the base over-shoulder camera and cam 1 the wrist camera. Index 0 is the camera mean, and 1..16 is a 4×4 spatial grid, row-major from the top of the 224×224 image.
- Add `"action_attention": true` to get:
  - `action.attn_mass` `[N, S, La, 8 heads, 5]`: attention mass from the action tokens onto [base cam, wrist cam, padding cam, prompt+state, action tokens].
  - `action.attn_map` `[N, S, La, cam, 16, 16]`: per-patch attention.
- These are how visual/scene information reaches the action expert; earlier captures did not include them.
- **Outputs** in `acts(aid)[1]`:
  - `action.mlp_hidden` `[N,S,La,T,4096]` (the input to `mlp_1.linear`) and `action.resid_out` `[N,S,La,T,1024]`.
  - `action.head_input` `[N,S,15,1024]`.
  - `vlm.mlp_hidden` `[N,Lv,P,16384]` and `vlm.resid_out` at the prompt/state token slots.
  - `actions` `[N,15,8]`.
- With `"action_tokens": "mean"`, T = 1 (mean over the 15 action tokens). Pass a list such as `[0,7,14]` to keep individual tokens.
- Use up to 96 items per call. The index records each item's sim step, gripper state and seed.

## Budget

- **1200 practice episodes** and 6000 activation items in total. Spend them on hypotheses, not lottery tickets.
- 64 base-policy practice episodes have already been run for you, on slots `s20000`.. (`owsp episodes --cand base -v`). Look at them first.
- In an operator survey, base pi0.5 succeeded on 6/16 episodes (37.5%). Most failures pick up the yellow block but end without a valid stack (placement, alignment, release or knock-over), and a few never secure the yellow block. Find out exactly which physical sub-skill breaks, and why, from the episodes, frames and activations.
- A slot fixes the policy's noise-seed stream, and the scene layout is the same in every slot. So evals of different candidates on the same slots are **paired**: the same noise goes into different weights. GPU physics is not bit-deterministic, so base itself does not reproduce exactly on a slot. Treat pairing as variance reduction, not as a guarantee.
- Rendering uses Isaac's `performance` preset, which is identical for every candidate.

## Rules (from the paper's protocol)

1. **No gradients, no training.**
   - No autodiff, optimizers, fine-tuning or regression fitted over many samples to produce weights.
   - No black-box search: no random perturbations evaluated en masse, no evolutionary search.
   - Allowed: anything you can *reason* about and compute in closed form from weights and activations. That includes means and contrasts, projections, SVD/PCA of a weight matrix or of activations, rank-1 edits along a direction you identified, neuron scaling, silencing or sign flips, rewiring specific heads, and so on.
2. **Every edit needs a mechanistic hypothesis** written down *before* you evaluate it, covering what the edited part computes, why the edit should change behavior, and what you predict. After the eval, record whether the prediction held.
3. Keep `NOTEBOOK.md` in your working directory up to date. It is your memory across context compaction. Record:
   - findings, with episode IDs and acts IDs;
   - the candidates and their results;
   - your current best candidate.
4. You may never see held-out results, and you must not try to.
5. Practice success is noisy, roughly ±12 percentage points at n=64. Before you call something a win, confirm it on a larger n, and check that it doesn't break episodes base already solves (the "broke" count).

## Finish

When you are done, or the budget is nearly spent, write `FINAL.md` in your working directory containing:
- the chosen candidate name;
- exactly what it changes: tensors, number of values, percentage of the model;
- the mechanism story and the evidence behind it;
- practice results.

Then stop. Choosing `base` (no edit) is allowed if nothing survives scrutiny; say so honestly.

Read `PRIOR_FINDINGS.md` first. It holds general lessons from earlier editors on *other* tasks; treat them as hints. Then read the base practice episodes: tally the failure modes and look at frames from the failures (`owsp frames`, both cameras; the wrist camera shows the grasp and stack geometry). Form hypotheses about which *motor or perceptual* computation goes wrong, then go find it in the network: action-expert MLPs and attention, the action head, and the VLM image-token pathways. The instruction is fixed, so do not look for a fix in prompt handling.
