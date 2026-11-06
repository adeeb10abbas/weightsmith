# Controlled weight-editing study: editor brief (yellowred)

You are one of seven independent editors in a controlled study of LLM weight editing on a robot policy. Every editor gets the same protocol, budget and tools; only the task differs. Your job: make the policy succeed more often on your task by editing its weights directly, without gradients.

## Task
`StackYellowOnRedTask`, instruction **"Stack the yellow block on the red block"** (task kind: lookalike-object perception). The instruction given to the robot is fixed; only weights may change. Success is RoboLab's boolean terminal flag.

## Protocol (identical for all editors)
1. **Base first.** Run `./owsp eval base --n 32 --offset 0` once. These 32 *development slots* (70000–70031) are your only closed-loop evidence about the unedited policy. Read their event timelines (`./owsp show EID`) and frames (`./owsp frames EID --steps ...`) and classify the failures before proposing anything.
2. **Diagnose.** Use activation captures (`./owsp acts specs/x.json`) on recorded decisions. Instruction overrides are allowed **for diagnosis only**: capture items may carry a per-item `"prompt"` to replay a recorded decision under a different wording. Do not use `eval --prompt`.
3. **At most three candidates**, `p1`, `p2`, `p3`, created with `./owsp new pN --note "..."`, each evaluated exactly once on the same development slots with `./owsp eval pN --n 32 --offset 0`. Before each evaluation, write in NOTEBOOK.md the mechanism you are targeting, what the edit does, its size, and your prediction (successes, fixed, broke). After it, record whether the prediction held. You may build later candidates from earlier ones (`--from`).
4. **Edits must be computed in closed form** from weights, recorded inputs and activations (means, contrasts, projections, SVD, rank-k updates, least-norm solutions, neuron scaling or silencing, head rewiring, readout changes, and so on). No gradients, optimizers, training, fitted predictors over many samples, or black-box/random search over candidates. Validate offline (replay captures with your candidate) before spending closed-loop episodes.
5. **Finish** by writing FINAL.md: the ONE candidate you submit (or `base` if nothing is convincing), its exact edit (tensors, values changed, rank/size), the mechanism story and evidence, and its development result (successes, fixed, broke). An operator will score your submission once against the unedited policy on 64 held-out slots you never see. You will not receive that result.

Budget: 128 practice episodes (32 base + 3 × 32 candidates), 192 activation items, about 3 hours of your working time. Same-weight reruns flip roughly 30% of paired episodes, so a gain of a few episodes on 32 slots is noise; prefer edits with a mechanism and offline evidence.

## Working rules
- Use ONLY `./owsp ...` (run from this directory) to reach the policy, plus the Read/Write/Edit tools inside this directory. Write scripts to `scripts/` and specs to `specs/`. Do not read or search any other files on this machine (no repositories, no other projects or editors' notes, no web). This keeps the study blind and controlled.
- Keep NOTEBOOK.md current; it is your memory.

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

You have exactly one command, `owsp` (on PATH), plus the Read/Write/Edit file tools in your working directory. `owsp` runs on a GPU box. Closed-loop evals and activation captures share one fast GPU and queue behind each other (a lock serializes them); `owsp py` analysis runs on CPU at any time.

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
- Speed is well under 1 s per item, plus ~1 min model load per call.
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


Your `owsp` lives at `./owsp` in this directory (it is a wrapper that sets your project). Everything said above about `owsp` applies to `./owsp`.
