# How weightsmith works

The method, the runtime architecture and the prerequisites. For the protocol, see [BENCHMARK.md](../BENCHMARK.md); for installation, [SETUP.md](SETUP.md); for MCP clients and edit files, [INTEGRATIONS.md](INTEGRATIONS.md).

## Method

The editor gets a brief, a budget and exactly one command, `owsp`. It reads weights and activations, forms a mechanistic hypothesis, writes a closed-form edit, and tests it on practice seeds. It then submits one candidate, or `base` if nothing survives. An operator scores that candidate on held-out seeds the editor never sees.

This replicates **Weight-Space Programming** (Takehi et al., MIT, Nov 2026; [blog](https://rikiyat.github.io/blog/weight-space-programming/)) on a vision-language-action policy:

| | Paper (Takehi et al.) | This repo |
|---|---|---|
| Network | Qwen3.5-0.8B (LLM) | π0.5 VLA: 3,353,433,872 parameters. 730 layer tensors (2.82B values) are editable. |
| Failure | Reasoning loops | Closed-loop task failures in RoboLab (Isaac Sim) |
| Editor | Frontier model | Claude Opus (Claude Code) or GPT-6 Astra (Codex CLI); any MCP client through `weightsmith mcp` |
| Allowed | Read weights and activations, then edit weights directly | The same. Closed-form edits only: contrasts, projections, SVD/PCA, rank-1 edits, unit or head scaling. Edits must be bf16-exact. |
| Forbidden | Gradient descent and training | Autodiff, optimizers, fitted regressions, random or evolutionary search |
| Evidence | Activations of the network | Activation capture that replays **exact recorded policy inputs and noise seeds**. It covers VLM prompt and image tokens, action-expert MLP and residual streams, and attention maps. |
| Practice signal | — | **Paired** closed-loop evals on 256 practice slots, under an episode and activation budget |
| Final score | Held-out, never seen by the editor | Operator-only paired base-vs-edit eval on unseen slots, with an exact McNemar test and Wilson 95% intervals |
| Headline edit | One residual direction from two MLPs, 0.016% of weights | Closed-form instruction distillation: a multi-key update of 17 VLM MLPs; see the [leaderboard](../LEADERBOARD.md) |

The rules the editor follows are in each brief ([example](../briefs/stack-yellow-red/BRIEF.md)). Every edit needs a hypothesis and prediction written down *before* it is evaluated.

## Architecture

```
 editor machine                                  GPU host  (project root = $WSP_ROOT)
 ┌──────────────────────────────┐                ┌───────────────────────────────────────────────────┐
 │ drivers/claude.sh            │   bin/owsp     │ weightsmith/owsp.py  (budget, validation, paths)  │
 │   Claude Code: Bash(owsp *)  │ ─────────────▶ │  tensors | new | py | diff                        │
 │ drivers/astra.sh             │  ssh or local  │    └─ bwrap numpy sandbox over the bf16 export    │
 │   Codex CLI: rules allow     │                │  acts                                             │
 │   only owsp                  │                │    └─ acts_worker.py: instrumented replay of      │
 │ or any MCP client:           │                │       recorded inputs (OpenPI venv)               │
 │   weightsmith mcp            │                │  eval                                             │
 │   (typed tools over owsp)    │                │    └─ pool.py: batched_server.py (pi0.5,          │
 │                              │                │       hot-swapped edits, per-request seeds)       │
 │ agent dir: BRIEF.md,         │                │       + run_batched.py (Isaac Sim, 16 envs)       │
 │   NOTEBOOK.md, FINAL.md      │                │       -> episodes, timelines, obs.npz, video      │
 └──────────────────────────────┘                │  episodes | show | frames | status                │
                                                 │                                                   │
 operator only                                   │ cands/<name>/edits/*.npy <-> weightsmith edit     │
   weightsmith/heldout.py ─────────────────────▶ │ heldout/  (not readable through owsp)             │
                                                 └───────────────────────────────────────────────────┘
```

- **One tool surface.** Every route to the policy goes through `owsp`, which enforces the project's budget, validates candidates (bf16-exact values, allowed tensors, change budget) and keeps held-out data out of reach. The headless drivers put only `bin/owsp` on the editor's `PATH`; the MCP server calls the same `bin/owsp` with validated arguments and never runs a shell.
- **Paired evals.** A *slot* fixes an episode's noise-seed stream. Candidates evaluated on the same slots are paired against base: the same scene and noise, with different weights. GPU physics is not bit-deterministic, so pairing reduces variance but does not guarantee identical episodes.
- **Slot ranges.** Practice slots are 20000–20255. Surveys use 60000 and up. Held-out slots are ≥ 80000 (90000+ by convention). See [BENCHMARK.md §4](../BENCHMARK.md#4-slots-and-seeds).
- **Exact replay.** Every practice episode records the 224×224 camera inputs, the robot state, the seed and the action chunk at each replan. `owsp acts` replays them through an instrumented forward pass and checks parity against the recorded actions.
- **Parallel lanes.** `WSP_EVAL_EXTRA_GPUS` lets an eval borrow idle GPUs. Each extra GPU runs its own policy server and Isaac process, and slots keep their seeds whichever lane runs them (`tests/test_pool_lanes.py`).
- **Portable edits.** A candidate is a directory of full edited tensors next to the base export. `weightsmith edit export` encodes it against that base as one self-describing file, and `weightsmith edit apply` turns the file back into a candidate with the same `owsp` fingerprint ([format](INTEGRATIONS.md#portable-edit-files)).

## Prerequisites (GPU host, Linux)

- **Policy stack.** You need an OpenPI fork with **request-scoped sampling seeds**: `sample_actions(..., noise=)` takes per-request noise, which `weightsmith/batched_server.py` builds from each request's `sampling_seed`. You also need the `pi05_droid_jointpos` checkpoint. `scripts/setup_deps.sh` installs both.
- **Simulator.** RoboLab and Isaac Sim / Isaac Lab run in their own venv (`SIM_PYTHON`). The RoboLab Pi0 client must accept `sampling_seed_base` (used by `weightsmith/batched_client.py`), and `owsp frames` needs `imageio` and `PIL` there.
- **Sandbox.** `bwrap` (bubblewrap), plus the system `/usr/bin/python3` with numpy.
- **GPU memory.** The reference box had 2× RTX 3090 (24 GB). One GPU runs closed-loop evals: the policy server (XLA memory fraction 0.40) plus one Isaac process with 16 envs. The other runs activation captures (fraction 0.9). A single GPU works too, because the per-GPU locks serialize evals and captures.
- **RAM and disk.** `owsp py` analyses of large tensors want tens of GB of RAM, and candidates allow up to ~100 GB. The base export needs ~11 GB of disk, and episodes add more.

**Editor machine:** Claude Code, the Codex CLI or another MCP client, plus key-based ssh to the GPU host. You can also run everything on the GPU host.

The runtime has no install step. Each module runs under the environment that needs it: `owsp.py` uses the system python3 with numpy, the server and capture worker use the OpenPI venv, and `run_batched.py` uses the Isaac Sim venv. They find the package through the repository path. The editor-side toolkit (`weightsmith edit`, `weightsmith mcp`) is the only part installed with pip.

## Code layout

```
weightsmith/
  owsp.py           the editor's tool surface (CLI)
  pool.py rollout.py batched_server.py batched_client.py run_batched.py   closed-loop eval engine
  acts_worker.py    activation capture on recorded inputs
  heldout.py        operator-only held-out eval
  survey.py         operator base-policy task survey
  config.py         all paths / GPUs / ports (env vars or wsp.env)
  edits.py          portable edit files (format, export, apply); _safetensors.py: numpy-only reader/writer
  mcp_server.py     the MCP server over bin/owsp
  distill.py        closed-form instruction distillation (Experimental): math, owsp py scripts, pipeline
  cli.py            the `weightsmith` command: edit export | info | apply, mcp, distill
  lab/              minimal vendored pieces of an earlier project (see NOTICE)
bin/owsp            editor-side wrapper (local or ssh); the only thing on the editor's PATH
drivers/            claude.sh, astra.sh (+ codex/owsp.rules)
integrations/       mcp/server.py (MCP launcher without install), hub/export_to_hub.py (Hugging Face folders)
examples/           worked configs, e.g. cube_then_banana/distill.json for `weightsmith distill`
scripts/            operator tools: setup_deps.sh, prepare_workspace.sh, new_project.py, base_practice.sh,
                    smoke_test.sh, progress.sh, killmatch.sh
tests/              CPU-only tests of the toolkit and the eval lanes
docker/             CPU image of the editor-side toolkit
```
