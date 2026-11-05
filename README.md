# weightsmith: weight-space programming on a robot VLA

A frontier model edits the weights of a robot foundation policy **directly**, with no
gradients, no training and no black-box search, to make it succeed more often in closed-loop
simulation.

This is a replication of **Weight-Space Programming** (Takehi et al., MIT, Nov 2026;
[blog](https://rikiyat.github.io/blog/weight-space-programming/)) on a VLA:

- **Policy:** OpenPI **pi0.5** `pi05_droid_jointpos`, ~3.35B params (PaliGemma VLM + Gemma flow-matching action expert).
- **Simulator:** **RoboLab** manipulation tasks in Isaac Sim.
- **Editors:** **Claude** (Claude Code CLI) or **GPT-6 Astra** (OpenAI Codex CLI). Both get the same brief, tools, budget and seeds, so the two can be compared on identical tasks.

## Method

The editor gets a brief, a budget and exactly one command, `owsp`. It reads weights and
activations, forms a mechanistic hypothesis, writes a closed-form edit, and tests it on
practice seeds. It then submits one candidate, or `base` if nothing survives. An operator
scores that candidate on held-out seeds the editor never sees.

| | Paper (Takehi et al.) | This repo |
|---|---|---|
| network | Qwen3.5-0.8B (LLM) | pi0.5 VLA: 3,353,433,872 params; 730 layer tensors (2.82B values) editable |
| failure | reasoning loops | closed-loop task failures in RoboLab (Isaac Sim, 900-step episodes) |
| editor | frontier model | Claude Opus (Claude Code) or GPT-6 Astra (Codex CLI) |
| allowed | read weights and activations, then edit weights directly | same. Closed-form edits only (contrasts, projections, SVD/PCA, rank-1 edits, unit or head scaling). Edits must be bf16-exact |
| forbidden | gradient descent / training | autodiff, optimizers, fitted regressions, random or evolutionary search |
| evidence | activations of the network | activation capture replaying **exact recorded policy inputs + noise seeds**. Covers VLM prompt and image tokens, action-expert MLP/residual streams, and attention maps |
| practice signal | | **paired** closed-loop evals on 256 practice seeds, under an episode and activation budget |
| final score | held-out, never seen by the editor | operator-only paired base-vs-edit eval on unseen seeds, with exact McNemar test and Wilson 95% CIs |
| headline edit | one residual direction from two MLPs, 0.016% of weights | see [results](docs/results/README.md) |

The rules the editor follows are in each brief ([example](briefs/stack-yellow-red/BRIEF.md)).
Every edit needs a written hypothesis and prediction *before* it is evaluated.

## Architecture

```
 editor machine                                  GPU host  (project root = $WSP_ROOT)
 ┌──────────────────────────────┐                ┌──────────────────────────────────────────────────┐
 │ drivers/claude.sh            │   bin/owsp     │ weightsmith/owsp.py   (budget, validation, paths)    │
 │   Claude Code: Bash(owsp *)  │ ─────────────▶ │  tensors | new | py | diff                       │
 │ drivers/astra.sh             │  ssh or local  │    └─ bwrap numpy sandbox over the bf16 export   │
 │   Codex CLI: rules allow     │                │  acts                                            │
 │   only owsp                  │                │    └─ acts_worker.py: instrumented replay of     │
 │                              │                │       recorded inputs (OpenPI venv)              │
 │ agent dir: BRIEF.md,         │                │  eval                                            │
 │   NOTEBOOK.md, FINAL.md,     │                │    └─ pool.py: batched_server.py (pi0.5,         │
 │   scripts/, specs/, frames/  │                │       hot-swapped edits, per-request seeds)      │
 └──────────────────────────────┘                │       + run_batched.py (Isaac Sim, 16 envs)      │
                                                 │       -> episodes, timelines, obs.npz, video     │
 operator only                                   │  episodes | show | frames | status               │
   weightsmith/heldout.py  ────────────────────────▶ │ heldout/  (not readable through owsp)            │
                                                 └──────────────────────────────────────────────────┘
```

- **Paired evals.** A *slot* fixes an episode's noise-seed stream. Candidates evaluated on the
  same slots are paired against base: same scene, same noise, different weights. GPU physics
  is not bit-deterministic, so pairing is variance reduction, not a guarantee.
- **Seed ranges.** Practice slots are 20000..20255, surveys use 60000+, and held-out slots are ≥ 80000 (90000+ by convention).
- **Exact replay.** Every practice episode records the 224×224 camera inputs, state, seed and
  action chunk at each replan. `owsp acts` replays them through an instrumented forward pass
  and checks parity against the recorded actions.

## Quickstart

**Prerequisites (GPU host, Linux)**
- **Policy stack.** An OpenPI fork with **request-scoped sampling seeds**: `sample_actions(..., noise=)` takes per-request noise, which `weightsmith/batched_server.py` builds from each request's `sampling_seed`. You also need the `pi05_droid_jointpos` checkpoint.
- **Simulator.** RoboLab + Isaac Sim / Isaac Lab in their own venv (`SIM_PYTHON`).
  - The RoboLab Pi0 client must accept `sampling_seed_base` (used by `weightsmith/batched_client.py`).
  - `imageio` and `PIL` are needed there for `owsp frames`.
- **Sandbox.** `bwrap` (bubblewrap), and system `/usr/bin/python3` with numpy.
- **GPU memory.** The reference box was 2× RTX 3090 (24 GB).
  - One GPU runs closed-loop evals: the policy server (XLA memory fraction 0.40) plus one Isaac process with 16 envs.
  - The other GPU runs activation captures (fraction 0.9).
  - A single GPU works too: the per-GPU locks serialize evals and captures.
- **RAM and disk.** `owsp py` analyses of large tensors want tens of GB of RAM; candidates allow up to ~100 GB.
  - The base export needs ~11 GB of disk, and episodes add more.

**Editor machine:** Claude Code and/or Codex CLI, plus key-based ssh to the GPU host (or run everything on the GPU host).

```bash
# 0. both machines: clone, then configure (paths, WSP_HOST, GPUs). See .env.example.
cp .env.example wsp.env && $EDITOR wsp.env

# 1. GPU host, once: layer-split bf16 export of all editable tensors (~11 GB)
#    = weightsmith.lab.openpi_vlm_full.prepare_workspace(CHECKPOINT_DIR, BASE_WORKSPACE, 1.0, include_action=True)
scripts/prepare_workspace.sh

# 2. a project (task config + editor working dir)
scripts/new_project.py --config configs/stack-yellow-red.json --root /data/wsp/projects/syr-claude \
    --brief briefs/stack-yellow-red/BRIEF.md --prior briefs/stack-yellow-red/PRIOR_FINDINGS.md \
    --agent-dir agents/syr-claude

# 3. GPU host: pre-run base practice episodes (uncharged; the editor starts by reading them)
WSP_ROOT=/data/wsp/projects/syr-claude scripts/base_practice.sh 64

# 4. smoke-test the editor's tool surface from the editor machine
WSP_ROOT=/data/wsp/projects/syr-claude bin/owsp status

# 5. launch an editor (resumes until FINAL.md; `touch agents/syr-claude/STOP` to end)
WSP_ROOT=/data/wsp/projects/syr-claude drivers/claude.sh agents/syr-claude   # or drivers/astra.sh
WSP_ROOT=/data/wsp/projects/syr-claude scripts/progress.sh agents/syr-claude

# 6. GPU host, operator only: held-out paired eval of the submitted candidate
#    (--n a multiple of 2 x 16; base and candidate alternate order by half)
WSP_ROOT=/data/wsp/projects/syr-claude python3 weightsmith/heldout.py --cand NAME --slot-start 90000 --n 192
```

To compare Claude and Astra on the same task with matched budgets and seeds, see
[docs/COMPARING_EDITORS.md](docs/COMPARING_EDITORS.md). The Astra driver has a few flags that
still need verifying on the first run; they are listed there.

## Results so far

Base policy, practice sets (operator pre-runs):

| task | base success |
|---|---|
| banana-then-cube, "put the banana then the cube in the bowl" | 99/128 (77.3%) |
| cube-then-banana, "Put the cube then the banana in the bowl" | 9/64 (14.1%) |
| stack yellow on red (capability task, fixed instruction) | survey 6/16; practice set running |

There is a 7-task base survey [here](docs/results/README.md).

- **Claude Opus, run 1 (banana-then-cube): honest negative.**
  - Found the mechanism behind premature cube releases: an *elliptical-instruction mis-parse*. In stage 2 the policy partly still targets the banana and drops the held cube as the "wrong object". An explicit two-clause prompt cuts these releases from 28/29 to 9/29 on replayed inputs.
  - It built three closed-form edits and rejected each on offline replays (not selective, no effect, or correlates rather than causes). The estimated ceiling of a perfect fix was about +3 pp.
  - It **submitted no edit**: [FINAL.md](docs/results/claude-opus-run1-banana-then-cube/FINAL.md).
- **Claude Opus, run 2 (cube-then-banana):** stopped before any weight candidate.
  - It confirmed the same mis-parse, and a diagnostic *prompt* override gave 22/32 vs base 3/32 on the same slots. This is not a weight edit.
  - See its [notebook](docs/results/claude-opus-run2-cube-then-banana/NOTEBOOK.md).
- **Capability project (stack yellow on red):** in progress.
  - Prompt overrides are disabled, so the fix has to be motor or perceptual and live in the weights.
- **GPT-6 Astra:** not run yet.
- No held-out evaluation has been run yet, because no editor has submitted a weight edit.

## Layout

```
weightsmith/            harness (runs on the GPU host)
  owsp.py           the editor's tool surface (CLI)
  pool.py rollout.py batched_server.py batched_client.py run_batched.py   closed-loop eval engine
  acts_worker.py    activation capture on recorded inputs
  heldout.py        operator-only held-out eval
  survey.py         operator base-policy task survey
  config.py         all paths / GPUs / ports (env vars or wsp.env)
  lab/              minimal vendored pieces of an earlier project (see NOTICE)
bin/owsp            editor-side wrapper (local or ssh); the only thing on the editor's PATH
drivers/            claude.sh, astra.sh (+ codex/owsp.rules)
scripts/            operator tools: prepare_workspace.sh, new_project.py, base_practice.sh, progress.sh, killmatch.sh
briefs/             editor briefs per task (+ PRIOR_FINDINGS.md hints)
configs/            task configs: task, instruction, budget, practice slots, envs
docs/               COMPARING_EDITORS.md, results/
```

No install step: each module runs under the environment that needs it. `owsp.py` uses system
python3 with numpy, the server and capture worker use the OpenPI venv, and `run_batched.py`
uses the Isaac Sim venv. They find the package through the repo path.

## License

MIT (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). OpenPI, RoboLab, Isaac Sim and the
checkpoint are separate dependencies under their own licenses.
