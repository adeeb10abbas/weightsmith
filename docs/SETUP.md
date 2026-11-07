# Setup

This page sets up a **GPU host** that runs the simulator, the policy and the weight sandbox, and optionally a
**laptop** that drives it over ssh. When `scripts/smoke_test.sh` passes, go to
[TUTORIAL.md](TUTORIAL.md) and make your first weight edit.

Everything is pinned, so a fresh machine reproduces the reference box at the source level. The π0.5 checkpoint
is verified by sha256.

| Component | Source | Pin | Local changes |
|---|---|---|---|
| OpenPI (policy) | `github.com/xuningy/openpi` | `aa64205` | `third_party/patches/openpi/` (1 patch: request-scoped `sampling_seed`) |
| RoboLab (tasks, sim eval) | `github.com/NVlabs/RoboLab` | `0aef241` (v0.2.1) | `third_party/patches/robolab/` (6 patches) |
| Isaac Sim / Isaac Lab | via `envs/robolab-sim` | Isaac Sim 5.0, Isaac Lab 2.2, torch 2.7.1+cu118 | `envs/robolab-sim/uv.lock` |
| Checkpoint | `gs://openpi-assets-simeval/pi05_droid_jointpos` | sha256 of all 26 files | `third_party/pi05_droid_jointpos.sha256` |

The 6 RoboLab patches add seeded per-request sampling and batched inference to the pi0 client, an
`--environment-seed` flag, matched spatial-steering tasks, predicate-oracle prompt control (unused here) and a
task-discovery fix. Applied to v0.2.1, they reproduce the reference checkout's source tree exactly: a recursive
diff found 0 differing files.

## Requirements

- Linux with an NVIDIA GPU of at least 24 GB (one eval needs about 22 GB) and driver 535 or newer.
- `git`, `git-lfs`, [`uv`](https://docs.astral.sh/uv/), `ffmpeg` and `bubblewrap` (`bwrap`, the weight sandbox).
- `numpy` for the system `/usr/bin/python3`: the sandbox runs that interpreter. On Ubuntu:
  `sudo apt install git git-lfs ffmpeg bubblewrap python3-numpy && git lfs install`.
- Python 3.10 or newer for the `weightsmith` command.
- About 60 GB of disk: RoboLab assets ~10 GB, checkpoint 12 GB, weight export 11 GB, plus the two environments.

`scripts/setup_deps.sh --check` lists everything that is missing in one go.

## Install (once, on the GPU host)

```bash
git clone https://github.com/adeeb10abbas/weightsmith && cd weightsmith
python3 -m venv .venv && source .venv/bin/activate   # any Python >= 3.10 environment works
pip install -e '.[mcp]'                              # the weightsmith command (and numpy for owsp)
scripts/setup_deps.sh                                # clone + patch OpenPI and RoboLab, install both envs, download + verify the checkpoint
cp .env.example wsp.env                              # your settings; see "wsp.env" below
scripts/setup_deps.sh --print-env >> wsp.env         # the six dependency paths setup_deps.sh just installed
scripts/prepare_workspace.sh                         # the editable layer-split export of the weights (~11 GB, one time)
scripts/smoke_test.sh                                # 16 closed-loop episodes and one activation capture
```

`setup_deps.sh` puts OpenPI and RoboLab under `third_party/` and the Isaac Sim environment under
`envs/robolab-sim/.venv`, all git-ignored. The downloads are large (RoboLab's git-lfs assets, Isaac Sim and the
12 GB checkpoint), so the first run is mostly a matter of bandwidth. `prepare_workspace.sh` writes
`BASE_WORKSPACE` and refuses to touch a workspace that is not empty.

### Already have OpenPI, RoboLab and the checkpoint?

Skip `setup_deps.sh` and point `wsp.env` at your copies. They must be the pinned, patched versions above: the
eval pool relies on the request-scoped sampling seeds and the batched client. If someone on the same host already
ran `prepare_workspace.sh`, reuse their `BASE_WORKSPACE`; candidates only ever read it.

```bash
OPENPI_DIR=/opt/openpi
OPENPI_PYTHON=/opt/openpi/.venv/bin/python
ROBOLAB_DIR=/opt/robolab
SIM_PYTHON=/opt/robolab-env/.venv/bin/python
CHECKPOINT_DIR=/data/openpi-cache/openpi-assets-simeval/pi05_droid_jointpos
BASE_WORKSPACE=/data/weightsmith-workspace/pi05_droid_jointpos-full
```

## wsp.env

`wsp.env` sits at the repo root, and [.env.example](../.env.example) documents every key. Three rules:

- **One `KEY=VALUE` per line, comments on their own lines.** Text after a value is part of the value, so
  `ROBOLAB_DIR=~/robolab  # my checkout` makes the path `~/robolab  # my checkout`.
- **The first line for a key wins.** Appending a key that is already set further up changes nothing. Every key in
  `.env.example` is commented out, so appending works on a fresh copy.
- **The environment wins over the file**, so `WSP_ROOT=... bin/owsp status` works for a single command.

### GPUs

| Setup | Settings |
|---|---|
| One 24 GB GPU | defaults (everything on GPU 0; per-GPU locks queue evals and captures) |
| Two GPUs (reference box: 2× RTX 3090) | `WSP_SERVER_GPU=1`, `WSP_SIM_GPU=1`, `WSP_ACTS_GPU=0`, `WSP_EVAL_EXTRA_GPUS=0` |

- **Evals.** One batched policy server plus one Isaac process with 16 parallel environments need about 22 GB of
  one GPU (`WSP_SERVER_GPU` = `WSP_SIM_GPU`). `WSP_EVAL_EXTRA_GPUS` lets an eval borrow idle GPUs, each running
  its own server and Isaac process; two lanes roughly halve the wall time.
- **Activation captures** load a second copy of the model (~9 GB), so on two GPUs give them the other one.
- **Never run several Isaac processes on one GPU.** They time-slice badly (more than 10× slower); the locks
  prevent it as long as every job on the host uses the same `WSP_LOCK_DIR`.

### Several checkouts or users on one host

Every checkout reads its own `wsp.env`, but the GPUs are shared. Give all of them the same `WSP_LOCK_DIR`.
With different lock directories two evals can start on one GPU and collide on the policy server's port
(`WSP_POOL_PORT`, default 8300). Sharing `WSP_JAX_CACHE` as well skips recompiling the policy in each checkout.

## Check the install

```bash
scripts/smoke_test.sh
```

It runs in its own scratch project (`projects/smoke`) and checks the three things the tutorial needs: the weight
sandbox (`owsp py`), closed-loop episodes (`owsp eval`) and activation capture (`owsp acts`). It stops with
`SMOKE TEST FAILED: ...` and a pointer to the logs when one of them does not work. On the reference box the
unedited policy stacks the yellow block on the red one in about 40% of episodes (28/64 on practice slots), so
expect roughly 3–10 of 16.

Where things are written:

| Path | What |
|---|---|
| `$WSP_ROOT/` | one project: `config.json`, `cands/` (candidates), `data/episodes/`, `data/acts/` (captures), `state/budget.json`, `heldout/` |
| `logs/pool-*` | per eval: policy-server and simulator logs (`server-l0.log`, `sim-l0a0.log`), `rows.json` |
| `$ROBOLAB_DIR/output/` | raw episode outputs and videos |

## Drive the GPU host from a laptop

```bash
pip install -e '.[mcp]'                       # on the laptop, in this repo
cat >> wsp.env <<'EOF'
WSP_HOST=my-gpu-box
WSP_REMOTE_REPO=weightsmith
EOF
export WSP_ROOT=/home/me/weightsmith/projects/ctb   # the project root ON THE GPU HOST
bin/owsp status
```

`bin/owsp` then runs every command over ssh (key-based login, no password prompt). `WSP_REMOTE_REPO` is the
repo path on the host, absolute or relative to the remote home directory. On the host, owsp runs under the ssh
login's `python3` (`WSP_REMOTE_PYTHON`), which needs numpy. `weightsmith distill run` and `weightsmith mcp` go
through `bin/owsp`, so they work from the laptop too.

## Start an editor on a task

```bash
python3 scripts/new_project.py ...         # or copy configs/<task>.json to $WSP_ROOT/config.json
WSP_ROOT=$PWD/projects/stack-yellow-red-astra scripts/base_practice.sh 64
WSP_ROOT=$PWD/projects/stack-yellow-red-astra drivers/astra.sh  agent-dirs/stack-astra    # or drivers/claude.sh
```

Give both editors the same brief (`briefs/stack-yellow-red/`), budget and practice slots. Only the operator runs
`weightsmith/heldout.py`; see [COMPARING_EDITORS.md](COMPARING_EDITORS.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ssh: ... hostname contains invalid characters`, or a path in an error ends in `# ...` | An inline comment in `wsp.env` (older copies of `.env.example` had them). Move each comment onto its own line. |
| A setting in `wsp.env` has no effect | The key appears twice and the first one wins, or the variable is also set in your shell (`env \| grep WSP_`). |
| `owsp: no project config at .../config.json` | `WSP_ROOT` is unset or wrong. It must name a project folder holding `config.json`, on the GPU host. |
| `owsp: error: unrecognized arguments: script.py` | `weightsmith/owsp.py py` and `acts` read stdin. Use `bin/owsp py CAND script.py` and `bin/owsp acts spec.json`, which pass the file for you. |
| `owsp py` prints `[stderr] ... [exit 1]` | Your script raised inside the sandbox; owsp itself exits 0, so read the output. `ModuleNotFoundError: numpy` means `/usr/bin/python3` lacks numpy. |
| An eval or capture prints nothing for minutes | It is waiting for a GPU lock held by another job. Check with `ps aux \| grep -E 'owsp.py\|heldout.py'`. Waiters are not served in order. |
| `instruction/task mismatch` | The project's `config.json` names a different task or instruction than the simulator reports. No budget is charged; fix the config. |
| `prompt overrides are disabled in this project` | The config sets `"prompt_overrides": false` (the smoke and benchmark configs do). Wording experiments need a project without it, such as `configs/cube-then-banana.json`. |
| `--slot-start must be a multiple of envs = 16` | Held-out slots run in groups of 16. Use e.g. 91008 instead of 91000. |
