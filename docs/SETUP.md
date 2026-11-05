# Setup: the exact stack behind the weightsmith results

Everything is pinned. A fresh machine reproduces the reference box bit-for-bit at the source level. The pi0.5 checkpoint is verified by sha256.

| Component | Source | Pin | Local changes |
|---|---|---|---|
| OpenPI (policy) | `github.com/xuningy/openpi` | `aa64205` | `third_party/patches/openpi/` (1 patch: request-scoped `sampling_seed`) |
| RoboLab (tasks, sim eval) | `github.com/NVlabs/RoboLab` | `0aef241` (v0.2.1) | `third_party/patches/robolab/` (6 patches) |
| Isaac Sim / Isaac Lab | via `envs/robolab-sim` | Isaac Sim 5.0, Isaac Lab 2.2, torch 2.7.1+cu118 | `envs/robolab-sim/uv.lock` |
| Checkpoint | `gs://openpi-assets-simeval/pi05_droid_jointpos` | sha256 of all 26 files | `third_party/pi05_droid_jointpos.sha256` |

The 6 RoboLab patches cover: seeded per-request sampling and batched inference in the pi0 client, `--environment-seed`, matched spatial-steering tasks, predicate-oracle prompt control (unused here), and a task-discovery fix. Applied on top of v0.2.1, they reproduce the reference checkout's source tree exactly; this was verified by a recursive diff that found 0 differing files.

## Requirements
- Linux, NVIDIA GPU(s) with ≥ 24 GB each, driver ≥ 535. The sim env uses CUDA 11.8 torch wheels for that reason.
- `git`, `git-lfs`, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, and `bubblewrap` (`bwrap`, used by the weight sandbox). The sandbox also needs `numpy` for the system `python3`.
- About 60 GB of disk: RoboLab assets ~10 GB, checkpoint 12 GB, weight workspace 11 GB, plus the envs.

## Steps
```bash
git clone https://github.com/adeeb10abbas/weightsmith && cd weightsmith
scripts/setup_deps.sh                 # clone + patch OpenPI/RoboLab, uv sync both envs, download + verify the checkpoint
cp .env.example wsp.env               # then paste the lines setup_deps.sh printed; set GPU indices
scripts/prepare_workspace.sh          # export the editable layer-split weights (~11 GB, one time)
scripts/smoke_test.sh                 # 16 closed-loop Isaac episodes + one activation capture (~10-15 min)
```

`smoke_test.sh` runs in its own scratch project (`projects/smoke`). On the reference box, unedited pi0.5 stacks the yellow block on the red block in about 40% of episodes (28/64 on practice slots), so expect roughly 3–10/16.

## GPU layout
- **Evals.** One batched policy server plus one Isaac process with 16 parallel envs need about 22 GB on one 24 GB GPU (`WSP_SERVER_GPU = WSP_SIM_GPU`).
- **Activation captures.** These load a second copy of the model (~9 GB), so on a second GPU set `WSP_ACTS_GPU` to it.
- **Single GPU.** Set all three to 0; the per-GPU locks then serialize evals and captures.
- **Do not run several Isaac processes on one GPU.** They time-slice catastrophically (> 10× slower). Use the 16-env single process.

## Starting an editor on a task
```bash
python3 scripts/new_project.py ...         # or copy configs/<task>.json to $WSP_ROOT/config.json
WSP_ROOT=$PWD/projects/stack-yellow-red-astra scripts/base_practice.sh 64
WSP_ROOT=$PWD/projects/stack-yellow-red-astra drivers/astra.sh  agent-dirs/stack-astra    # or drivers/claude.sh
```
Give both editors the same brief (`briefs/stack-yellow-red/`), budget and practice slots. Only the operator runs `weightsmith/heldout.py`; see `docs/COMPARING_EDITORS.md`.
