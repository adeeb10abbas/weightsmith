"""Single source of machine-specific configuration (stdlib only).

Every path, port and GPU index the harness needs is read from environment variables.
Values may also be put in a KEY=VALUE file: `$WSP_ENV_FILE`, or `wsp.env` at the repo root.
Variables already set in the environment win over the file. See `.env.example`.

This module is imported from three different Python environments (system python3 for
`owsp`, the OpenPI venv for the policy server / activation worker, and the Isaac Sim venv
for the RoboLab client), so it must stay dependency-free.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent


def _load_env_file() -> None:
    path = Path(os.environ.get("WSP_ENV_FILE") or (REPO / "wsp.env"))
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


def _path(name: str, default: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(os.environ.get(name) or str(default))))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


# --- per-project root: config.json, cands/, data/, state/, notes/, heldout/ ----------------
# (OWSP_ROOT is accepted as a legacy alias.)
PROJECT_ROOT = _path("WSP_ROOT", os.environ.get("OWSP_ROOT") or (REPO / "projects" / "default"))

# --- shared, cross-project locations --------------------------------------------------------
LOCK_DIR = _path("WSP_LOCK_DIR", "~/.cache/weightsmith/locks")       # GPU flock files (global across projects)
LOG_DIR = _path("WSP_LOG_DIR", REPO / "logs")                     # pool / server / sim logs
SURVEY_DIR = _path("WSP_SURVEY_DIR", REPO / "survey")             # operator base-policy surveys
JAX_CACHE = _path("WSP_JAX_CACHE", "~/.cache/weightsmith/jax")         # persistent XLA compilation cache

# --- external dependencies -----------------------------------------------------------------
OPENPI_DIR = _path("OPENPI_DIR", "~/openpi")                       # OpenPI fork (request-scoped sampling seeds)
OPENPI_PY = str(_path("OPENPI_PYTHON", OPENPI_DIR / ".venv/bin/python"))
OPENPI_DATA_HOME = _path("OPENPI_DATA_HOME", "~/.cache/openpi")    # OpenPI download cache (tokenizer)
ROBOLAB = _path("ROBOLAB_DIR", "~/robolab")                        # RoboLab checkout (cwd of the sim)
SIM_PY = _path("SIM_PYTHON", ROBOLAB / ".venv/bin/python")         # Isaac Sim / RoboLab python (also renders frames)
CHECKPOINT = _path("CHECKPOINT_DIR", OPENPI_DATA_HOME / "openpi-assets/checkpoints/pi05_droid_jointpos")
BASE_WS = _path("BASE_WORKSPACE", "~/weightsmith-workspace/pi05_droid_jointpos-full")  # layer-split bf16 export
VK_ICD = os.environ.get("WSP_VK_ICD", "/usr/share/vulkan/icd.d/nvidia_icd.json")

# --- GPUs and ports ------------------------------------------------------------------------
SERVER_GPU = _int("WSP_SERVER_GPU", 0)    # batched policy server for closed-loop evals
SIM_GPU = _int("WSP_SIM_GPU", 0)          # Isaac Sim process (16 envs)
ACTS_GPU = _int("WSP_ACTS_GPU", 0)        # activation capture worker
EVAL_EXTRA_GPUS = tuple(int(x) for x in (os.environ.get("WSP_EVAL_EXTRA_GPUS") or "").split(",") if x.strip())
# ^ extra GPUs an eval may borrow (each hosts its own server + Isaac lane) when they are idle at eval start
POOL_PORT = _int("WSP_POOL_PORT", 8300)   # websocket port of the eval-pool policy server
SERVER_MEM_FRACTION = os.environ.get("WSP_SERVER_MEM_FRACTION", "0.40")  # server shares its GPU with Isaac
ACTS_MEM_FRACTION = os.environ.get("WSP_ACTS_MEM_FRACTION", ".9")
