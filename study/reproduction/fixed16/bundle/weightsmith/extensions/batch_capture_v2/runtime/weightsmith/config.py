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
import re
import shlex
from collections.abc import MutableMapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent


def parse_env(text: str, source: str = "environment file") -> dict[str, str]:
    """Parse literal KEY=VALUE assignments, never shell commands or substitutions.

    Blank values, optional ``export``, single/double quotes, and whitespace-prefixed
    inline comments are supported. A # inside quotes or within a word is literal.
    Values stay unexpanded so paths can be resolved by the machine using them.
    """
    values = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line, count=1)
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        prefix = f"{source}:{number}"
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"{prefix}: expected KEY=VALUE")
        if value.startswith(("'", '"')):
            quote = value[0]
            chars = []
            index = 1
            while index < len(value):
                char = value[index]
                if char == quote:
                    trailing = value[index + 1:].strip()
                    if trailing and not trailing.startswith("#"):
                        raise ValueError(f"{prefix}: unexpected text after quoted value")
                    value = "".join(chars)
                    break
                if quote == '"' and char == "\\" and index + 1 < len(value) and value[index + 1] in ('"', "\\"):
                    index += 1
                    char = value[index]
                chars.append(char)
                index += 1
            else:
                raise ValueError(f"{prefix}: unterminated quoted value")
        else:
            value = re.split(r"(?:^|\s)#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    return values


def load_env_file(path: str | Path | None = None,
                  environ: MutableMapping[str, str] | None = None) -> None:
    """Load wsp.env without overriding even explicitly empty environment values."""
    environ = os.environ if environ is None else environ
    path = Path(path or environ.get("WSP_ENV_FILE") or (REPO / "wsp.env")).expanduser()
    if not path.is_file():
        return
    for key, value in parse_env(path.read_text(), str(path)).items():
        environ.setdefault(key, value)


def remote_shell_path(value: str) -> str:
    """Quote a path while allowing only a leading home reference on the SSH host."""
    if value in ("~", "$HOME", "${HOME}"):
        return '"$HOME"'
    for prefix in ("~/", "$HOME/", "${HOME}/"):
        if value.startswith(prefix):
            return '"$HOME"/' + shlex.quote(value[len(prefix):])
    return shlex.quote(value)


def ssh_options(*, scp: bool = False) -> list[str]:
    options = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=60",
               "-o", "ServerAliveCountMax=10"]
    port = os.environ.get("WSP_SSH_PORT", "")
    if port:
        if not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("WSP_SSH_PORT must be an integer from 1 to 65535")
        options += ["-P" if scp else "-p", port]
    return options


load_env_file()


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
POOL_PORT = _int("WSP_POOL_PORT", 8300)   # websocket port of the eval-pool policy server
SERVER_MEM_FRACTION = os.environ.get("WSP_SERVER_MEM_FRACTION", "0.40")  # server shares its GPU with Isaac
ACTS_MEM_FRACTION = os.environ.get("WSP_ACTS_MEM_FRACTION", ".9")
