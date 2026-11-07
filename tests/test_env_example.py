""".env.example works when copied verbatim, uncommented line by line, or extended with setup_deps.sh --print-env.

Both readers of wsp.env (weightsmith/config.py and bin/owsp) keep everything after `=` as the value, so an inline
`# comment` silently becomes part of a path or of the ssh host name. That is how a verbatim copy used to break.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / ".env.example"
KEY_LINE = re.compile(r"^# ([A-Z][A-Z0-9_]*)=(.*)$")


def test_every_line_is_a_comment_and_every_key_can_be_uncommented():
    keys = []
    for line in EXAMPLE.read_text().splitlines():
        assert line == "" or line.startswith("#"), f"active line in .env.example: {line!r}"
        m = KEY_LINE.match(line)
        if m:
            keys.append(m.group(1))
            assert "#" not in m.group(2) and m.group(2) == m.group(2).strip(), f"inline comment or padding: {line!r}"
    for key in ("WSP_HOST", "WSP_REMOTE_REPO", "OPENPI_DIR", "OPENPI_PYTHON", "ROBOLAB_DIR", "SIM_PYTHON",
                "CHECKPOINT_DIR", "BASE_WORKSPACE", "WSP_SERVER_GPU", "WSP_SIM_GPU", "WSP_ACTS_GPU", "WSP_LOCK_DIR"):
        assert key in keys, f"{key} is not documented in .env.example"


def _config(env_file):
    code = ("import json, os; from weightsmith import config as C; print(json.dumps({'openpi': str(C.OPENPI_DIR), "
            "'robolab': str(C.ROBOLAB), 'ws': str(C.BASE_WS), 'gpu': C.SERVER_GPU, "
            "'host': os.environ.get('WSP_HOST')}))")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WSP_", "OPENPI", "ROBOLAB", "SIM_", "CHECKPOINT",
                                                                     "BASE_WORKSPACE"))}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=dict(env, WSP_ENV_FILE=str(env_file)),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_a_verbatim_copy_and_appended_print_env_lines_are_read_as_intended(tmp_path):
    copy = tmp_path / "wsp.env"
    copy.write_text(EXAMPLE.read_text())
    c = _config(copy)
    assert "#" not in c["openpi"] + c["robolab"] + c["ws"] and c["host"] is None and c["gpu"] == 0
    lines = subprocess.run(["bash", str(REPO / "scripts" / "setup_deps.sh"), "--print-env"], capture_output=True,
                           text=True, check=True, env=dict(os.environ, HOME=str(tmp_path))).stdout
    assert len(lines.splitlines()) == 6
    copy.write_text(EXAMPLE.read_text() + lines + "WSP_SERVER_GPU=1\n")
    c = _config(copy)
    assert c["openpi"] == str(REPO / "third_party" / "openpi") and c["robolab"] == str(REPO / "third_party" / "RoboLab")
    assert c["ws"] == str(tmp_path / "weightsmith-workspace" / "pi05_droid_jointpos-full") and c["gpu"] == 1


def test_bin_owsp_reads_a_verbatim_copy_as_local(tmp_path):
    copy = tmp_path / "wsp.env"
    copy.write_text(EXAMPLE.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("WSP_")}
    p = subprocess.run([str(REPO / "bin" / "owsp"), "status"], capture_output=True, text=True,
                       env=dict(env, WSP_ENV_FILE=str(copy), WSP_ROOT=str(tmp_path / "no-project")))
    # runs owsp.py locally (no ssh to a host named after a comment) and reports the missing project
    assert "no project config" in p.stdout + p.stderr and "ssh" not in p.stdout + p.stderr
