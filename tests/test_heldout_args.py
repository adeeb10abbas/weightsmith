"""weightsmith/heldout.py rejects bad arguments before it creates the run folder.

A held-out run folder is never overwritten, so a run that dies on a bad --slot-start, --n or candidate
name after creating its folder can never be repeated under the same tag. No GPU or simulator is needed:
every case here must stop before the eval pool starts.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "cands" / "v1" / "edits").mkdir(parents=True)
    (root / "cands" / "v1" / "workspace.json").write_text("{}")
    (root / "config.json").write_text(json.dumps({
        "task": "FakeTask", "instruction": "do it", "practice_seed_base": 20000, "practice_slots": 256,
        "limits": {"practice_episodes": 10, "acts_items": 10, "acts_items_per_call": 4},
        "rendering_type": "performance", "envs": 16}))
    return root


def heldout(root, *args):
    env = dict(os.environ, WSP_ROOT=str(root), WSP_ENV_FILE=str(root / "no-such-wsp.env"),
               WSP_LOCK_DIR=str(root / "locks"), WSP_LOG_DIR=str(root / "logs"))
    return subprocess.run([sys.executable, str(REPO / "weightsmith" / "heldout.py"), *args], env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("args, message", [
    (["--cand", "v1", "--slot-start", "91000", "--n", "64"], "multiple of envs = 16"),   # 91000 = 16 * 5687.5
    (["--cand", "v1", "--slot-start", "90000", "--n", "48"], "multiple of 2*envs = 32"),
    (["--cand", "v1", "--slot-start", "90000", "--n", "0"], "multiple of 2*envs = 32"),
    (["--cand", "nope", "--slot-start", "90000", "--n", "64"], "no candidate 'nope'"),
    (["--cand", "v1", "--slot-start", "20000", "--n", "64"], "held-out slots must be >= 80000"),
])
def test_bad_arguments_stop_before_the_run_folder_exists(project, args, message):
    p = heldout(project, *args)
    assert p.returncode != 0 and message in p.stderr, p.stdout + p.stderr
    assert not (project / "heldout").exists()


def test_the_suggested_slot_start_is_a_whole_group(project):
    p = heldout(project, "--cand", "v1", "--slot-start", "91000", "--n", "64")
    assert "e.g. 91008" in p.stderr and 91008 % 16 == 0
