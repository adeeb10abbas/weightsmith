"""The edit zoo in edits/: the committed file passes edits/verify.sh, and a damaged copy fails it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "edits" / "verify.sh"
COMMITTED = REPO / "edits" / "ctrl-ctb-p1.safetensors"


def _verify(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHON": sys.executable}
    return subprocess.run(["bash", str(VERIFY), *args], capture_output=True, text=True, env=env, timeout=120)


def test_committed_edit_matches_its_checksum_and_heldout_record():
    out = _verify(str(COMMITTED))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "matches the published ctrl-ctb-p1.safetensors" in out.stdout
    assert "fingerprint 4ca8023b0592b79f" in out.stdout
    assert "ctrl-ctb__p1-RubiksCubeThenBananaTask-91008-64.json" in out.stdout
    assert "11/64 -> 27/64" in out.stdout


def test_a_published_name_resolves_to_the_committed_file():
    out = _verify("ctrl-ctb-p1")
    assert out.returncode == 0, out.stdout + out.stderr


def test_damaged_copy_fails(tmp_path):
    data = bytearray(COMMITTED.read_bytes())
    data[-1] ^= 0x01  # one bit of the low-rank payload
    damaged = tmp_path / "ctrl-ctb-p1.safetensors"
    damaged.write_bytes(bytes(data))
    out = _verify(str(damaged))
    assert out.returncode == 1, out.stdout + out.stderr
    assert "is not in edits/SHA256SUMS" in out.stdout


def test_missing_file_fails_without_download(tmp_path):
    out = _verify(str(tmp_path / "v2mi-f.safetensors"), "--dir", str(tmp_path))
    assert out.returncode == 1
    assert "not found; pass --download" in out.stdout
