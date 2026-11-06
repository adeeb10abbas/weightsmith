"""The `weightsmith` command, packaging metadata, and import hygiene."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_edits import make_candidate

from weightsmith import cli, edits
from weightsmith.lab.openpi_head import _file_name

REPO = Path(__file__).resolve().parents[1]


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_export_info_apply_round_trip(tmp_path, synthetic_workspace, capsys):
    cand, targets = make_candidate(synthetic_workspace)
    out = tmp_path / "toy.safetensors"
    code, text, err = run(capsys, "edit", "export", str(cand), "-o", str(out), "--rank", "auto")
    assert code == 0 and out.is_file() and "encoding language.blocks.08.mlp.linear" in err
    assert "exact     yes" in text and f"wrote {out}" in text

    code, text, _ = run(capsys, "edit", "info", str(out))
    assert code == 0 and "fingerprint " + edits.fingerprint(cand / "edits") in text
    code, text, _ = run(capsys, "edit", "info", str(out), "--json")
    meta = json.loads(text)
    assert meta["format"] == "weightsmith.edit" and set(meta["tensors"]) == set(targets)

    new = tmp_path / "cands" / "toy-copy"
    code, text, _ = run(capsys, "edit", "apply", str(out), "--workspace", str(synthetic_workspace), "--out", str(new))
    assert code == 0 and "identical to the source candidate" in text and "owsp diff toy-copy" in text
    assert edits.fingerprint(new / "edits") == edits.fingerprint(cand / "edits")

    names = list(targets) + ["language.blocks.09.mlp.linear"]
    base = {n: np.load(synthetic_workspace / "base" / _file_name(n)) for n in names}
    with open(tmp_path / "base.npz", "wb") as f:
        np.savez(f, **base)
    code, text, _ = run(capsys, "edit", "apply", str(out), "--arrays", str(tmp_path / "base.npz"),
                        "--out", str(tmp_path / "edited.npz"))
    assert code == 0 and "2 of 3 arrays edited" in text
    with np.load(tmp_path / "edited.npz") as z:
        for n in names:
            expected = targets.get(n, base[n])
            assert np.array_equal(z[n].view(np.uint32), expected.view(np.uint32))


def test_lossy_export_and_npz_container(tmp_path, synthetic_workspace, capsys):
    cand, _ = make_candidate(synthetic_workspace)
    out = tmp_path / "toy.npz"
    code, text, _ = run(capsys, "edit", "export", str(cand), "-o", str(out), "--rank", "1", "--lossy", "-q")
    assert code == 0 and out.is_file() and "lowrank r=1" in text
    assert edits.load(out).tensors["language.blocks.08.mlp.linear"].encoding == "lowrank"


def test_errors_are_reported_not_raised(tmp_path, capsys):
    code, _, err = run(capsys, "edit", "export", str(tmp_path), "-o", str(tmp_path / "x.safetensors"))
    assert code == 1 and "weightsmith: error:" in err and "not an owsp candidate" in err
    with pytest.raises(SystemExit):
        cli.main(["edit", "export", str(tmp_path), "-o", "x", "--rank", "0"])


def test_mcp_subcommand_forwards_its_options(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["mcp", "--help"])
    assert e.value.code == 0 and "--root" in capsys.readouterr().out


def test_distill_subcommand_is_experimental_and_dry_runs(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0 and "EXPERIMENTAL" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["distill", "--help"])
    help_text = capsys.readouterr().out
    assert e.value.code == 0 and help_text.startswith("usage: weightsmith distill")
    assert "Experimental" in help_text and "dry-run" in help_text and "bit for bit" in help_text

    example = REPO / "examples" / "cube_then_banana" / "distill.json"
    code, script, _ = run(capsys, "distill", "render", str(example), "--script", "build")
    assert code == 0 and script.startswith("# distill build: mode=image")
    compile(script, "build.py", "exec")

    def no_commands(*args, **kwargs):
        raise AssertionError("a dry run must not run commands")

    monkeypatch.setattr(subprocess, "run", no_commands)
    code, out, _ = run(capsys, "distill", "run", str(example), "--dry-run", "--workdir", str(tmp_path))
    assert code == 0 and "[dry-run]" in out and "heldout.py" in out and (tmp_path / "build.py").is_file()


def test_version_flag():
    out = subprocess.run([sys.executable, "-m", "weightsmith.cli", "--version"], capture_output=True, text=True,
                         cwd=REPO, check=True)
    assert out.stdout.startswith("weightsmith ")


def test_toolkit_imports_without_gpu_or_optional_dependencies():
    code = ("import sys, weightsmith, weightsmith.cli, weightsmith.edits, weightsmith.mcp_server\n"
            "heavy = ('jax', 'flax', 'torch', 'tensorflow', 'openpi', 'isaaclab', 'isaacsim', 'robolab', 'cv2',\n"
            "         'mcp', 'pydantic', 'safetensors', 'huggingface_hub', 'weightsmith.config')\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy or m in heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    assert out.stdout.strip() == "[]"


def test_package_metadata():
    tomllib = pytest.importorskip("tomllib")
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    project = meta["project"]
    assert project["name"] == "weightsmith" and project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.10" and project["dependencies"] == ["numpy>=1.22"]
    assert project["scripts"] == {"weightsmith": "weightsmith.cli:main"}
    assert set(project["optional-dependencies"]) == {"mcp", "hub", "dev"}
