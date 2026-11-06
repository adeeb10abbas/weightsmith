"""Hugging Face export: folders and cards built from the real result files; upload only on request."""
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from test_edits import make_candidate

from weightsmith import edits

REPO = Path(__file__).resolve().parents[1]
HELDOUT = REPO / "study" / "results" / "opus" / "heldout"
spec = importlib.util.spec_from_file_location("export_to_hub", REPO / "integrations" / "hub" / "export_to_hub.py")
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)


def test_dataset_folder_from_the_published_results(tmp_path, capsys):
    out = tmp_path / "results"
    assert hub.main(["dataset", "--out", str(out)]) == 0
    assert "dry run: nothing was uploaded" in capsys.readouterr().out
    rows = json.loads((REPO / "study/results/opus/comparisons.json").read_text())["rows"]
    with open(out / "comparisons.csv") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    with open(out / "heldout.csv") as f:
        heldout = {r["file"]: r for r in csv.DictReader(f)}
    assert set(heldout) == {p.name for p in HELDOUT.glob("*.json")}
    for path in HELDOUT.glob("*.json"):
        assert (out / "heldout" / path.name).read_bytes() == path.read_bytes()  # unmodified
        summary = json.loads(path.read_text())["summary"]
        assert int(heldout[path.name]["cand_success"]) == summary["cand_success"]
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: mit\n") and "config_name: heldout" in card
    v2mi = json.loads(next(HELDOUT.glob("cube-then-banana__v2mi-f-*.json")).read_text())["summary"]
    assert f"| {v2mi['base_success']}/128 | {v2mi['cand_success']}/128 |" in card
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["EVIDENCE.md"]["sha256"] == hub._sha256(REPO / "study/results/opus/EVIDENCE.md")


def test_edit_folder_and_card(tmp_path, synthetic_workspace):
    cand, _ = make_candidate(synthetic_workspace, name="v2mi-f")
    edit_file = edits.from_candidate(cand, rank="auto").save(tmp_path / "v2mi-f.safetensors")
    heldout = next(HELDOUT.glob("cube-then-banana__v2mi-f-*.json"))
    out = tmp_path / "model"
    files = hub.build_edit(edit_file, out, [heldout])
    names = {str(f.relative_to(out)) for f in files}
    assert {"README.md", "v2mi-f.safetensors", "edit.json", "manifest.json", f"heldout/{heldout.name}"} == names
    assert edits.load(out / "v2mi-f.safetensors").exact
    card = (out / "README.md").read_text()
    assert "fingerprint `" + edits.fingerprint(cand / "edits") + "`" in card
    assert "bit for bit" in card and "| 27/128 | 106/128 | 81 / 2 | 7.2×10⁻²² |" in card
    with pytest.raises(SystemExit, match="not empty"):
        hub.build_edit(edit_file, out)


def test_push_is_explicit_and_private_by_default(tmp_path, monkeypatch, capsys):
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            calls.append(("create_repo", repo_id, kw))
            return f"https://huggingface.co/datasets/{repo_id}"

        def upload_folder(self, **kw):
            calls.append(("upload_folder", kw))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    out = tmp_path / "results"
    assert hub.main(["dataset", "--out", str(out), "--push", "me/weightsmith-results"]) == 0
    assert calls[0] == ("create_repo", "me/weightsmith-results",
                        {"repo_type": "dataset", "private": True, "exist_ok": True})
    assert calls[1][1]["folder_path"] == str(out) and calls[1][1]["repo_type"] == "dataset"
    assert "uploaded to https://huggingface.co/datasets/me/weightsmith-results" in capsys.readouterr().out


def test_p_values_print_like_the_report():
    assert hub._p(7.210946989931019e-22) == "7.2×10⁻²²"
    assert hub._p(0.0006795482549932785) == "6.8×10⁻⁴"
    assert hub._p(0.0123) == "0.012"
