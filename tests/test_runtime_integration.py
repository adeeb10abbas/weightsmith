"""The toolkit against the real runtime: bin/owsp and weightsmith/owsp.py in local mode on a synthetic
project. Covers the owsp commands that need no GPU or sandbox (status, tensors, episodes, show, new,
diff), and checks that an exported and re-applied edit passes owsp's own validation unchanged."""
import json

import numpy as np

from weightsmith import edits
from weightsmith.lab.openpi_head import _file_name
from weightsmith.mcp_server import Owsp


def save_like_the_sandbox(cand, name, value):
    """What `save(name, value)` in `owsp py` writes: the bf16 values as float32 .npy."""
    np.save(cand / "edits" / _file_name(name), np.asarray(value).astype(np.float32), allow_pickle=False)


def test_read_only_commands(synthetic_project):
    o = Owsp(env=synthetic_project.env)
    status = json.loads(o.status())
    assert status["instruction"] == "Put the cube then the banana in the bowl" and status["candidates"] == []
    listing = o.tensors(r"^language\.blocks\.0[89]\.mlp")
    assert "language.blocks.08.mlp.linear" in listing and "[2 tensors]" in listing
    assert "base" in o.episodes() and "1/2 (50.0%)" in o.episodes("base")
    shown = json.loads(o.show("base.base.s20000"))
    assert shown["success"] is False and shown["events"][0]["name"] == "OBJECT_GRABBED_SUCCESS"


def test_exported_edit_passes_owsp_validation_after_a_round_trip(synthetic_project, tmp_path):
    o = Owsp(env=synthetic_project.env)
    assert "created candidate v1" in o.new("v1", note="rank-1 MLP edit")
    cand = synthetic_project.root / "cands" / "v1"
    rng = np.random.default_rng(3)
    w = np.load(cand / "base" / _file_name("language.blocks.08.mlp.linear"))
    save_like_the_sandbox(cand, "language.blocks.08.mlp.linear",
                          edits.bf16_round(w + 0.01 * np.outer(rng.normal(size=64), rng.normal(size=32))))
    s = np.load(cand / "base" / _file_name("action.final_norm.scale"))
    s[5] = edits.bf16_round(np.float32(0.75))
    save_like_the_sandbox(cand, "action.final_norm.scale", s)
    original = json.loads(o.diff("v1"))

    edit = edits.from_candidate(cand, rank="auto")
    for name, report in original["edited_tensors"].items():
        assert edit.tensors[name].info["changed"] == report["changed"]
    assert edit.changed_values == original["changed_elements"]

    path = edit.save(tmp_path / "v1.safetensors")
    result = edits.apply_to_workspace(edits.load(path), synthetic_project.workspace,
                                      synthetic_project.root / "cands" / "v1-copy")
    assert result["fingerprint_matches"]
    copy = json.loads(o.diff("v1-copy"))
    assert copy == original
    assert json.loads(o.status())["candidates"] == ["v1", "v1-copy"]
