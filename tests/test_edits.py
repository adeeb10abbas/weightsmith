"""Portable edit format: exact round trips, low-rank error reports, containers and application targets."""
import json

import numpy as np
import pytest
from conftest import TENSORS

from weightsmith import _safetensors, edits
from weightsmith.lab.openpi_head import _effective_bfloat16, _file_name

RNG = np.random.default_rng(0)


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def same(a, b):
    """Bitwise equality: also distinguishes +0.0 from -0.0."""
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def base_and_lowrank_target(m=96, n=40, k=3, scale=2e-4, seed=1):
    rng = np.random.default_rng(seed)
    w = edits.bf16_round(rng.normal(0, 0.05, (m, n)))
    delta = (rng.normal(0, 1, (m, k)) @ rng.normal(0, 1, (k, n))) * scale
    return w, edits.bf16_round(w.astype(np.float64) + delta)


# ------------------------------------------------------------------------------ bf16 and hashing
def test_bf16_round_matches_the_export_rule():
    x = np.concatenate([RNG.normal(0, 3, 10_000).astype(np.float32),
                        np.array([0.0, -0.0, 1.0, -2.5, 1e-40, -1e-40, 3.0e38, 1 + 2**-8, 1 + 3 * 2**-8], np.float32)])
    assert same(edits.bf16_round(x), _effective_bfloat16(x))
    assert same(edits.bf16_round(edits.bf16_round(x)), edits.bf16_round(x))


def test_tensor_sha256_is_layout_independent():
    a = RNG.normal(size=(7, 5)).astype(np.float32)
    assert edits.tensor_sha256(a) == edits.tensor_sha256(np.asfortranarray(a)) == edits.tensor_sha256(a.astype(">f4"))
    assert edits.tensor_sha256(a) != edits.tensor_sha256(a.T)


# ------------------------------------------------------------------------------ exact round trips
@pytest.mark.parametrize("rank", [None, 1, 3, 8, "auto"])
def test_exact_round_trip_for_every_encoding(rank):
    w, t = base_and_lowrank_target()
    te = edits.encode_tensor("w", w, t, rank=rank)
    assert te.info["exact"] and te.info["values_off"] == 0
    assert same(te.reconstruct(w), t)
    assert te.info["changed"] == int(np.count_nonzero(t.astype(np.float64) - w))


def test_hard_values_round_trip_exactly():
    """Sign of zero, huge exponent gaps and subnormals: float32 deltas cannot carry these; fix-ups do."""
    w = edits.bf16_round(RNG.normal(0, 1, (16, 8)))
    t = w.copy()
    t[0, 0] = -0.0
    t[0, 1] = np.float32(2.0**-120)
    t[1, 2] = edits.bf16_round(np.float32(3.0e38))
    t[2, 3] = edits.bf16_round(np.float32(1e-40))
    w[3, 4], t[3, 4] = np.float32(1.0), edits.bf16_round(np.float32(2.0**-30))
    for rank in (None, 2):
        te = edits.encode_tensor("w", w, t, rank=rank)
        assert same(te.reconstruct(w), t)
    assert edits.encode_tensor("w", w, t).info["fixups"] >= 1


def test_encoding_choice_follows_size():
    w = edits.bf16_round(RNG.normal(0, 0.05, (64, 32)))
    few = w.copy()
    few[3, 4] = edits.bf16_round(few[3, 4] + 0.25)
    assert edits.encode_tensor("w", w, few).encoding == "sparse"
    many = edits.bf16_round(w * 1.5 + 0.01)
    assert edits.encode_tensor("w", w, many).encoding == "dense"
    w2, lowrank = base_and_lowrank_target(m=512, n=128, k=4, scale=0.01)
    auto = edits.encode_tensor("w", w2, lowrank, rank="auto")
    dense = edits.encode_tensor("w", w2, lowrank)
    assert auto.encoding == "lowrank" and auto.nbytes < dense.nbytes
    assert same(auto.reconstruct(w2), lowrank)


def test_one_dimensional_tensors_ignore_rank():
    w = edits.bf16_round(RNG.normal(0, 1, 32))
    t = edits.bf16_round(w + 0.5)
    te = edits.encode_tensor("scale", w, t, rank=4)
    assert te.encoding in ("dense", "sparse") and same(te.reconstruct(w), t)


def test_unchanged_tensor_encodes_to_nothing():
    w = edits.bf16_round(RNG.normal(size=(8, 8)))
    assert edits.encode_tensor("w", w, w.copy()) is None
    assert edits.from_arrays({"w": w}, {"w": w.copy()}).tensors == {}


# ------------------------------------------------------------------------------ low-rank error report
def test_lowrank_truncation_error_matches_svd():
    w, t = base_and_lowrank_target(k=4, scale=0.02)
    d = t.astype(np.float64) - w.astype(np.float64)
    s = np.linalg.svd(d, compute_uv=False)
    errors = []
    for r in (1, 2, 4, 8):
        te = edits.encode_tensor("w", w, t, rank=r)
        expected = np.sqrt(np.sum(s[r:] ** 2) / np.sum(s**2))
        assert te.info["rel_fro_error"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert te.info["rank"] == r and te.info["matrix_shape"] == [96, 40]
        errors.append(te.info["rel_fro_error"])
    assert errors == sorted(errors, reverse=True)
    assert errors[2] < 1e-2  # a rank-4 delta is captured up to bf16 rounding noise


def test_lossy_lowrank_reports_what_is_off():
    w, t = base_and_lowrank_target(scale=2e-4)
    te = edits.encode_tensor("w", w, t, rank=1, lossy=True)
    rec = te.reconstruct(w)
    off = int(np.count_nonzero(bits(rec) != bits(t)))
    assert off > 0 and not te.info["exact"]
    assert te.info["values_off"] == off and te.info["fixups"] == 0 and "fix_index" not in te.arrays
    assert te.info["max_abs_error"] == pytest.approx(float(np.max(np.abs(rec.astype(np.float64) - t))))
    exact = edits.encode_tensor("w", w, t, rank=1)
    assert exact.info["fixups"] == off and same(exact.reconstruct(w), t)


def test_matricizes_higher_rank_tensors():
    w = edits.bf16_round(RNG.normal(0, 0.05, (4, 32, 8)))
    outer = np.einsum("i,j,k->ijk", RNG.normal(size=4), RNG.normal(size=32), RNG.normal(size=8))
    t = edits.bf16_round(w + 0.01 * outer)
    te = edits.encode_tensor("q", w, t, rank=2)
    assert te.info["matrix_shape"] == [128, 8] and same(te.reconstruct(w), t)


# ------------------------------------------------------------------------------ validation
def test_rejects_invalid_edits():
    w = edits.bf16_round(RNG.normal(size=(4, 4)))
    with pytest.raises(edits.EditError, match="bfloat16"):
        edits.encode_tensor("w", w, w + np.float32(1e-6))
    with pytest.raises(edits.EditError, match="shape"):
        edits.encode_tensor("w", w, w[:2])
    bad = w.copy()
    bad[0, 0] = np.inf
    with pytest.raises(edits.EditError, match="non-finite"):
        edits.encode_tensor("w", w, bad)
    with pytest.raises(edits.EditError, match="explicit"):
        edits.encode_tensor("w", w, edits.bf16_round(w + 1), rank="auto", lossy=True)


# ------------------------------------------------------------------------------ containers
def make_edit(rank=None):
    base = {name: edits.bf16_round(RNG.normal(0, 0.05, shape)) for name, (shape, _, _) in TENSORS.items()}
    target = {name: v.copy() for name, v in base.items()}
    target["language.blocks.08.mlp.linear"] = edits.bf16_round(
        base["language.blocks.08.mlp.linear"] + 0.01 * np.outer(RNG.normal(size=64), RNG.normal(size=32)))
    target["action.final_norm.scale"][3] = edits.bf16_round(np.float32(1.5))
    info = {name: {"model_path": path, "scan_index": index} for name, (_, path, index) in TENSORS.items()}
    edit = edits.from_arrays(base, target, rank=rank, tensor_info=info,
                             meta={"source": {"candidate": "toy", "fingerprint": "0" * 16},
                                   "base": {"checkpoint": "pi05_droid_jointpos", "total_parameters": 3_353_433_872}})
    return base, target, edit


@pytest.mark.parametrize("suffix", [".safetensors", ".npz"])
def test_save_and_load_round_trip(tmp_path, suffix):
    base, target, edit = make_edit(rank="auto")
    assert sorted(edit.tensors) == ["action.final_norm.scale", "language.blocks.08.mlp.linear"]
    path = edit.save(tmp_path / f"toy{suffix}")
    assert path.name == f"toy{suffix}"
    loaded = edits.load(path)
    assert loaded.meta["format"] == edits.FORMAT and loaded.meta["format_version"] == edits.FORMAT_VERSION
    assert loaded.meta["source"]["candidate"] == "toy" and loaded.exact
    for name, t in edit.tensors.items():
        assert loaded.tensors[name].info == json.loads(json.dumps(t.info))
        assert same(loaded.tensors[name].reconstruct(base[name]), target[name])
    meta = loaded.metadata()
    assert meta["summary"]["changed_values"] == edit.changed_values
    assert meta["summary"]["changed_fraction"] == pytest.approx(edit.changed_values / 3_353_433_872)


def test_load_rejects_foreign_and_future_files(tmp_path):
    _safetensors.save_file({"x": np.zeros(3, np.float32)}, tmp_path / "plain.safetensors")
    with pytest.raises(edits.EditError, match="not a weightsmith edit"):
        edits.load(tmp_path / "plain.safetensors")
    _, _, edit = make_edit()
    meta = edit.metadata()
    meta["format_version"] = edits.FORMAT_VERSION + 1
    payload = {f"{n}/{p}": a for n, t in edit.tensors.items() for p, a in t.arrays.items()}
    _safetensors.save_file(payload, tmp_path / "future.safetensors", {edits.METADATA_KEY: json.dumps(meta)})
    with pytest.raises(edits.EditError, match="newer"):
        edits.load(tmp_path / "future.safetensors")


def test_safetensors_container_round_trip(tmp_path):
    arrays = {"a": RNG.normal(size=(3, 4)).astype(np.float32), "b": np.arange(5, dtype=np.int64),
              "c": np.array(7, dtype=np.int32), "d": np.zeros((0, 3), np.float32), "e": np.array([True, False])}
    _safetensors.save_file(arrays, tmp_path / "x.safetensors", {"k": "v"})
    got, meta = _safetensors.load_file(tmp_path / "x.safetensors")
    assert meta == {"k": "v"} and set(got) == set(arrays)
    for k, v in arrays.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape and np.array_equal(got[k], v)


def test_interoperates_with_the_official_safetensors_library(tmp_path):
    st = pytest.importorskip("safetensors")
    from safetensors.numpy import load_file, save_file

    _, _, edit = make_edit(rank=2)
    path = edit.save(tmp_path / "e.safetensors")
    official = load_file(str(path))
    ours, _ = _safetensors.load_file(path)
    assert set(official) == set(ours)
    for key in ours:
        assert np.array_equal(official[key], ours[key]) and official[key].dtype == ours[key].dtype
    with st.safe_open(str(path), framework="np") as f:
        assert json.loads(f.metadata()[edits.METADATA_KEY])["format"] == edits.FORMAT
    arrays = {"w": RNG.normal(size=(5, 6)).astype(np.float32), "i": np.arange(4, dtype=np.int32)}
    save_file(arrays, str(tmp_path / "official.safetensors"), metadata={"m": "1"})
    back, meta = _safetensors.load_file(tmp_path / "official.safetensors")
    assert meta == {"m": "1"} and all(np.array_equal(back[k], arrays[k]) for k in arrays)


# ------------------------------------------------------------------------------ application targets
def test_apply_to_arrays():
    base, target, edit = make_edit()
    out = edit.apply(base)
    for name in base:
        assert same(out[name], target[name])
    assert out["language.blocks.09.mlp.linear"] is base["language.blocks.09.mlp.linear"]  # untouched: passed through
    raw = dict(base)  # raw float32 checkpoint values that round to the same bf16 base are accepted
    raw["language.blocks.08.mlp.linear"] = base["language.blocks.08.mlp.linear"] * np.float32(1 + 2**-12)
    assert same(edit.apply(raw)["language.blocks.08.mlp.linear"], target["language.blocks.08.mlp.linear"])
    wrong = dict(base, **{"action.final_norm.scale": base["action.final_norm.scale"] + 1})
    with pytest.raises(edits.EditError, match="base values differ"):
        edit.apply(wrong)
    forced = edit.apply(wrong, verify=False)["action.final_norm.scale"]  # the delta lands on the other base
    delta = np.float32(1.5) - base["action.final_norm.scale"][3]
    assert forced[3] == edits.bf16_round(wrong["action.final_norm.scale"][3] + delta)
    with pytest.raises(edits.EditError, match="missing"):
        edit.apply({"language.blocks.08.mlp.linear": base["language.blocks.08.mlp.linear"]})


def test_apply_to_openpi_params_tree():
    base, target, edit = make_edit()
    layers = np.stack([base["language.blocks.08.mlp.linear"] if i == 8 else edits.bf16_round(RNG.normal(size=(64, 32)))
                       for i in range(18)])
    params = {"PaliGemma": {"llm": {"layers": {"mlp": {"linear": layers.copy()}},
                                    "final_norm_1": {"scale": base["action.final_norm.scale"].copy()}}}}
    done = edit.apply_to_params(params)
    assert sorted(done) == ["PaliGemma/llm/final_norm_1/scale", "PaliGemma/llm/layers/mlp/linear[8]"]
    stacked = params["PaliGemma"]["llm"]["layers"]["mlp"]["linear"]
    assert same(stacked[8], target["language.blocks.08.mlp.linear"])
    assert same(np.delete(stacked, 8, axis=0), np.delete(layers, 8, axis=0))
    assert same(params["PaliGemma"]["llm"]["final_norm_1"]["scale"], target["action.final_norm.scale"])
    with pytest.raises(edits.EditError, match="no PaliGemma"):
        edit.apply_to_params({"PaliGemma": {}})


# ------------------------------------------------------------------------------ owsp candidates
def make_candidate(workspace, name="toy", rank_one=True):
    """An owsp-style candidate: as `owsp new` lays it out, with edits saved as the sandbox's save() does."""
    cfg = json.loads((workspace / "workspace.json").read_text())
    cand = workspace.parent / "cands" / name
    (cand / "edits").mkdir(parents=True)
    (cand / "scratch").mkdir()
    (cand / "base").symlink_to(workspace / "base")
    (cand / "workspace.json").write_text(json.dumps(dict(cfg, max_changed_fraction=1.0)))
    (cand / "meta.json").write_text(json.dumps({"created": 0.0, "from": None, "note": "rank-1 test edit"}))
    targets = {}
    for tname in ("language.blocks.08.mlp.linear", "action.final_norm.scale"):
        w = np.load(workspace / "base" / _file_name(tname))
        delta = 0.01 * np.outer(RNG.normal(size=w.shape[0]), RNG.normal(size=w.shape[-1])) if w.ndim == 2 \
            else np.eye(1, w.size, 2).ravel() * 0.5
        targets[tname] = edits.bf16_round(w + delta)
        np.save(cand / "edits" / _file_name(tname), targets[tname].astype(np.float32), allow_pickle=False)
    return cand, targets


@pytest.mark.parametrize("rank", [None, "auto"])
def test_candidate_export_and_apply_reproduce_the_fingerprint(tmp_path, synthetic_workspace, rank):
    cand, targets = make_candidate(synthetic_workspace)
    edit = edits.from_candidate(cand, rank=rank)
    assert edit.meta["source"] == {"candidate": "toy", "fingerprint": edits.fingerprint(cand / "edits"),
                                   "from": None, "note": "rank-1 test edit", "created": 0.0}
    assert edit.meta["base"]["checkpoint"] == "pi05_droid_jointpos"
    assert edit.tensors["language.blocks.08.mlp.linear"].info["model_path"][-1] == "linear"
    path = edit.save(tmp_path / "toy.safetensors")
    report = edits.apply_to_workspace(edits.load(path), synthetic_workspace, tmp_path / "cands" / "toy-copy")
    copy = tmp_path / "cands" / "toy-copy"
    assert report["fingerprint_matches"] and report["fingerprint"] == edits.fingerprint(cand / "edits")
    assert (copy / "base").resolve() == (synthetic_workspace / "base").resolve() and (copy / "scratch").is_dir()
    cfg = json.loads((copy / "workspace.json").read_text())
    assert cfg["max_changed_fraction"] == 1.0 and cfg["tensors"] == json.loads(
        (synthetic_workspace / "workspace.json").read_text())["tensors"]
    assert json.loads((copy / "meta.json").read_text())["edit"]["source"]["candidate"] == "toy"
    for tname, value in targets.items():
        assert same(np.load(copy / "edits" / _file_name(tname)), value)
    with pytest.raises(edits.EditError, match="exists"):
        edits.apply_to_workspace(edit, synthetic_workspace, copy)


def test_apply_to_workspace_cleans_up_on_failure(tmp_path, synthetic_workspace):
    cand, _ = make_candidate(synthetic_workspace)
    edit = edits.from_candidate(cand)
    edit.tensors["action.final_norm.scale"].info["base_sha256"] = "0" * 64
    with pytest.raises(edits.EditError, match="base values differ"):
        edits.apply_to_workspace(edit, synthetic_workspace, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_from_candidate_rejects_unknown_files(synthetic_workspace):
    cand, _ = make_candidate(synthetic_workspace)
    np.save(cand / "edits" / "0123456789abcdef.npy", np.zeros(3, np.float32))
    with pytest.raises(edits.EditError, match="not exported tensors"):
        edits.from_candidate(cand)


def test_describe_summarizes(synthetic_workspace):
    cand, _ = make_candidate(synthetic_workspace)
    text = edits.describe(edits.from_candidate(cand, rank="auto"))
    assert "toy" in text and "pi05_droid_jointpos" in text and "exact     yes" in text
    assert "language.blocks.08.mlp.linear" in text and "of 3,353,433,872 parameters" in text
