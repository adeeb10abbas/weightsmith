"""Unit tests for weightsmith.distill (CPU only, synthetic data; no GPU host, no OpenPI).

Run: python3 -m pytest tests/test_distill.py   (or: python3 tests/test_distill.py)
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weightsmith import distill as D  # noqa: E402
from weightsmith.lab.sandbox import PRELUDE  # noqa: E402

RNG = np.random.default_rng(0)

# Token ids of the first cube-then-banana decision (capture c2-multi, item 0 and item 12), prompt slots 0..63.
CTB_ORIG = [2, 7071, 235292, 13298, 573, 28660, 1492, 573, 31985, 575, 573, 14581, 235269, 3040, 235292, 235248,
            235274, 235274, 235308, 235248, 235274, 235276, 235284, 235248, 235274, 235304, 235276, 235248, 235304,
            235284, 235248, 235274, 235284, 235310, 235248, 235274, 235310, 235274, 235248, 235274, 235274, 235321,
            235248, 235321, 235321, 235289, 108, 4022, 235292, 235248] + [0] * 14
CTB_EXPL = [2, 7071, 235292, 13298, 573, 28660, 575, 573, 14581, 1492, 2507, 573, 31985, 575, 573, 14581, 235269,
            3040, 235292, 235248, 235274, 235274, 235308, 235248, 235274, 235276, 235284, 235248, 235274, 235304,
            235276, 235248, 235304, 235284, 235248, 235274, 235284, 235310, 235248, 235274, 235310, 235274, 235248,
            235274, 235274, 235321, 235248, 235321, 235321, 235289, 108, 4022, 235292, 235248] + [0] * 10


def _sandbox_bf16():
    g: dict = {}
    exec(PRELUDE.split("\nnp.set_printoptions")[0] + "\n" + PRELUDE[PRELUDE.index("def bf16"):], g)
    return g["bf16"]


# ------------------------------------------------------------------ closed form
def test_closed_form_fits_keys_exactly_without_ridge():
    K = RNG.normal(size=(20, 300))
    V = RNG.normal(size=(20, 50))
    dW = D.closed_form_edit(K, V, lam_rel=1e-12)
    assert np.linalg.norm(K @ dW - V) / np.linalg.norm(V) < 1e-8
    # it is the minimum-norm least-squares solution
    ref = np.linalg.pinv(K) @ V
    assert np.linalg.norm(dW - ref) / np.linalg.norm(ref) < 1e-6


def test_closed_form_ridge_is_exact_up_to_lambda():
    K = RNG.normal(size=(30, 200)) + 3.0  # correlated keys (common offset), like mean activations
    V = RNG.normal(size=(30, 40))
    lam_rel = 1e-3
    dW = D.closed_form_edit(K, V, lam_rel)
    G = K @ K.T
    lam = lam_rel * np.trace(G) / len(G)
    expect = G @ np.linalg.solve(G + lam * np.eye(30), V)  # K dW = G (G + lam I)^-1 V
    assert np.allclose(K @ dW, expect, rtol=1e-9, atol=1e-9 * np.abs(expect).max())
    resid = V - K @ dW
    assert np.allclose(resid, lam * np.linalg.solve(G + lam * np.eye(30), V), atol=1e-9 * np.abs(V).max())
    # row space of K: no component in K's null space; rank <= number of keys
    P = K.T @ np.linalg.solve(K @ K.T, K)
    assert np.linalg.norm(dW - P @ dW) / np.linalg.norm(dW) < 1e-8
    assert np.linalg.matrix_rank(dW) <= 30


def test_closed_form_is_minimum_norm_among_exact_solutions():
    K = RNG.normal(size=(10, 60))
    V = RNG.normal(size=(10, 5))
    dW = D.closed_form_edit(K, V, 1e-14)
    N = RNG.normal(size=(60, 5))
    N -= K.T @ np.linalg.solve(K @ K.T, K @ N)  # K N = 0
    assert np.allclose(K @ (dW + N), V, atol=1e-8)
    assert np.linalg.norm(dW) < np.linalg.norm(dW + N)


def test_closed_form_rejects_bad_shapes():
    with pytest.raises(ValueError):
        D.closed_form_edit(np.ones((3, 4)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        D.closed_form_edit(np.ones((3, 4)), np.ones((3, 4)), lam_rel=-1)


# ------------------------------------------------------------------ increments
def test_layer_increments_telescope_and_need_previous_layer():
    gaps = {L: RNG.normal(size=(7, 5)) for L in range(6)}
    inc = D.layer_increments(gaps, range(6), alpha=1.0)
    assert np.allclose(sum(inc[L] for L in range(4)), gaps[3])
    assert np.array_equal(inc[0], gaps[0])
    sub = D.layer_increments(gaps, [3, 5], alpha=0.5)
    assert np.allclose(sub[3], 0.5 * (gaps[3] - gaps[2])) and np.allclose(sub[5], 0.5 * (gaps[5] - gaps[4]))
    with pytest.raises(KeyError):
        D.layer_increments({3: gaps[3]}, [3])


# ------------------------------------------------------------------ slot alignment
def test_pi05_slot_map_reproduces_the_cube_then_banana_alignment():
    valid_o = [t != 0 for t in CTB_ORIG]
    valid_x = [t != 0 for t in CTB_EXPL]
    m = D.pi05_slot_map(CTB_ORIG, CTB_EXPL, valid_o, valid_x)
    assert m == [(0, 0), (1, 1), (2, 2)] + [(13 + j, 17 + j) for j in range(37)]
    assert all(CTB_ORIG[a] == CTB_EXPL[b] for a, b in m)


def test_align_slots_generic_cases():
    a = [1, 2, 3, 9, 9, 4, 5]
    b = [1, 2, 3, 7, 4, 5]
    assert D.align_slots(a, b) == [(0, 0), (1, 1), (2, 2), (5, 4), (6, 5)]
    assert D.align_slots(a, b, max_prefix=1) == [(0, 0), (5, 4), (6, 5)]
    assert D.align_slots(a, b, suffix_from=5) == [(0, 0), (1, 1), (2, 2), (6, 5)]
    assert D.align_slots(a, a) == [(i, i) for i in range(7)]  # identical prompts: each position once
    pa, pb = a + [0, 0], b + [0, 0, 0]
    assert D.align_slots(pa, pb, orig_valid=[1] * 7 + [0, 0], expl_valid=[1] * 6 + [0] * 3) == D.align_slots(a, b)
    with pytest.raises(ValueError):
        D.align_slots(a, b, suffix_from=42)
    with pytest.raises(ValueError):
        D.align_slots(a, b, orig_valid=[1, 0, 1, 1, 1, 1, 1])


# ------------------------------------------------------------------ keys and values
def _fake_capture(N=3, Lc=4, P=10, F=12, Dd=6, seed=1):
    r = np.random.default_rng(seed)
    f16 = lambda *s: r.normal(size=s).astype(np.float16)  # noqa: E731
    return {"vlm.mlp_hidden": f16(N, Lc, P, F), "vlm.resid_out": f16(N, Lc, P, Dd),
            "vlm.img_mlp_hidden": f16(N, Lc, 2, 17, F), "vlm.img_resid_out": f16(N, Lc, 2, 17, Dd)}


def test_build_keys_values_matches_definition():
    o, x = _fake_capture(seed=1), _fake_capture(seed=2)
    cap_layers = [5, 6, 7, 8]
    slot_map = [(0, 0), (4, 6), (5, 7)]
    cells = [(0, 1), (1, 16)]
    kv = D.build_keys_values(o, x, layers=[6, 8], mode="both", alpha=2.0, slot_map=slot_map, cells=cells,
                             capture_layers=cap_layers)
    f = lambda a: a.astype(np.float64)  # noqa: E731

    def gap(li):
        rows = [(f(x["vlm.resid_out"][:, li, b]) - f(o["vlm.resid_out"][:, li, a])).mean(0) for a, b in slot_map]
        rows += [(f(x["vlm.img_resid_out"][:, li, c, e]) - f(o["vlm.img_resid_out"][:, li, c, e])).mean(0)
                 for c, e in cells]
        return np.stack(rows)

    for L in (6, 8):
        li = cap_layers.index(L)
        K, V = kv[L]
        keys = [f(o["vlm.mlp_hidden"][:, li, a]).mean(0) for a, _ in slot_map]
        keys += [f(o["vlm.img_mlp_hidden"][:, li, c, e]).mean(0) for c, e in cells]
        assert np.array_equal(K, np.stack(keys))  # slots first, then cells
        assert np.array_equal(V, 2.0 * (gap(li) - gap(li - 1)))
    img = D.build_keys_values(o, x, layers=[7], mode="image", capture_layers=cap_layers)
    assert img[7][0].shape == (32, 12) and img[7][1].shape == (32, 6)
    with pytest.raises(KeyError):  # layer 5 needs layer 4, which was not captured
        D.build_keys_values(o, x, layers=[5], mode="image", capture_layers=cap_layers)
    with pytest.raises(ValueError):
        D.build_keys_values(o, x, layers=[6], mode="state", capture_layers=cap_layers)  # no slot map
    with pytest.raises(ValueError):
        D.build_keys_values(o, {k: v[:2] for k, v in x.items()}, layers=[6], mode="image", capture_layers=cap_layers)


def test_build_edit_factored_equals_dense_closed_form():
    kv = {3: (RNG.normal(size=(8, 40)), RNG.normal(size=(8, 9))),
          4: (RNG.normal(size=(5, 40)), RNG.normal(size=(5, 9)))}
    e = D.build_edit(kv, 1e-3)
    for L, (K, V) in kv.items():
        lr = e.deltas[D.MLP_DOWN.format(L)]
        assert isinstance(lr, D.LowRank) and lr.rank == len(K) and lr.rel_error == 0.0
        assert np.allclose(lr.dense(), D.closed_form_edit(K, V, 1e-3), rtol=0, atol=1e-12)
    assert e.num_params() == 8 * (40 + 9) + 5 * (40 + 9)


# ------------------------------------------------------------------ low rank
def test_low_rank_reports_exact_reconstruction_error():
    M = RNG.normal(size=(200, 5)) @ RNG.normal(size=(5, 80)) + 1e-3 * RNG.normal(size=(200, 80))
    lr = D.low_rank(M, tol=1e-2)
    assert lr.rank == 5
    explicit = np.linalg.norm(M - lr.dense()) / np.linalg.norm(M)
    assert abs(explicit - lr.rel_error) < 1e-10 and 0 < lr.rel_error < 1e-2
    full = D.low_rank(M, tol=0.0)
    assert full.rank == 80 and full.rel_error == 0.0 and np.allclose(full.dense(), M)
    capped = D.low_rank(M, tol=0.0, max_rank=3)
    assert capped.rank == 3
    assert abs(np.linalg.norm(M - capped.dense()) / np.linalg.norm(M) - capped.rel_error) < 1e-10


def test_low_rank_factor_path_matches_dense_path():
    A, B = RNG.normal(size=(300, 12)), RNG.normal(size=(12, 70))
    a, b = D.low_rank((A, B), tol=1e-9), D.low_rank(A @ B, tol=1e-9)
    assert a.rank == b.rank == 12
    assert np.allclose(a.dense(), A @ B, atol=1e-9) and a.rel_error < 1e-12
    t1, t2 = D.low_rank((A, B), max_rank=4), D.low_rank(A @ B, max_rank=4)
    assert abs(t1.rel_error - t2.rel_error) < 1e-10
    assert np.allclose(t1.dense(), t2.dense(), atol=1e-8)


def test_closed_form_edit_rank_equals_number_of_keys():
    K, V = RNG.normal(size=(32, 500)), RNG.normal(size=(32, 64))
    lr = D.low_rank(D.closed_form_edit(K, V), tol=1e-10)
    assert lr.rank == 32 and lr.rel_error < 1e-12


# ------------------------------------------------------------------ bf16, Edit container
def test_bf16_round_matches_sandbox_helper_and_rounds_to_even():
    sb = _sandbox_bf16()
    x = RNG.normal(size=10000) * 10.0 ** RNG.integers(-6, 3, size=10000)
    assert np.array_equal(D.bf16_round(x), sb(x))
    one = np.float64(1.0)
    assert D.bf16_round(one + 2.0 ** -8)[()] == 1.0  # tie -> even (mantissa 0)
    assert D.bf16_round(one + 3 * 2.0 ** -8)[()] == 1.0 + 2.0 ** -6  # tie -> even (mantissa 2)
    assert D.bf16_round(one + 2.0 ** -7)[()] == 1.0 + 2.0 ** -7


def test_edit_save_load_apply_roundtrip(tmp_path):
    W = {"a": D.bf16_round(RNG.normal(size=(30, 7))), "b": D.bf16_round(RNG.normal(size=(30, 7)))}
    e = D.Edit({"a": RNG.normal(size=(30, 7)) * 1e-2,
                "b": D.LowRank(RNG.normal(size=(30, 2)), RNG.normal(size=(2, 7)) * 1e-2)},
               {"note": "test"})
    p = tmp_path / "edit.npz"
    e.save(p)
    e2 = D.Edit.load(p)
    assert e2.meta == {"note": "test"} and e2.names() == ["a", "b"]
    for n in ("a", "b"):
        assert np.array_equal(e2.dense(n), e.dense(n))
    new = e2.apply(W)
    assert new["a"].dtype == np.float32 and np.array_equal(new["a"], D.bf16_round(W["a"] + e.dense("a")))
    lr = e.to_low_rank(tol=0.0)
    assert lr.deltas["b"].rank == 2 and np.allclose(lr.dense("a"), e.dense("a"))


# ------------------------------------------------------------------ decision helpers
def _episode(closed_by_replan, grip):
    acts = np.zeros((len(closed_by_replan), 15, 8))
    acts[:, :, 7] = np.asarray(closed_by_replan, dtype=float)[:, None]
    return acts, np.asarray(grip, dtype=float)[:, None]


def test_select_early_releases_and_grasps():
    # cube-then-banana base.base.s20001: grasp at k=4, release of the held cube at k=5 (gripper 0.341)
    closed = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    grip = [0, 0, 0, 0, 0, .341, 0, 0, 0, .58, .6, .58, .347, 0]
    a, g = _episode(closed, grip)
    assert D.select_early_releases(a, g) == [5]
    assert D.select_early_releases(a, g, first_only=False) == [5]  # the 4-chunk banana carry is not early
    assert D.select_early_releases(a, g, max_hold=4, first_only=False) == [5, 12]
    assert D.select_grasps(a, g, first_only=False) == [4, 8]
    a2, g2 = _episode([0, 1, 0], [0, 0, 0.95])  # closed on nothing, then reopened: not a release of an object
    assert D.select_early_releases(a2, g2) == []


def test_gripper_closed_majority_and_fk():
    a = np.zeros((2, 15, 8))
    a[0, :8, 7] = 1.0
    a[1, :7, 7] = 1.0
    assert D.gripper_closed(a).tolist() == [True, False]
    p = D.panda_fk(np.zeros(7))
    assert np.allclose(p, [0.088, 0.0, 0.333 + 0.316 + 0.384 - 0.257], atol=1e-9)


def test_decision_summary_counts():
    base, expl = np.zeros((4, 15, 8)), np.zeros((4, 15, 8))
    expl[:, :, 7] = 1
    cand = expl.copy()
    cand[0, :, 7] = 0
    s = D.decision_summary(base, expl, cand)
    assert (s["closed_base"], s["closed_expl"], s["closed_cand"], s["flips_vs_base"]) == (0, 4, 3, 3)
    assert (s["base_agrees_expl"], s["cand_agrees_expl"]) == (0, 3)


# ------------------------------------------------------------------ rendered sandbox scripts
TOK_O = [2, 7071, 235292, 10, 11, 12, 235269, 3040, 235292, 5, 6, 108, 4022, 235292, 235248] + [0] * 5
TOK_X = [2, 7071, 235292, 10, 13, 14, 11, 12, 235269, 3040, 235292, 5, 6, 108, 4022, 235292, 235248] + [0] * 3


def _write_capture(root, aid, prompt, tokens, arrays, episodes):
    d = Path(root) / "acts" / aid
    d.mkdir(parents=True)
    np.savez(d / "acts.npz", **arrays)
    items = [{"i": i, "episode": e, "k": k, "prompt": prompt, "prompt_token_ids_at_slots": tokens,
              "prompt_token_valid_at_slots": [t != 0 for t in tokens]} for i, (e, k) in enumerate(episodes)]
    (d / "index.json").write_text(json.dumps({"items": items, "vlm_layers": [5, 6, 7, 8],
                                              "prompt_token_slots": list(range(len(tokens)))}))


class FakeSandbox:
    """Stand-in for the `owsp py` prelude: base weights, edits dict, load/base/save/bf16."""

    def __init__(self, F=64, Dd=6, layers=range(17)):
        r = np.random.default_rng(7)
        self.W = {D.MLP_DOWN.format(L): D.bf16_round(r.normal(size=(F, Dd)) * 0.1) for L in layers}
        self.edits: dict = {}

    def globals(self):
        bf16 = _sandbox_bf16()
        return {"np": np, "json": json, "os": os, "__name__": "__main__", "bf16": bf16,
                "base": lambda n: self.W[n].copy(), "load": lambda n: self.edits.get(n, self.W[n]).copy(),
                "save": lambda n, v: self.edits.__setitem__(n, np.asarray(v).astype(np.float32))}

    def run(self, script):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(script, "<sandbox>", "exec"), self.globals())
        return buf.getvalue()


@pytest.fixture()
def captures(tmp_path):
    eps = [("base.base.s1", 3), ("base.base.s2", 5), ("base.base.s3", 4)]
    o = _fake_capture(P=20, F=64, seed=11)  # more hidden units than keys (43 in 'both'), as in the model (16384)
    x = _fake_capture(P=20, F=64, seed=12)
    acts = np.zeros((3, 15, 8), np.float32)
    _write_capture(tmp_path, "t-o", "orig words", TOK_O, {**o, "actions": acts}, eps)
    xa = acts.copy()
    xa[:, :, 7] = 1
    _write_capture(tmp_path, "t-x", "explicit words", TOK_X, {**x, "actions": xa}, eps)
    return tmp_path, o, x


@pytest.mark.parametrize("mode", ["image", "state", "both"])
def test_rendered_build_script_reproduces_library_result(captures, mode):
    root, o, x = captures
    cfg = D.BuildConfig(orig_capture="t-o", expl_capture="t-x", layers=[6, 8], mode=mode, alpha=1.0, lam_rel=1e-3,
                        instruction="orig words", explicit_instruction="explicit words")
    script = D.render_sandbox_script(cfg, data_root=str(root), scratch=str(root))
    tree = ast.parse(script)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and "weightsmith" in ast.dump(n) for n in ast.walk(tree))
    sb = FakeSandbox()
    out = sb.run(script)
    res = D.parse_result(out)
    slot_map = (D.pi05_slot_map(TOK_O, TOK_X, [t != 0 for t in TOK_O], [t != 0 for t in TOK_X])
                if mode != "image" else None)
    if slot_map:
        assert slot_map == [(0, 0), (1, 1), (2, 2)] + [(7 + j, 9 + j) for j in range(8)]
        assert res["min_slot_consistency"] == 1.0
    kv = D.build_keys_values(o, x, layers=[6, 8], mode=mode, slot_map=slot_map, capture_layers=[5, 6, 7, 8])
    for L in (6, 8):
        name = D.MLP_DOWN.format(L)
        want = D.bf16_round(sb.W[name] + D.closed_form_edit(*kv[L], 1e-3))
        assert np.array_equal(sb.edits[name], want)
    assert sorted(sb.edits) == [D.MLP_DOWN.format(6), D.MLP_DOWN.format(8)]
    assert [r["layer"] for r in res["layers"]] == [6, 8] and all(r["fit_rel_err"] < 1e-2 for r in res["layers"])
    assert json.loads((root / "distill_manifest.json").read_text())["n_items"] == 3

    # compare script: zero difference against the stored tensors, then detects a one-value change
    cmp = D.render_compare_script(cfg, data_root=str(root), scratch=str(root))
    res2 = D.parse_result(sb.run(cmp))
    assert all(r["n_diff_vs_stored"] == 0 and r["sha256"] == r["stored_sha256"] for r in res2["layers"])
    sb.edits[D.MLP_DOWN.format(8)][0, 0] += 0.5
    res3 = D.parse_result(sb.run(cmp))
    assert [r["n_diff_vs_stored"] for r in res3["layers"]] == [0, 1]


def test_build_script_rejects_mismatched_items(captures):
    root, _, _ = captures
    cfg = D.BuildConfig(orig_capture="t-o", expl_capture="t-x", layers=[6], orig_items=[0, 1], expl_items=[1, 0])
    with pytest.raises(ValueError, match="different decisions"):
        FakeSandbox().run(D.render_sandbox_script(cfg, data_root=str(root)))
    cfg = D.BuildConfig(orig_capture="t-o", expl_capture="t-x", layers=[6], instruction="something else")
    with pytest.raises(ValueError, match="captured with"):
        FakeSandbox().run(D.render_sandbox_script(cfg, data_root=str(root)))


def test_check_and_select_scripts(captures):
    root, _, _ = captures
    eps = [("base.base.s1", 3), ("base.base.s2", 5), ("base.base.s3", 4)]
    cand = np.zeros((3, 15, 8), np.float32)
    cand[:2, :, 7] = 1
    _write_capture(root, "t-c", "orig words", TOK_O, {"actions": cand}, eps)
    out = FakeSandbox().run(D.render_check_script({"orig_capture": "t-o", "expl_capture": "t-x", "cand_capture": "t-c"},
                                                  data_root=str(root)))
    d = D.parse_result(out)["decisions"]
    assert (d["closed_base"], d["closed_expl"], d["closed_cand"], d["cand_agrees_expl"]) == (0, 3, 2, 2)

    a, g = _episode([0, 1, 0, 0], [0, 0, 0.3, 0])
    sb = FakeSandbox()
    glb = sb.globals()
    glb["episodes"] = lambda cand=None, success=None: [{"eid": "base.base.s9", "cand": "base", "success": False}]
    glb["obs"] = lambda eid: {"actions": a, "gripper": g, "step": np.arange(4) * 15}
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(D.render_select_script({"rule": "early_release"}, data_root=str(root)), glb)
    assert D.parse_result(buf.getvalue())["items"] == [{"episode": "base.base.s9", "k": 2}]


def test_build_config_validation():
    with pytest.raises(ValueError):
        D.BuildConfig(orig_capture="a", expl_capture="b", layers=[17])  # layer 17 is never read by the actions
    with pytest.raises(ValueError):
        D.BuildConfig(orig_capture="a", expl_capture="b", layers=[3], mode="text")
    with pytest.raises(ValueError):
        D.BuildConfig.from_dict({"orig_capture": "a", "expl_capture": "b", "layers": [3], "lamda": 1})


# ------------------------------------------------------------------ pipeline (dry run)
def test_pipeline_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry run must not run commands")
    monkeypatch.setattr(subprocess, "run", boom)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "examples/cube_then_banana/distill.json").read_text())
    res = D.run_pipeline(cfg, dry_run=True, workdir=tmp_path)
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "heldout.py" in out and res["candidate"] == cfg["candidate"]
    specs = [json.loads((tmp_path / f"{cfg['name']}-{s}.json").read_text()) for s in ("o", "x")]
    assert specs[0]["prompt"] == cfg["instruction"] and specs[1]["prompt"] == cfg["explicit_instruction"]
    assert specs[0]["vlm_layers"] == list(range(7, 17)) and specs[0]["image_tokens"] is True
    ast.parse((tmp_path / "build.py").read_text())


def test_capture_specs_state_mode_captures_prompt_slots():
    cfg = {"name": "n", "instruction": "a", "explicit_instruction": "b", "edit": {"mode": "state", "layers": [0, 1]}}
    o, x = D.capture_specs(cfg, [{"episode": "e", "k": 1}])
    assert o["vlm_layers"] == [0, 1] and o["prompt_tokens"] == list(range(64)) and o["image_tokens"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
