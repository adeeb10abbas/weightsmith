"""CPU tests for the cookbook recipes in examples/recipes/ (no GPU host, OpenPI or simulator).

Every numbered recipe is an ``owsp py`` script. These tests import each one without running it, lint the folder
with ruff, check the example capture specs, and run every recipe's ``main()`` on small synthetic data through a
stand-in for the sandbox prelude (load/base/save/bf16 and episodes/obs/acts). The recipes were also run on the GPU
host against real captures; docs/COOKBOOK.md reports what each produced.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "examples" / "recipes"
sys.path.insert(0, str(REPO))

from weightsmith import distill as D  # noqa: E402
from weightsmith import edits as E  # noqa: E402
from weightsmith import stats as S  # noqa: E402

RECIPE_FILES = sorted(RECIPES.glob("[0-9][0-9]_*.py"))
# names the owsp py prelude defines (weightsmith/lab/sandbox.py, PRELUDE_EXTRA in weightsmith/owsp.py): a recipe
# that rebinds one breaks the sandbox (e.g. CONFIG is the workspace config that save() checks)
PRELUDE_NAMES = {"WORK", "CONFIG", "TENSORS", "READABLE_NAMES", "NAMES", "REFERENCES", "load", "base",
                 "load_original", "save", "save_edit", "drop_edit", "stats", "saved", "edited", "bf16", "DATA",
                 "NOTES", "episodes", "obs", "acts"}
REF = "Put the cube then the banana in the bowl"


def load_recipe(path: Path):
    spec = importlib.util.spec_from_file_location(f"recipe_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines functions only: main() runs under __main__, as in `owsp py`
    return mod


def recipe(stem: str) -> Path:
    return next(p for p in RECIPE_FILES if p.stem.startswith(stem))


# ------------------------------------------------------------------ static checks
def test_recipes_exist_and_import():
    assert len(RECIPE_FILES) == 10
    for path in RECIPE_FILES:
        mod = load_recipe(path)
        assert callable(mod.main) and isinstance(mod.SETTINGS, dict), path.name
        assert "docs/COOKBOOK.md" in (mod.__doc__ or ""), path.name
        assert path.read_text().endswith('if __name__ == "__main__":\n    main()\n'), path.name


def test_recipes_do_not_shadow_the_sandbox_prelude():
    for path in RECIPE_FILES:
        tree = ast.parse(path.read_text())
        bound = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        assert not bound & PRELUDE_NAMES, (path.name, bound & PRELUDE_NAMES)
        assert "weightsmith" not in bound, path.name  # the sandbox cannot import the repository


def test_recipes_pass_ruff():
    cmd = [shutil.which("ruff")] if shutil.which("ruff") else None
    if cmd is None:
        try:
            import ruff  # noqa: F401

            cmd = [sys.executable, "-m", "ruff"]
        except ImportError:
            pytest.skip("ruff is not installed (pip install -e '.[dev]')")
    p = subprocess.run([*cmd, "check", str(RECIPES)], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_example_specs_are_valid():
    check = load_recipe(RECIPES / "check_spec.py")
    specs = sorted((RECIPES / "specs").glob("*.json"))
    assert len(specs) == 6
    for f in specs:
        errors, _, info = check.check_spec(json.loads(f.read_text()))
        assert not errors, (f.name, errors)
        assert info["items"] <= 96 and info["mb_total"] < 1000
    errors, warnings, _ = check.check_spec({"items": [{"episode": "e", "k": 1, "step": 2}], "vlm_layer": [3],
                                            "vlm_layers": [18], "action_tokens": "all"})
    assert any("exactly one of k" in e for e in errors) and any("0..17" in e for e in errors)
    assert any("action_tokens" in e for e in errors) and any("vlm_layer" in w for w in warnings)
    assert check.main([str(f) for f in specs]) == 0


# ------------------------------------------------------------------ a stand-in for the owsp py sandbox
class Sandbox:
    """Weights, edits, episodes and captures of a temporary project, plus the prelude's functions."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.data = self.root / "data"
        self.edits_dir = self.root / "edits"
        for d in (self.data / "acts", self.data / "episodes", self.edits_dir):
            d.mkdir(parents=True)
        self.weights: dict[str, np.ndarray] = {}
        self.edits: dict[str, np.ndarray] = {}
        self.index: list[dict] = []

    def weight(self, name, value):
        self.weights[name] = D.bf16_round(value)

    def capture(self, aid, items, arrays, **index):
        d = self.data / "acts" / aid
        d.mkdir()
        np.savez(d / "acts.npz", **arrays)
        idx = {"cand": "base", "items": items, "action_layers": [17], "vlm_layers": [], "denoise_steps": [0],
               "denoise_t": [1.0], "action_tokens": "mean", "prompt_token_slots": list(range(24)),
               "base_parity_max_abs_action_error": 0.01, **index}
        (d / "index.json").write_text(json.dumps(idx))

    def episode(self, eid, cand, success, obs=None, prompt=REF):
        self.index.append({"eid": eid, "cand": cand, "fingerprint": "base", "slot": eid.rsplit(".", 1)[-1],
                           "success": success, "steps": 450})
        d = self.data / "episodes" / eid
        d.mkdir(exist_ok=True)
        (d / "summary.json").write_text(json.dumps({"eid": eid, "prompt": prompt}))
        if obs is not None:
            np.savez(d / "obs.npz", **obs)

    # the prelude
    def load(self, name):
        return self.edits.get(name, self.weights[name]).copy()

    def base(self, name):
        return self.weights[name].copy()

    def save(self, name, value):
        value = np.asarray(value, dtype=np.float32)
        assert value.shape == self.weights[name].shape and np.isfinite(value).all()
        self.edits[name] = value
        np.save(self.edits_dir / f"{name}.npy", value, allow_pickle=False)

    def saved(self):
        return sorted(self.edits)

    def episodes(self, cand=None, success=None):
        return [r for r in self.index if (cand is None or r["cand"] == cand) and (success is None or
                                                                                  r["success"] == success)]

    def obs(self, eid):
        z = np.load(self.data / "episodes" / eid / "obs.npz")
        return {k: z[k] for k in z.files}

    def acts(self, aid):
        idx = json.loads((self.data / "acts" / aid / "index.json").read_text())
        z = np.load(self.data / "acts" / aid / "acts.npz")
        return idx, {k: z[k] for k in z.files}

    def run(self, stem, **settings):
        mod = load_recipe(recipe(stem))
        for name in ("load", "base", "save", "saved", "episodes", "obs", "acts"):
            setattr(mod, name, getattr(self, name))
        mod.bf16 = D.bf16_round
        mod.DATA_ROOT = str(self.data)
        mod.EDITS_DIR = str(self.edits_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = mod.main({**mod.SETTINGS, **settings})
        return result, buf.getvalue(), mod


@pytest.fixture()
def sb(tmp_path):
    return Sandbox(tmp_path)


def chunk(closed_steps=0, n=15):
    """An action chunk whose last `closed_steps` steps command the gripper closed."""
    a = np.zeros((n, 8), np.float32)
    if closed_steps:
        a[-closed_steps:, 7] = 1.0
    return a


Q_FAR = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]   # fingertip about 34 cm from Q_GOAL in x-y
Q_GOAL = [0.6, 0.0, 0.0, -1.6, 0.0, 1.8, 0.8]


def episode_obs(chunks, grip, joints):
    K = len(chunks)
    return {"actions": np.stack(chunks), "gripper": np.asarray(grip, np.float32).reshape(K, 1),
            "joint": np.asarray(joints, np.float32), "step": np.arange(K) * 15, "seed": np.arange(K)}


# ------------------------------------------------------------------ each recipe on synthetic data
def test_01_find_decisions(sb):
    K = 8
    # failed: the chunk at k=3 closes only in its last steps, the object is in hand at k=4, chunk 4 opens: released
    fail = [chunk(), chunk(), chunk(), chunk(3), chunk(), chunk(), chunk(), chunk()]
    sb.episode("base.base.s1", "base", False,
               episode_obs(fail, [0, 0, 0, 0, 0.35, 0, 0, 0], [Q_FAR] * K))
    # success: holds k=4-6, chunk 6 opens over the goal
    ok = [chunk(), chunk(), chunk(), chunk(15), chunk(15), chunk(15), chunk(), chunk()]
    sb.episode("base.base.s2", "base", True,
               episode_obs(ok, [0, 0, 0, 0, 0.35, 0.36, 0.35, 0], [Q_FAR] * 6 + [Q_GOAL] * 2))
    lists, out, _ = sb.run("01", compare_with=[["base.base.s1", 4]], show_episodes=1)
    assert [(t["episode"], t["k"]) for t in lists["targets"]] == [("base.base.s1", 4)]
    assert lists["goal_releases"] == [{"episode": "base.base.s2", "k": 6}]
    assert [(c["episode"], c["k"]) for c in lists["carries"]] == [("base.base.s2", 4), ("base.base.s2", 5)]
    assert "compare_with: 1 of 1 found" in out


def test_02_prompt_swap(sb):
    items = [{"episode": f"base.base.s{i}", "k": 4, "prompt": REF} for i in range(3)]
    sb.capture("t-orig", items, {"actions": np.stack([chunk()] * 3)})
    alt = [{"episode": f"base.base.s{i}", "k": 4, "prompt": p} for p in ("alt A", "alt B") for i in range(3)]
    sb.capture("t-alt", alt, {"actions": np.stack([chunk(15), chunk(15), chunk()] + [chunk()] * 3)})
    rows, out, _ = sb.run("02", captures=["t-orig", "t-alt"], reference=REF, decisions=None, goal_xy=None)
    got = {r["prompt"]: (r["n"], r["closed"], r["flips"]) for r in rows}
    assert got == {REF: (3, 0, 0), "alt A": (3, 2, 2), "alt B": (3, 0, 0)}


def test_03_capture_info(sb):
    sb.episode("base.base.s1", "base", False, episode_obs([chunk()] * 5, [0] * 5, [Q_FAR] * 5))
    items = [{"episode": "base.base.s1", "k": 4, "prompt": REF}, {"episode": "base.base.s1", "k": 2, "prompt": "other"}]
    arrays = {"vlm.img_mlp_hidden": np.zeros((2, 1, 2, 17, 8), np.float16),
              "vlm.mlp_hidden": np.zeros((2, 1, 4, 8), np.float16), "actions": np.zeros((2, 15, 8), np.float32)}
    sb.capture("t-cap", items, arrays, vlm_layers=[3], prompt_token_slots=[0, 1, 2, 3],
               prompt_token_pieces_item0=["<bos>", "Task", ":", "<pad>"])
    check = [{"episode": "base.base.s1", "k": 4}, {"episode": "base.base.s9", "k": 0}, {"episode": "base.base.s1",
                                                                                         "k": 99}]
    out_specs, out, _ = sb.run("03", captures=["t-cap"], print_spec="t-cap", check_items=check)
    spec = out_specs["t-cap"]
    assert spec["items"] == [{"episode": "base.base.s1", "k": 4}, {"episode": "base.base.s1", "k": 2,
                                                                     "prompt": "other"}]
    assert spec["image_tokens"] is True and "action_attention" not in spec and spec["vlm_layers"] == [3]
    assert "vlm.img_mlp_hidden" in out and "MISSING" in out and "BAD" in out and "1 ok, 2 not usable" in out


def test_04_edit_distill_matches_the_library(sb):
    rng = np.random.default_rng(0)
    F, Dm, P = 64, 8, 10
    eps = [("base.base.s1", 4), ("base.base.s2", 5), ("base.base.s3", 4)]
    items = [{"episode": e, "k": k, "prompt": p} for p in ("orig", "expl") for e, k in eps]
    arrays = {"vlm.img_mlp_hidden": rng.normal(size=(6, 2, 2, 17, F)).astype(np.float16),
              "vlm.img_resid_out": rng.normal(size=(6, 2, 2, 17, Dm)).astype(np.float16),
              "vlm.mlp_hidden": rng.normal(size=(6, 2, P, F)).astype(np.float16),
              "vlm.resid_out": rng.normal(size=(6, 2, P, Dm)).astype(np.float16)}
    sb.capture("t-d", items, arrays, vlm_layers=[7, 8], prompt_token_slots=list(range(P)))
    name = "language.blocks.08.mlp.linear"
    sb.weight(name, rng.normal(size=(F, Dm)) * 0.1)
    _, out, mod = sb.run("04", capture="t-d", orig_items=[0, 1, 2], expl_items=[3, 4, 5], layers=[8],
                         prompt_slots=[[0, 0], [1, 1], [2, 2]])
    orig = {k: v[:3] for k, v in arrays.items()}
    expl = {k: v[3:] for k, v in arrays.items()}
    kv = D.build_keys_values(orig, expl, layers=[8], mode="both", slot_map=[(0, 0), (1, 1), (2, 2)],
                             capture_layers=[7, 8])
    want = D.bf16_round(sb.weights[name] + D.closed_form_edit(*kv[8], 1e-3))
    assert np.array_equal(sb.edits[name], want)
    assert f"candidate fingerprint {E.fingerprint(sb.edits_dir)}" in out and "3 prompt/state positions" in out
    # the pi0.5 template alignment agrees with weightsmith.distill (tokens of the distill tests)
    tok_o = [2, 7071, 235292, 10, 11, 12, 235269, 3040, 235292, 5, 6, 108, 4022, 235292, 235248] + [0] * 5
    tok_x = [2, 7071, 235292, 10, 13, 14, 11, 12, 235269, 3040, 235292, 5, 6, 108, 4022, 235292, 235248] + [0] * 3
    o_item = {"prompt_token_ids_at_slots": tok_o, "prompt_token_valid_at_slots": [t != 0 for t in tok_o]}
    x_item = {"prompt_token_ids_at_slots": tok_x, "prompt_token_valid_at_slots": [t != 0 for t in tok_x]}
    assert mod.pi05_slot_map(o_item, x_item) == D.pi05_slot_map(tok_o, tok_x, o_item["prompt_token_valid_at_slots"],
                                                                x_item["prompt_token_valid_at_slots"])


def readout_capture(sb, rng, N=20, Dh=48):
    u = rng.normal(size=Dh)
    H = rng.normal(size=(N, 1, 1, Dh)) + 0.05 * rng.normal(size=(N, 1, 15, Dh))
    H[:8] += 3 * u
    actions = np.stack([chunk(15) if 8 <= i < 12 or i >= 16 else chunk() for i in range(N)])
    items = [{"episode": f"base.base.s{i}", "k": 4, "prompt": REF} for i in range(N)]
    sb.capture("t-r", items, {"action.head_input": H.astype(np.float16), "actions": actions})
    return H.astype(np.float16).astype(np.float64)


def test_05_gated_readout(sb):
    rng = np.random.default_rng(1)
    H = readout_capture(sb, rng)
    sb.weight("action.action_out_proj.kernel", rng.normal(size=(48, 32)) * 0.1)
    groups = {"t": [0, 8], "carry": [8, 12], "g1": [12, 16], "g2": [16, 20]}
    a, out, _ = sb.run("05", capture="t-r", groups=groups, target="t", zero_on=["g1", "g2"], nuisance_pcs=8)
    resp = (H @ a).mean(axis=(1, 2))
    assert abs(resp[:8].mean() - 1) < 1e-6 and abs(resp[12:16].mean()) < 1e-6 and abs(resp[16:].mean()) < 1e-6
    new, old = sb.edits["action.action_out_proj.kernel"], sb.weights["action.action_out_proj.kernel"]
    assert np.array_equal(np.delete(new, 7, axis=1), np.delete(old, 7, axis=1))
    assert np.array_equal(new[:, 7], D.bf16_round(old[:, 7].astype(np.float64) - 1.8 * a))
    assert "split-half" in out and "gripper closed  0 ->  8" in out


def test_06_repurpose_unit(sb):
    rng = np.random.default_rng(2)
    Dw, F, N = 48, 24, 20  # wider than the 32 action dims, like the real expert (1,024)
    blk = "action.blocks.17"
    for name, shape, scale in (("action.time_mlp_in.kernel", (Dw, Dw), 0.3), ("action.time_mlp_in.bias", (Dw,), 0.1),
                               ("action.time_mlp_out.kernel", (Dw, Dw), 0.3), ("action.time_mlp_out.bias", (Dw,), 0.1),
                               (blk + ".pre_ffw_norm_1.Dense_0.kernel", (Dw, 3 * Dw), 0.01),
                               ("action.final_norm.Dense_0.kernel", (Dw, 3 * Dw), 0.01),
                               (blk + ".mlp_1.gating_einsum", (2, Dw, F), 0.3), (blk + ".mlp_1.linear", (F, Dw), 0.3),
                               ("action.action_out_proj.kernel", (Dw, 32), 0.3)):
        sb.weight(name, rng.normal(size=shape) * scale)
    shift = np.full(Dw, 3.0)  # a dominant constant direction in the MLP input, as in the real model
    for name in (blk + ".pre_ffw_norm_1.Dense_0.bias", "action.final_norm.Dense_0.bias"):
        sb.weight(name, np.concatenate([np.zeros(Dw), shift, np.full(Dw, 0.5)]))  # scale 0, shift, gate 0.5
    d = rng.normal(size=Dw)
    R = rng.normal(size=(N, Dw)) * 0.3 + np.where(np.arange(N)[:, None] < 8, 2.0, -2.0) * d
    M = np.abs(rng.normal(size=(N, F))) * 0.05
    items = [{"episode": f"base.base.s{i}", "k": 4, "prompt": REF} for i in range(N)]
    sb.capture("t-u", items, {"action.mlp_hidden": M.reshape(N, 1, 1, 1, F).astype(np.float16),
                              "action.resid_out": R.reshape(N, 1, 1, 1, Dw).astype(np.float16),
                              "actions": np.stack([chunk()] * 8 + [chunk(15)] * 12)})
    groups = {"t": [0, 8], "s": [8, 20]}
    (r, h), out, _ = sb.run("06", capture="t-u", groups=groups, target="t", silent_on=["s"])
    G0, L0 = sb.weights[blk + ".mlp_1.gating_einsum"], sb.weights[blk + ".mlp_1.linear"]
    G1, L1 = sb.edits[blk + ".mlp_1.gating_einsum"], sb.edits[blk + ".mlp_1.linear"]
    others = [u for u in range(F) if u != r]
    assert np.array_equal(G1[:, :, others], G0[:, :, others]) and np.array_equal(L1[others], L0[others])
    assert not np.array_equal(G1[:, :, r], G0[:, :, r]) and not np.array_equal(L1[r], L0[r])
    assert np.abs(h[8:]).max() < 1e-4 and (h[:8] > 0.1).sum() >= 4  # one-sided: zero on every silent item
    assert "split-half" in out and "other 31 dims <= " in out and "stored values changed" in out


def test_07_lexical_unit(sb):
    rng = np.random.default_rng(3)
    F, Dm, P = 32, 8, 6
    prompts = {"go left": [2, 10, 20, 30, 0, 0], "go right": [2, 10, 21, 30, 0, 0], "left again": [2, 11, 12, 20, 0, 0]}
    items, H = [], []
    for p, ids in prompts.items():
        for e in range(3):
            items.append({"episode": f"base.base.s{e}", "k": 4, "prompt": p, "prompt_token_ids_at_slots": ids,
                          "prompt_token_valid_at_slots": [t != 0 for t in ids]})
            h = rng.uniform(-0.05, 0.05, size=(P, F))
            for j, t in enumerate(ids):
                if t == 20:
                    h[j, 5], h[j, 9] = 4.0, 2.0
            H.append(h)
    H = np.stack(H)[:, None]
    Rz = rng.normal(size=(len(items), 1, P, Dm)) * 3
    sb.capture("t-l", items, {"vlm.mlp_hidden": H.astype(np.float16), "vlm.resid_out": Rz.astype(np.float16)},
               vlm_layers=[0], prompt_token_slots=list(range(P)))
    name = "language.blocks.00.mlp.linear"
    sb.weight(name, rng.normal(size=(F, Dm)))
    units, out, _ = sb.run("07", capture="t-l", token_prompt="go left", contrast_prompt="go right", beta=2.0)
    assert units == [5, 9]
    changed_rows = np.where((sb.edits[name] != sb.weights[name]).any(axis=1))[0].tolist()
    assert changed_rows == [5, 9]
    assert "cosine 1.000" in out and "size 2.00 x |dr|" in out and "1 other captured wording" in out


def test_08_random_control(sb):
    rng = np.random.default_rng(4)
    W1, W2 = rng.normal(size=(40, 30)) * 0.1, rng.normal(size=(20, 8)) * 0.1
    sb.weight("a.dense", W1)
    sb.weight("a.sparse", W2)
    U, _ = np.linalg.qr(rng.normal(size=(40, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(30, 3)))
    sb.edits["a.dense"] = D.bf16_round(sb.weights["a.dense"] + (U * [5.0, 3.0, 2.0]) @ V.T)
    sparse = sb.weights["a.sparse"].copy()
    sparse[:, 3] += rng.normal(size=20)
    sb.edits["a.sparse"] = D.bf16_round(sparse)
    delta_d = sb.edits["a.dense"].astype(np.float64) - sb.weights["a.dense"]
    delta_s = sb.edits["a.sparse"].astype(np.float64) - sb.weights["a.sparse"]
    _, out, _ = sb.run("08")
    ctl_d = sb.edits["a.dense"].astype(np.float64) - sb.weights["a.dense"]
    ctl_s = sb.edits["a.sparse"].astype(np.float64) - sb.weights["a.sparse"]
    sv_e, sv_c = np.linalg.svd(delta_d, compute_uv=False), np.linalg.svd(ctl_d, compute_uv=False)
    assert np.allclose(sv_c[:3], sv_e[:3], rtol=1e-2) and sv_c[3] < 1e-2
    assert abs(np.linalg.svd(ctl_d)[0][:, 0] @ np.linalg.svd(delta_d)[0][:, 0]) < 0.9  # new directions
    assert np.array_equal(ctl_s != 0, delta_s != 0) and abs(np.linalg.norm(ctl_s) / np.linalg.norm(delta_s) - 1) < 1e-2
    assert "svd" in out and "rank 3" in out and "support" in out


def test_09_offline_check(sb):
    keys = [{"episode": f"base.base.s{i}", "k": 4, "prompt": REF} for i in range(6)]
    j = np.zeros((15, 8), np.float32)
    j[:, :7] = Q_FAR

    def acts_(closed):
        out = np.stack([j.copy() for _ in closed])
        out[:, :, 7] = np.asarray(closed, float)[:, None]
        return out

    sb.capture("t-b", keys, {"actions": acts_([0, 0, 0, 1, 1, 1])})
    sb.capture("t-good", keys, {"actions": acts_([1, 1, 1, 1, 1, 1])}, cand="good")
    sb.capture("t-bad", keys, {"actions": acts_([1, 0, 0, 1, 0, 1])}, cand="bad")
    sb.capture("t-t", keys[:3], {"actions": acts_([1, 1, 1])})
    res, out, _ = sb.run("09", base_capture="t-b", cand_captures=["t-good", "t-bad"], teacher_capture="t-t",
                         groups={"target": [0, 3], "keep": [3, 6]}, targets=["target"])
    assert res == {"t-good": True, "t-bad": False}
    assert "only 1/3 target flipped" in out and "keep: 1 flips" in out


def test_10_paired_outcomes_feed_weightsmith_stats(sb, tmp_path):
    for i, (b, c) in enumerate([(0, 1), (0, 0), (1, 1), (1, 0)]):
        sb.index.append({"eid": f"base.base.s{20000 + i}", "cand": "base", "slot": f"s{20000 + i}", "success": bool(b)})
        sb.index.append({"eid": f"e1.x.s{20000 + i}", "cand": "e1", "slot": f"s{20000 + i}", "success": bool(c)})
    sb.index.append({"eid": "base.base.s20000.r2", "cand": "base", "slot": "s20000", "success": True})  # a rerun
    _, out, _ = sb.run("10", arms=["base", "e1"])
    f = tmp_path / "practice.csv"
    f.write_text(out)
    arms = S.read_arm_file(f).arms
    assert arms["base"] == {20000: False, 20001: False, 20002: True, 20003: True}  # first run of s20000
    c = S.compare(arms["base"], arms["e1"])
    assert (c.n, c.fixed, c.broke) == (4, 1, 1)
    _, out, _ = sb.run("10", arms=["base", "e1"], run="last")
    assert "20000,1,1" in out
