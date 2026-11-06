"""Shared fixtures for the toolkit tests. No GPU, simulator or network is needed.

``synthetic_workspace`` and ``synthetic_project`` lay out a tiny base export and project root in
exactly the on-disk format of the real runtime (``prepare_workspace`` and ``owsp``), so tests can
drive the real ``bin/owsp`` and ``weightsmith/owsp.py`` locally for the commands that need no GPU.
"""
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from weightsmith.lab.openpi_head import _effective_bfloat16, _file_name  # noqa: E402

# name -> (shape, model_path, scan_index), mimicking pi0.5's layer-split export at toy sizes
TENSORS = {
    "language.blocks.08.mlp.linear": ((64, 32), ["PaliGemma", "llm", "layers", "mlp", "linear"], 8),
    "language.blocks.09.mlp.linear": ((64, 32), ["PaliGemma", "llm", "layers", "mlp", "linear"], 9),
    "language.blocks.09.attn.q_einsum.w": ((4, 32, 8), ["PaliGemma", "llm", "layers", "attn", "q_einsum", "w"], 9),
    "action.blocks.03.mlp_1.linear": ((48, 16), ["PaliGemma", "llm", "layers", "mlp_1", "linear"], 3),
    "action.final_norm.scale": ((16,), ["PaliGemma", "llm", "final_norm_1", "scale"], None),
}

FAKE_OWSP = r'''
import json, os, sys
args = sys.argv[1:]
rec = {"args": args, "root": os.environ.get("WSP_ROOT"), "host": os.environ.get("WSP_HOST")}
if args and args[0] in ("py", "acts"):
    rec["file"] = open(args[-1]).read()
    rec["file_path"] = args[-1]
with open(os.environ["FAKE_OWSP_LOG"], "a") as f:
    f.write(json.dumps(rec) + "\n")
cmd = args[0]
if cmd == "status":
    print(json.dumps({"task": "FakeTask", "budget_used": {"episodes": 0}, "candidates": []}))
elif cmd == "frames":
    d = os.environ.get("WSP_FRAMES_DIR", "frames")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, args[1] + ".jpg")
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9")
    print("/remote/" + os.path.basename(p))
    print(os.path.abspath(p))
elif cmd == "eval" and args[1] == "over-budget":
    print("budget: practice episodes 1200+64 > 1200", file=sys.stderr)
    sys.exit(1)
else:
    print("ok " + " ".join(args))
'''


@pytest.fixture
def fake_owsp(tmp_path):
    """A stand-in for bin/owsp that logs each call and answers with canned output."""
    script = tmp_path / "fake_owsp.py"
    script.write_text(FAKE_OWSP)
    log = tmp_path / "owsp_calls.jsonl"
    env = dict(os.environ, FAKE_OWSP_LOG=str(log), WSP_FRAMES_DIR=str(tmp_path / "frames"),
               WSP_ROOT="/projects/demo", WSP_HOST="gpu-box")

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return SimpleNamespace(command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}",
                           env=env, log=log, calls=calls)


@pytest.fixture
def synthetic_workspace(tmp_path):
    """A tiny base export: workspace.json + base/*.npy, as scripts/prepare_workspace.sh writes them."""
    ws = tmp_path / "base-export"
    (ws / "base").mkdir(parents=True)
    (ws / "edits").mkdir()
    (ws / "scratch").mkdir()
    rng = np.random.default_rng(7)
    tensors, editable = {}, 0
    for name, (shape, path, index) in TENSORS.items():
        value = _effective_bfloat16(rng.normal(0, 0.05, shape).astype(np.float32))
        np.save(ws / "base" / _file_name(name), value, allow_pickle=False)
        tensors[name] = {"shape": list(shape), "dtype": "float32-effective-bfloat16", "numel": int(value.size),
                         "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()),
                         "std": float(value.std()), "file": _file_name(name), "model_path": path,
                         "scan_index": index, "packed_shape": [18, *shape] if index is not None else list(shape),
                         "writable": True}
        editable += value.size
    cfg = {"format": "openpi-vlm-action-internals-v1", "checkpoint": "/ckpt/pi05_droid_jointpos",
           "base_vlm_sha256": "f" * 64, "base_editable_sha256": "f" * 64, "include_action": True,
           "edit_scope": "VLM blocks, action expert, action projections, and time MLP",
           "packed_tensor_count": len(tensors), "tensor_count": len(tensors), "editable_parameters": editable,
           "total_parameters_and_buffers": 3_353_433_872, "max_changed_fraction": 1.0,
           "max_changed_elements": 3_353_433_872, "sandbox_memory_bytes": 1 << 30,
           "sandbox_timeout_seconds": 60, "tensors": tensors}
    (ws / "workspace.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return ws


@pytest.fixture
def synthetic_project(tmp_path, synthetic_workspace):
    """A project root with config.json and two recorded base practice episodes, plus env for local bin/owsp."""
    root = tmp_path / "project"
    episodes = root / "data" / "episodes"
    episodes.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "task": "RubiksCubeThenBananaTask", "instruction": "Put the cube then the banana in the bowl",
        "practice_seed_base": 20000, "practice_slots": 256, "envs": 16, "rendering_type": "performance",
        "limits": {"practice_episodes": 1200, "acts_items": 6000, "acts_items_per_call": 96}}))
    rows = []
    for slot, success in ((20000, False), (20001, True)):
        eid = f"base.base.s{slot}"
        rec = {"eid": eid, "cand": "base", "prompt": "Put the cube then the banana in the bowl",
               "fingerprint": "base", "slot": f"s{slot}", "success": success, "steps": 412, "score": 0.0,
               "reason": "" if success else "cube not in bowl",
               "events": [{"step": 120, "name": "OBJECT_GRABBED_SUCCESS", "info": "cube"}], "video": []}
        (episodes / eid).mkdir()
        (episodes / eid / "summary.json").write_text(json.dumps(rec))
        rows.append({k: v for k, v in rec.items() if k not in ("events", "video")})
    (episodes / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    env = dict(os.environ, WSP_ROOT=str(root), WSP_HOST="", BASE_WORKSPACE=str(synthetic_workspace),
               WSP_REMOTE_PYTHON=sys.executable, WSP_ENV_FILE=str(tmp_path / "no-wsp.env"),
               WSP_LOCK_DIR=str(tmp_path / "locks"), WSP_FRAMES_DIR=str(tmp_path / "frames"))
    return SimpleNamespace(root=root, workspace=synthetic_workspace, env=env)
