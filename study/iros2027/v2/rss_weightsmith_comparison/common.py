"""Pure planning and identity helpers for the successful-recipe comparison."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PRIOR = HERE.parent / "rss_repair_comparison"
ROOT = REPO / "projects/rss-weightsmith-comparison-20261118"
POOL = HERE.parent / "rss_goal_expansion_v2/runtime/single_attempt_pool.py"
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def protocol():
    return read(HERE / "protocol.json")


def payload_hash(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "global_freeze"},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_layouts(split):
    p = protocol(); setting = p[split]
    poses = load("weightsmith_comparison_layout_source", PRIOR / "runtime/generate_layouts.py").POSES["temporal"]
    rng = np.random.default_rng(setting["generator_seed"])
    layouts = []
    for i in range(setting["layouts"]):
        shift = rng.uniform(-.012, .012, size=2)
        placed = {}
        for name, (position, rotation) in poses.items():
            xyz = np.array(position)
            xyz[:2] += shift + rng.uniform(-.007, .007, size=2)
            placed[name] = {"pos": xyz.round(10).tolist(), "quat": rotation}
        layouts.append({"id": f"WSC-{'D' if split == 'development' else 'H'}{i+1:02}", "poses": placed})
    return {"case": "temporal", "split": split, "generator_seed": setting["generator_seed"],
            "generation": "Fixed small XY perturbations, without policy-outcome selection",
            "objects": list(poses), "settle_steps": 15, "renders": 8, "layouts": layouts}


def test_batches(workspaces):
    p = protocol(); result = []
    # Four blocks cannot perfectly balance five positions. Alternate cyclic
    # order and its reversal; freeze the exact schedule before any outcomes.
    for block in range(4):
        order = p["conditions"][block:] + p["conditions"][:block]
        if block % 2:
            order = list(reversed(order))
        for condition in order:
            result.append({"cell": f"original-block{block}", "condition": condition,
                           "task": p["task"], "goal": p["goal"],
                           "input_prompt": p["original_prompt"],
                           "effective_prompt": p["clarified_prompt"] if condition == "clarified_input" else p["original_prompt"],
                           "workspace": workspaces.get(condition),
                           "layouts": str(ROOT / "design/heldout-layouts.json"),
                           "slot_start": p["heldout"]["slot_start"] + 16 * block, "n": 16})
    return result


def validate_plan(plan):
    p = protocol()
    if (plan.get("cohort_id") != p["cohort_id"] or plan.get("case") != "temporal"
            or plan.get("inference_batching") != "fixed16" or plan.get("protocol_sha256") != sha(HERE / "protocol.json")):
        raise ValueError("Plan does not match this protocol")
    batches = plan["batches"]
    for b in batches:
        if (b.get("n") != 16 or b["task"] != p["task"] or b["goal"] != p["goal"]
                or b["input_prompt"] != p["original_prompt"]):
            raise ValueError("Batch differs from the registered task or episode budget")
    if plan["split"] == "heldout":
        spaces = {b["condition"]: b["workspace"] for b in batches}
        if batches != test_batches(spaces) or len(batches) != 20:
            raise ValueError("Heldout schedule or condition is not the declared 320-episode comparison")
        if spaces.get("base") is not None or spaces.get("clarified_input") is not None:
            raise ValueError("Unchanged-policy comparator has an edit")
        for arm in ("weightsmith", "matched_random", "teacher_action_lora"):
            if not spaces.get(arm):
                raise ValueError("Missing frozen candidate")
        if (spaces["weightsmith"] != p["historical_full"]["workspace"]
                or spaces["matched_random"] != p["matched_random"]["workspace"]):
            raise ValueError("Heldout plan replaces a frozen historical candidate")
        for arm in p["conditions"]:
            slots = [s for b in batches if b["condition"] == arm for s in range(b["slot_start"], b["slot_start"] + 16)]
            if slots != list(range(p["heldout"]["slot_start"], p["heldout"]["slot_start"] + 64)):
                raise ValueError("Heldout slots duplicated or missing")
    elif plan["split"] == "development":
        expected = sorted(p["lora"]["learning_rates"])
        if (len(batches) != 3 or sorted(b["learning_rate"] for b in batches) != expected
                or len({b["condition"] for b in batches}) != 3):
            raise ValueError("Selection must include exactly the three predeclared rates")
        for b in batches:
            if (b["cell"] != "lora_selection" or b["slot_start"] != p["development"]["slot_start"]
                    or b["effective_prompt"] != p["original_prompt"] or not b.get("workspace")
                    or b["layouts"] != str(ROOT / "design/development-layouts.json")):
                raise ValueError("Selection batch differs from development protocol")
    else:
        raise ValueError("Unknown split")


def select_winner(rows):
    if len(rows) != 3 or len({r["learning_rate"] for r in rows}) != 3:
        raise ValueError("Selection requires all three candidates")
    if any(r["n"] != 16 or not 0 <= r["successes"] <= 16 for r in rows):
        raise ValueError("Selection requires complete candidate trials")
    return sorted(rows, key=lambda r: (-r["successes"], r["learning_rate"]))[0]


def workspace_identity(path, expected_fingerprint=None):
    from weightsmith import config as C
    from weightsmith.edits import fingerprint
    path = Path(path).resolve(); manifest = read(path / "workspace.json"); base = read(C.BASE_WS / "workspace.json")
    for key in ("base_vlm_sha256", "checkpoint", "total_parameters_and_buffers"):
        if manifest[key] != base[key]:
            raise ValueError(f"Candidate base differs: {path}/{key}")
    if (path / "base").resolve() != (C.BASE_WS / "base").resolve():
        raise ValueError("Candidate does not use the shared immutable base")
    edited = sorted((path / "edits").glob("*.npy"))
    names = {v["file"]: k for k, v in manifest["tensors"].items()}
    if len(edited) != 17 or {names[f.name] for f in edited} != {f"language.blocks.{i:02d}.mlp.linear" for i in range(17)}:
        raise ValueError("Expected full VLM layers 0–16 and no other edits")
    fp = fingerprint(path / "edits")
    if expected_fingerprint and fp != expected_fingerprint:
        raise ValueError("Frozen historical candidate fingerprint differs")
    return {"workspace": str(path), "fingerprint": fp,
            "artifacts": {str(f): sha(f) for f in [path / "workspace.json", *edited]},
            "base_artifacts": {str((path / "base" / f.name).resolve()): sha(path / "base" / f.name) for f in edited}}


def verify_registry():
    path = ROOT / "lora-adaptation/candidates.json"
    registry = read(path); p = protocol(); lr = p["lora"]
    if (registry.get("split") != "development" or registry.get("rank") != lr["rank"]
            or registry.get("layers") != lr["layers"] or registry.get("steps") != lr["steps"]):
        raise ValueError("LoRA registry differs from the registered recipe")
    candidates = registry["candidates"]
    if (len(candidates) != 3 or sorted(c["learning_rate"] for c in candidates) != lr["learning_rates"]
            or len({c["candidate_id"] for c in candidates}) != 3):
        raise ValueError("LoRA registry must contain all three rates")
    numerical_boundary = ROOT / "lora-adaptation/numerical-boundary.json"
    if not numerical_boundary.is_file():
        raise ValueError("Historical teacher-target replay boundary must be recorded before selection")
    artifacts = {str(path): sha(path), str(numerical_boundary): sha(numerical_boundary)}
    dataset_path = ROOT / "lora-preparation/dataset.json"
    dataset = read(dataset_path)
    if (dataset.get("cohort_id") != p["cohort_id"] or dataset.get("split") != "development"
            or dataset.get("train_inputs") != 6 or dataset.get("validation_inputs") != 0
            or [(r["episode"], r["decision_index"]) for r in dataset["records"]]
            != [tuple(x) for x in p["source_decisions"]]):
        raise ValueError("LoRA training observations do not match the six historical decisions")
    artifacts[str(dataset_path)] = sha(dataset_path)
    for record in dataset["records"]:
        item = dataset_path.parent / record["file"]
        if sha(item) != record["sha256"]:
            raise ValueError("Prepared teacher-action item changed")
        artifacts[str(item)] = record["sha256"]
    for candidate in candidates:
        if not candidate.get("artifacts"):
            raise ValueError("Candidate is missing its artifact seal")
        for file, checksum in candidate["artifacts"].items():
            if sha(file) != checksum:
                raise ValueError(f"LoRA candidate artifact changed: {file}")
            artifacts[str(Path(file).resolve())] = checksum
        identity = workspace_identity(candidate["workspace"])
        for file, checksum in identity["artifacts"].items():
            if candidate["artifacts"].get(file) != checksum:
                raise ValueError("LoRA workspace is not covered by its post-training seal")
        for key in ("training_path", "seal_path"):
            artifacts[str(Path(candidate[key]).resolve())] = sha(candidate[key])
        training = read(candidate["training_path"])
        if (training.get("cohort_id") != p["cohort_id"] or training.get("split") != "development"
                or training.get("rank") != lr["rank"] or training.get("layers") != lr["layers"]
                or training.get("steps") != lr["steps"] or training.get("seed") != lr["seed"]
                or training.get("learning_rate") != candidate["learning_rate"]
                or training.get("dataset_sha256") != sha(dataset_path)
                or training.get("train_inputs") != 6 or training.get("validation_inputs") != 0
                or training.get("weight_decay") != 0 or training.get("image_augmentation") is not False):
            raise ValueError("Training receipt differs from the predeclared matched-source recipe")
        artifacts.update(identity["base_artifacts"])
    return candidates, artifacts


def runtime_artifacts():
    m = load("weightsmith_comparison_prior_plans", PRIOR / "make_eval_plan.py")
    paths = m.runtime_sources() + [POOL, HERE / "protocol.json"]
    paths += sorted(p for p in HERE.rglob("*.py") if "tests" not in p.parts)
    paths += sorted((ROOT / "design").glob("*.json"))
    return {str(p.resolve()): sha(p) for p in paths}


def freeze_plan(directory, plan, artifacts):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    validate_plan(plan)
    all_artifacts = {**runtime_artifacts(), **artifacts}
    directory.mkdir(parents=True)
    freeze = directory / "freeze.json"
    save(freeze, {"artifacts": all_artifacts, "plan_sha256": {"temporal": payload_hash(plan)},
                  "note": "Complete plan payload and source/candidate identities sealed before execution"})
    plan = {**plan, "global_freeze": {"path": str(freeze), "sha256": sha(freeze)}}
    save(directory / "plan.json", plan)
    return directory / "plan.json"
