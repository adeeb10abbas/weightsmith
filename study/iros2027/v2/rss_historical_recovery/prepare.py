"""Seal exact existing candidates and fresh development slots; no policy calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PRIOR = HERE.parent / "rss_repair_comparison"
sys.path.insert(0, str(REPO))
from weightsmith import config as C
from weightsmith.edits import fingerprint


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("rss_recovery_plan_helpers", PRIOR / "make_eval_plan.py")
sha = M.sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    root = REPO / "projects" / p["cohort_id"]
    if root.exists():
        raise FileExistsError(root)
    start, end = p["slot_start"], p["slot_start"] + 16
    checked = []
    for path in (REPO / "projects").glob("**/*plan*.json"):
        try:
            prior = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(prior, dict):
            continue
        for batch in prior.get("batches", []):
            if "slot_start" in batch and max(start, batch["slot_start"]) < min(end, batch["slot_start"] + batch.get("n", 16)):
                raise ValueError(f"Slot overlap with existing plan {path}")
        if prior.get("batches"):
            checked.append(str(path.resolve()))
    assets = {
        "historical_full": Path("/home/ali/vibes/opus-weight-surgery/projects/where-to-edit/heldout/where-94016-64/frozen-v1full"),
        "historical_small": Path("/home/ali/vibes/opus-weight-surgery/projects/where-to-edit/heldout/where-94016-64/frozen-v1img816"),
        "recent_closed_form": REPO / "projects/rss-repair-comparison-20261117/temporal-adaptation/edit-fit/workspace",
        "recent_lora": REPO / "projects/rss-repair-comparison-20261117/temporal-adaptation/lora-selected/workspace",
    }
    expected_fp = {"historical_full": "63db0c52a51306c9", "historical_small": "5bf8c4329c1e416a"}
    base = json.loads((C.BASE_WS / "workspace.json").read_text())
    frozen = {str(path): sha(path) for path in M.runtime_sources()}
    frozen.update({str(path.resolve()): sha(path) for path in HERE.glob("*") if path.is_file()})
    pool = HERE.parent / "rss_goal_expansion_v2/runtime/single_attempt_pool.py"
    frozen[str(pool)] = sha(pool)
    candidates = {}
    for condition, workspace in assets.items():
        manifest = json.loads((workspace / "workspace.json").read_text())
        for key in ["base_vlm_sha256", "checkpoint", "total_parameters_and_buffers"]:
            if manifest[key] != base[key]:
                raise ValueError(f"Base metadata mismatch: {condition}/{key}")
        if (workspace / "base").resolve() != (C.BASE_WS / "base").resolve():
            raise ValueError(f"Candidate does not refer to unchanged shared base: {condition}")
        actual_fp = fingerprint(workspace / "edits")
        if condition in expected_fp and actual_fp != expected_fp[condition]:
            raise ValueError(f"Historical artifact fingerprint mismatch: {condition}")
        edited = sorted((workspace / "edits").glob("*.npy"))
        expected_n = 17 if condition == "historical_full" else 9
        names = {entry["file"]: name for name, entry in manifest["tensors"].items()}
        expected_layers = range(17) if condition == "historical_full" else range(8, 17)
        if (len(edited) != expected_n or {names[f.name] for f in edited}
                != {f"language.blocks.{i:02d}.mlp.linear" for i in expected_layers}):
            raise ValueError(f"Unexpected edited tensors: {condition}")
        files = [workspace / "workspace.json", *edited]
        entries = {str(f.resolve()): sha(f) for f in files}
        frozen.update(entries)
        for f in edited:
            base_file = (workspace / "base" / f.name).resolve()
            frozen[str(base_file)] = sha(base_file)
        candidates[condition] = {"workspace": str(workspace), "fingerprint": actual_fp,
                                 "edited_tensors": len(edited), "artifacts": entries,
                                 "reuse_or_new": "exact existing frozen weights; no refitting"}
    adaptation = REPO / "projects/rss-repair-comparison-20261117/temporal-adaptation"
    cfseal = adaptation / "candidate-seals.json"
    lsseal = adaptation / "lora-selected/seal.json"
    for condition, sealpath, entry in [("recent_closed_form", cfseal, "closed_form"), ("recent_lora", lsseal, None)]:
        seal = json.loads(sealpath.read_text())
        if entry:
            seal = seal[entry]
        for file, expected in seal["artifacts"].items():
            if sha(file) != expected:
                raise ValueError(f"Prior candidate seal changed: {file}")
            frozen[str(Path(file).resolve())] = expected
        frozen[str(sealpath)] = sha(sealpath)
    # Hash real checkpoint content against the pre-existing checkpoint manifest.
    # The loader also checks each edited tensor's effective BF16 base exactly.
    checkpoint = {}
    for line in (REPO / "third_party/pi05_droid_jointpos.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        file = C.CHECKPOINT / relative
        if sha(file) != expected:
            raise ValueError(f"Checkpoint mismatch: {file}")
        checkpoint[str(file.resolve())] = expected
    frozen.update(checkpoint)
    generator = load("rss_recovery_layout_generator", PRIOR / "runtime/generate_layouts.py")
    rng = np.random.default_rng(p["layout_generator_seed"])
    layouts = []
    for i in range(8):
        shift = rng.uniform(-.012, .012, size=2)
        poses = {}
        for name, (xyz, quat) in generator.POSES["temporal"].items():
            pos = np.array(xyz)
            pos[:2] += shift + rng.uniform(-.007, .007, size=2)
            poses[name] = {"pos": pos.round(10).tolist(), "quat": quat}
        layouts.append({"id": f"T-RD{i+1:02}", "poses": poses})
    for file in (PRIOR / "runtime/layouts").glob("temporal_*.json"):
        old = json.loads(file.read_text())["layouts"]
        if any(a["poses"] == b["poses"] for a in layouts for b in old):
            raise ValueError(f"Layout repeated from {file}")
    layoutfile = root / "freeze/temporal_development.json"
    save(layoutfile, {"case": "temporal", "split": "development", "generator_seed": p["layout_generator_seed"],
                      "generation": "fixed deterministic XY perturbations, no policy feedback",
                      "objects": list(generator.POSES["temporal"]), "settle_steps": 15, "renders": 8, "layouts": layouts})
    frozen[str(layoutfile)] = sha(layoutfile)
    identityfile = root / "freeze/candidate-identities.json"
    save(identityfile, {"candidates": candidates, "base_workspace": str(C.BASE_WS),
                        "base_manifest_sha256": sha(C.BASE_WS / "workspace.json"),
                        "base_vlm_sha256": base["base_vlm_sha256"],
                        "base_verification": "Same base directory for all candidates; edited base tensors sealed; checkpoint content matches existing SHA manifest; loader enforces exact effective BF16 base equality",
                        "checkpoint_artifacts": checkpoint, "prior_plans_checked_for_slot_overlap": checked,
                        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})
    frozen[str(identityfile)] = sha(identityfile)
    batches = [{"cell": "original", "condition": condition, "task": p["task"], "goal": p["goal"],
                "input_prompt": p["original_prompt"],
                "effective_prompt": p["clarified_prompt"] if condition == "clarified_input" else p["original_prompt"],
                "workspace": str(assets[condition]) if condition in assets else None,
                "layouts": str(layoutfile), "slot_start": start, "n": 16} for condition in p["conditions"]]
    plan = {"cohort_id": p["cohort_id"], "split": "development", "case": "temporal",
            "inference_batching": "fixed16", "batches": batches,
            "protocol_sha256": sha(HERE / "protocol.json"), "reuse_or_new": p["reuse_or_new"],
            "purpose": p["purpose"], "max_policy_episodes": 96, "gpu": 0}
    freeze = root / "freeze/freeze.json"
    save(freeze, {"artifacts": frozen, "plan_sha256": {"temporal": M.plan_payload_sha256(plan)},
                  "created_unix": time.time(), "note": "Frozen before any new policy call. Development diagnostic only."})
    plan["global_freeze"] = {"path": str(freeze), "sha256": sha(freeze)}
    save(root / "freeze/temporal-plan.json", plan)
    print(json.dumps({"root": str(root), "planned_episodes": 96, "frozen_artifacts": len(frozen),
                      "candidate_fingerprints": {k: v["fingerprint"] for k, v in candidates.items()},
                      "checkpoint_files_verified": len(checkpoint), "freeze_sha256": sha(freeze)}, indent=2))


if __name__ == "__main__":
    main()
