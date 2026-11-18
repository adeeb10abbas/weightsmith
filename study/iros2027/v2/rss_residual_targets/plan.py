"""Generate fresh layouts and seal all target-ablation candidates before tests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline import HERE, PRIOR, METHODS, LAYERS, dump, read, require, sha, verify_map, module


def payload_sha(plan):
    value = {key: item for key, item in plan.items() if key != "global_freeze"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def layouts(case, protocol):
    old = module("residual_targets_layout_geometry", PRIOR / "runtime/generate_layouts.py")
    seed = protocol["heldout"]["layout_generator_seed"] + (100 if case == "spatial" else 0)
    rng = np.random.default_rng(seed)
    rows = []
    for number in range(8):
        shift = rng.uniform(-.012, .012, size=2)
        poses = {}
        for name, (base, quat) in old.POSES[case].items():
            pos = np.array(base)
            pos[:2] += shift + rng.uniform(-.007, .007, size=2)
            poses[name] = {"pos": pos.round(10).tolist(), "quat": list(quat)}
        rows.append({"id": f"{case[0].upper()}-A{number + 1:02d}", "poses": poses})
    old_rows = []
    for split in ("development", "heldout"):
        old_rows += read(PRIOR / "runtime/layouts" / f"{case}_{split}.json")["layouts"]
    signatures = [json.dumps(row["poses"], sort_keys=True) for row in rows]
    require(len(set(signatures)) == 8, "New layouts must be distinct")
    require(not set(signatures) & {json.dumps(row["poses"], sort_keys=True) for row in old_rows},
            "New target-ablation layouts overlap earlier layout poses")
    require(not {row["id"] for row in rows} & {row["id"] for row in old_rows}, "Layout identifiers overlap")
    return {"case": case, "split": "heldout", "cohort": protocol["cohort"],
            "generator_seed": seed, "generation": "New deterministic XY perturbations; no policy-outcome selection",
            "objects": list(old.POSES[case]), "settle_steps": 15, "renders": 8, "layouts": rows,
            "interpretation": "Fresh small pose variations in the same scene, not new-object or new-scene generalization"}


def check_plan(plan, protocol):
    case = plan["case"]
    require(case in protocol["cases"] and plan["cohort_id"] == protocol["cohort"], "Wrong case/cohort")
    require(plan["split"] == "heldout" and plan["inference_batching"] == "fixed16", "Wrong split or inference batching")
    cells = {cell["id"]: cell for cell in protocol["cases"][case]["cells"]}
    wanted = Counter((cell, method) for cell in cells for method in METHODS)
    require(Counter((b["cell"], b["condition"]) for b in plan["batches"]) == wanted, "Missing or repeated cell/condition")
    for batch in plan["batches"]:
        cell = cells[batch["cell"]]
        require(batch["n"] == 16 and batch["slot_start"] == protocol["heldout"]["slot_start"][case], "Wrong slots/batch size")
        require(batch["task"] == cell["task"] and batch["goal"] == cell["goal"], "Changed task or predicate")
        require(batch["input_prompt"] == batch["effective_prompt"] == cell["prompt"], "All conditions must receive the same request")
        require(bool(batch["workspace"]) == (batch["condition"] != "base"), "Wrong edited/unedited workspace")
    require(sum(batch["n"] for batch in plan["batches"]) == 192, "Expected 192 episodes per case")


def make(run_root, output):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    require(not output.exists(), "Refusing to overwrite a frozen evaluation plan")
    protocol = read(HERE / "protocol.json")
    require(protocol["conditions"] == list(METHODS), "Unexpected target conditions")
    require(protocol["heldout"]["planned_total"] == 384, "Expected 384 new episodes")
    frozen, cases = {}, {}
    common_base = None
    for case in ("temporal", "spatial"):
        construction = run_root / f"{case}-construction-v1"
        status = read(construction / "status.json")
        require(status["status"] == "complete" and status["original_residual_candidate_byte_identical"] is True,
                "All target constructions must complete with identical residual reconstruction")
        seal_path = construction / "candidate-seals.json"
        require(sha(seal_path) == status["candidate_seals_sha256"], "Candidate seal changed")
        seals = read(seal_path)
        require(set(seals) == set(METHODS) - {"base"}, "All three target candidates are required")
        workspaces = {}
        for method, artifacts in seals.items():
            verify_map(artifacts)
            fit_path = construction / method / "fit.json"
            fit = read(fit_path)
            require(fit["method"] == method and fit["case"] == case and fit["split"] == "development"
                    and fit["validation_used_in_fit"] is False, "Candidate fit identity differs")
            verify_map(fit["source_sha256"])
            workspace = Path(fit["workspace"])
            manifest = read(workspace / "workspace.json")
            require(set(manifest["tensors"]) == {f"language.blocks.{l:02d}.mlp.linear" for l in LAYERS},
                    "Candidate has an unexpected module footprint")
            if common_base is None:
                common_base = manifest["base_vlm_sha256"]
            require(common_base == manifest["base_vlm_sha256"], "Candidates do not share the same base")
            for row in fit["layers"]:
                path = workspace / "edits" / manifest["tensors"][row["tensor"]]["file"]
                require(sha(path) == row["edited_sha256"], "Candidate differs from fit receipt")
                base = (workspace / "base" / manifest["tensors"][row["tensor"]]["file"]).resolve()
                require(sha(base) == row["base_sha256"], "Candidate base tensor changed")
                frozen[str(base)] = row["base_sha256"]
            frozen.update(artifacts)
            workspaces[method] = str(workspace)
        for path in (seal_path, construction / "status.json"):
            frozen[str(path)] = sha(path)
        layout = layouts(case, protocol)
        layout_path = run_root / "layouts" / f"{case}_target_ablation.json"
        require(not layout_path.exists(), "Fresh layout artifact already exists")
        dump(layout_path, layout)
        frozen[str(layout_path)] = sha(layout_path)
        batches = []
        for number, cell in enumerate(protocol["cases"][case]["cells"]):
            offset = (number + (2 if case == "spatial" else 0)) % 4
            methods = METHODS[offset:] + METHODS[:offset]
            for method in methods:
                batches.append({"cell": cell["id"], "condition": method, "task": cell["task"],
                                "goal": cell["goal"], "input_prompt": cell["prompt"], "effective_prompt": cell["prompt"],
                                "workspace": workspaces.get(method), "layouts": str(layout_path),
                                "slot_start": protocol["heldout"]["slot_start"][case], "n": 16})
        plan = {"case": case, "cohort_id": protocol["cohort"], "split": "heldout", "inference_batching": "fixed16",
                "protocol_sha256": sha(HERE / "protocol.json"), "batches": batches,
                "reuse_or_new": protocol["reuse_or_new"], "selection": "No candidate selection; all predeclared targets evaluated"}
        check_plan(plan, protocol)
        cases[case] = plan
    prior_planner = module("residual_targets_reused_runtime_inventory", PRIOR / "make_eval_plan.py")
    paths = [*prior_planner.runtime_sources(), HERE / "protocol.json", *sorted(HERE.glob("*.py"))]
    for path in paths:
        frozen[str(path.resolve())] = sha(path)
    output.mkdir(parents=True)
    freeze_path = output / "freeze.json"
    dump(freeze_path, {"artifacts": frozen, "plan_sha256": {case: payload_sha(plan) for case, plan in cases.items()},
                       "cohort": protocol["cohort"], "planned_new_policy_episodes": 384,
                       "note": "Both cases, all targets, protocol, fresh layouts and complete case plans frozen before either evaluation begins."})
    for case, plan in cases.items():
        plan["global_freeze"] = {"path": str(freeze_path), "sha256": sha(freeze_path)}
        dump(output / f"{case}-plan.json", plan)
    return {"output": str(output), "planned_new_policy_episodes": 384, "new_policy_episodes_generated": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make(args.run_root, args.output), indent=2))
