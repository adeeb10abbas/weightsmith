"""Freeze a two-candidate development evaluation and select LoRA before heldout.

This command never launches a policy. It preserves the original and sweep fits.
Selection: total strict success on repair+primary preservation (32 trials per
candidate), then balanced validation physical-action MSE, then lower learning rate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

HERE = Path(__file__).resolve().parents[1]
LAYERS = list(range(8, 17))
NAMES = {f"language.blocks.{i:02d}.mlp.linear" for i in LAYERS}
SLOT_START = {"temporal": 122000, "spatial": 122016}
RULE = "maximize total strict successes across 16 repair and 16 primary-preservation trials; ties: lower balanced validation physical-action MSE, then lower learning rate"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def payload_sha(plan):
    value = {k: v for k, v in plan.items() if k != "global_freeze"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_artifacts(artifacts):
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("An artifact seal must contain files")
    for path, expected in artifacts.items():
        if not Path(path).is_absolute() or sha(path) != expected:
            raise ValueError(f"Sealed artifact changed: {path}")


def balanced_mse(training):
    roles = training["final_validation_actions"]["by_role"]
    if set(roles) != {"repair", "preservation"}:
        raise ValueError("Validation must report both repair and preservation roles")
    values = [float(roles[role]["robot_action_mse"]) for role in ("repair", "preservation")]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid validation physical-action MSE")
    return sum(values) / 2


def workspace_files(workspace):
    workspace = Path(workspace).resolve()
    config = read(workspace / "workspace.json")
    if set(config["tensors"]) != NAMES:
        raise ValueError("Candidate must edit exactly VLM MLP output layers 8–16")
    for layer in LAYERS:
        entry = config["tensors"][f"language.blocks.{layer:02d}.mlp.linear"]
        if entry.get("model_path") != ["PaliGemma", "llm", "layers", "mlp", "linear"] or entry.get("scan_index") != layer:
            raise ValueError("Candidate logical tensors do not map to the declared VLM modules")
    tensors = sorted((workspace / "edits").glob("*.npy"))
    if len(tensors) != 9 or {p.name for p in tensors} != {v["file"] for v in config["tensors"].values()}:
        raise ValueError("Candidate must contain exactly nine declared merged tensors")
    return [workspace / "workspace.json", *tensors]


def server_fingerprint(workspace, training):
    config = read(Path(workspace) / "workspace.json")
    edited = training["workspace_validation"]["edited_tensors"]
    if not set(edited) <= NAMES or not edited:
        raise ValueError("Training receipt lacks a nonempty edited-tensor set")
    h = hashlib.sha256()
    for name in sorted(edited):
        value = np.load(Path(workspace) / "edits" / config["tensors"][name]["file"], allow_pickle=False)
        h.update(name.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def nomination(adaptation):
    adaptation = Path(adaptation).resolve()
    path = adaptation / "lora-sweep/nomination.json"
    nom = read(path)
    case = nom.get("case")
    if case not in SLOT_START or nom.get("status") != "complete" or nom.get("split") != "development":
        raise ValueError("A completed development-only sweep nomination is required")
    spec_path = adaptation / "spec.json"
    dataset_path = adaptation / "lora-dataset/dataset.json"
    if Path(nom["spec_path"]).resolve() != spec_path or Path(nom["dataset_path"]).resolve() != dataset_path.parent:
        raise ValueError("Nomination belongs to another adaptation dataset")
    if sha(spec_path) != nom["spec_sha256"] or sha(dataset_path) != nom["dataset_manifest_sha256"]:
        raise ValueError("Nomination specification or targets changed")
    spec, dataset = read(spec_path), read(dataset_path)
    if spec.get("split") != "development" or dataset.get("split") != "development" or spec.get("case") != case:
        raise ValueError("LoRA selection must use this case's development data")
    if dataset["spec_sha256"] != nom["spec_sha256"]:
        raise ValueError("Teacher dataset and shared development specification differ")
    artifacts = {str(path): sha(path), str(spec_path): sha(spec_path), str(dataset_path): sha(dataset_path)}
    for prefix in ("source_freeze", "source_identity"):
        source = Path(nom[f"{prefix}_path"]).resolve()
        if sha(source) != nom[f"{prefix}_sha256"]:
            raise ValueError("Sweep source identity changed")
        contents = read(source)
        identity = contents.get("sources", contents.get("artifacts", contents))
        check_artifacts(identity)
        artifacts.update(identity)
        artifacts[str(source)] = sha(source)
    candidates = {}
    for entry in nom["candidates"]:
        cid = entry["candidate_id"]
        if cid in candidates or entry.get("status") != "complete":
            raise ValueError("All sweep candidates must be uniquely identified and complete")
        fit = Path(entry["fit_path"]).resolve()
        expected_parent = adaptation if entry["stage"] == "initial" else adaptation / "lora-sweep"
        if (entry["stage"] == "initial" and fit != adaptation / "lora-fit") or (entry["stage"] == "new" and not fit.is_relative_to(expected_parent)):
            raise ValueError("Candidate is outside its declared initial/sweep fit")
        workspace, training_path = Path(entry["workspace"]).resolve(), Path(entry["training_path"]).resolve()
        if workspace != fit / "workspace" or training_path != fit / "training.json":
            raise ValueError("Candidate workspace and training receipt have different owners")
        if sha(training_path) != entry["training_sha256"]:
            raise ValueError("Candidate training receipt changed")
        training = read(training_path)
        rate = float(entry["learning_rate"])
        if (training["dataset_manifest_sha256"] != nom["dataset_manifest_sha256"] or training["rank"] != 32
                or training["layers"] != LAYERS or training["steps"] != 200 or training["learning_rate"] != rate
                or training["seed"] != 17 or training["weight_decay"] != 0):
            raise ValueError("Candidate differs from the fixed shared-data LoRA recipe")
        mse = balanced_mse(training)
        if not math.isclose(mse, entry["balanced_validation_action_mse"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Nomination validation metric differs from training receipt")
        actual = {str(p): sha(p) for p in workspace_files(workspace)}
        if actual != entry["workspace_artifacts"]:
            raise ValueError("Candidate workspace differs from nomination")
        seal_path = Path(entry["seal_path"]).resolve()
        if sha(seal_path) != entry["seal_sha256"]:
            raise ValueError("Candidate seal changed")
        seal = read(seal_path)
        if seal.get("candidate_id") != cid or seal.get("case") != case or seal.get("workspace_artifacts") != actual:
            raise ValueError("Candidate seal belongs to another fit")
        if seal.get("dataset_manifest_sha256") != nom["dataset_manifest_sha256"]:
            raise ValueError("Candidate seal identifies different teacher targets")
        if seal.get("source_identity_sha256") != nom["source_identity_sha256"]:
            raise ValueError("Candidate seal identifies another frozen sweep source")
        check_artifacts(seal["artifacts"])
        artifacts.update(seal["artifacts"])
        artifacts.update(actual)
        artifacts[str(training_path)] = sha(training_path)
        artifacts[str(seal_path)] = sha(seal_path)
        candidates[cid] = {**entry, "balanced_validation_action_mse": mse,
                           "server_fingerprint": server_fingerprint(workspace, training)}
    if len(candidates) != 3 or {float(c["learning_rate"]) for c in candidates.values()} != {.001, .00003, .0001}:
        raise ValueError("Nomination must retain the initial rate and both predeclared additional rates")
    initial = [c for c in candidates.values() if c["stage"] == "initial"]
    new = [c for c in candidates.values() if c["stage"] == "new"]
    if len(initial) != 1 or initial[0]["learning_rate"] != .001 or len(new) != 2:
        raise ValueError("Nomination must compare the original fit with one new-rate nominee")
    nominee = min(new, key=lambda c: (c["balanced_validation_action_mse"], c["learning_rate"]))
    if nom["selected_new_candidate_id"] != nominee["candidate_id"] or nom["closed_loop_candidate_ids"] != [initial[0]["candidate_id"], nominee["candidate_id"]]:
        raise ValueError("Nomination violates the predeclared validation-MSE rule")
    return nom, candidates, artifacts


def runtime_sources():
    # Reuse the common environment inventory, excluding unrelated offline
    # producers that the root may update before making the later heldout plan.
    source = HERE / "make_eval_plan.py"
    spec = importlib.util.spec_from_file_location("selection_runtime_inventory", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    excluded = {source, HERE / "adapt_case.py", HERE / "prepare_shared.py", HERE / "edit/pipeline.py", HERE / "lora/train_lora.py"}
    return [p for p in module.runtime_sources() if p not in excluded]


def make_plan(adaptation, output):
    adaptation, output = Path(adaptation).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    nom, candidates, artifacts = nomination(adaptation)
    case = nom["case"]
    protocol = read(HERE / "protocol.json")
    cells = protocol["cases"][case]["cells"]
    original = [c for c in cells if c["id"] == "original"]
    preservation = [c for c in cells if c.get("role") == "primary_preservation"]
    if len(original) != 1 or len(preservation) != 1:
        raise ValueError("Protocol must declare one original and one primary preservation cell")
    layout_path = (HERE / "runtime/layouts" / f"{case}_development.json").resolve()
    layouts = read(layout_path)
    prefix = "T" if case == "temporal" else "S"
    if layouts.get("split") != "development" or layouts.get("case") != case or {x["id"] for x in layouts["layouts"]} != {f"{prefix}-D{i:02d}" for i in range(1, 5)}:
        raise ValueError("Selection must use exactly the four development layouts")
    active = dict(zip(("lora_initial", "lora_nominee"), nom["closed_loop_candidate_ids"], strict=True))
    batches = []
    for cell_number, cell in enumerate((original[0], preservation[0])):
        order = ("lora_initial", "lora_nominee") if cell_number == 0 else ("lora_nominee", "lora_initial")
        for condition in order:
            candidate = candidates[active[condition]]
            batches.append({"cell": "original" if cell_number == 0 else "primary_preservation", "protocol_cell_id": cell["id"],
                            "condition": condition, "candidate_id": candidate["candidate_id"], "task": cell["task"], "goal": cell["goal"],
                            "input_prompt": cell["prompt"], "effective_prompt": cell["prompt"], "workspace": candidate["workspace"],
                            "server_fingerprint": candidate["server_fingerprint"], "layouts": str(layout_path),
                            "slot_start": SLOT_START[case], "n": 16})
    plan = {"cohort_id": f"rss-repair-comparison-20261117-lora-selection-{case}-v1", "case": case,
            "split": "development", "purpose": "pre-heldout LoRA candidate selection", "inference_batching": "fixed16",
            "selection_rule": RULE, "adaptation": str(adaptation), "nomination_sha256": sha(adaptation / "lora-sweep/nomination.json"),
            "protocol_sha256": sha(HERE / "protocol.json"), "candidates": active, "batches": batches}
    for p in [*runtime_sources(), Path(__file__).resolve(), layout_path, HERE / "protocol.json"]:
        artifacts[str(p.resolve())] = sha(p)
    output.mkdir(parents=True)
    freeze = output / "freeze.json"
    dump(freeze, {"artifacts": artifacts, "plan_sha256": {case: payload_sha(plan)},
                  "note": "Development-only candidate selection; all three fitted rates retained; no heldout outcomes used."})
    plan["global_freeze"] = {"path": str(freeze), "sha256": sha(freeze)}
    dump(output / "plan.json", plan)
    print(json.dumps({"plan": str(output / "plan.json"), "episodes": 64, "case": case}))
    return plan


def verify_evaluation(root, adaptation, nom, candidates):
    root, adaptation = Path(root).resolve(), Path(adaptation).resolve()
    plan, status = read(root / "plan.json"), read(root / "status.json")
    case = nom["case"]
    if (plan.get("split") != "development" or status.get("split") != "development" or plan.get("case") != case
            or plan.get("inference_batching") != "fixed16" or plan.get("selection_rule") != RULE
            or Path(plan["adaptation"]).resolve() != adaptation or status.get("cohort_id") != plan["cohort_id"]
            or status.get("status") != "complete" or status.get("completed_batches") != 4 or status.get("completed_episodes") != 64):
        raise ValueError("Selection requires the complete, declared fixed16 development evaluation")
    if plan["nomination_sha256"] != sha(adaptation / "lora-sweep/nomination.json"):
        raise ValueError("Evaluation nomination changed")
    freeze = Path(plan["global_freeze"]["path"])
    if sha(freeze) != plan["global_freeze"]["sha256"]:
        raise ValueError("Evaluation source freeze changed")
    frozen = read(freeze)
    if frozen["plan_sha256"].get(case) != payload_sha(plan):
        raise ValueError("Evaluation plan differs from its original freeze")
    check_artifacts(frozen["artifacts"])
    identity = read(root / "identity.json")
    if any(identity.get(p) != digest for p, digest in frozen["artifacts"].items()):
        raise ValueError("Worker did not enforce the declared candidate/runtime freeze")
    active = dict(zip(("lora_initial", "lora_nominee"), nom["closed_loop_candidate_ids"], strict=True))
    if plan["candidates"] != active or len(plan["batches"]) != 4:
        raise ValueError("Evaluation does not compare the two nominated candidates")
    records, receipts, inputs, costs = {}, {}, {}, []
    paired_reference = None
    for number, batch_spec in enumerate(plan["batches"]):
        cell, condition = batch_spec["cell"], batch_spec["condition"]
        if (cell not in {"original", "primary_preservation"} or condition not in active
                or batch_spec["candidate_id"] != active[condition] or batch_spec["n"] != 16
                or batch_spec["slot_start"] != SLOT_START[case]):
            raise ValueError("Unexpected candidate, cell, or development slot schedule")
        key = (cell, condition)
        if key in records:
            raise ValueError("Repeated candidate/cell evaluation")
        candidate = candidates[active[condition]]
        if Path(batch_spec["workspace"]).resolve() != Path(candidate["workspace"]).resolve():
            raise ValueError("Evaluation uses another candidate workspace")
        batch = root / f"batch-{number:02d}-{cell}-{condition}"
        receipt, rows = read(batch / "completed.json"), read(batch / "episodes.json")
        if receipt["episodes_sha256"] != sha(batch / "episodes.json"):
            raise ValueError("Evaluation episode receipt changed")
        if (receipt.get("batch") != number or receipt.get("cell") != cell or receipt.get("condition") != condition
                or receipt.get("n") != 16 or len(rows) != 16):
            raise ValueError("Evaluation batch receipt identity differs")
        server = json.loads(receipt["server"]) if isinstance(receipt["server"], str) else receipt["server"]
        if Path(server.get("workspace") or "/").resolve() != Path(candidate["workspace"]).resolve() or server.get("fingerprint") != candidate["server_fingerprint"]:
            raise ValueError("Loaded model differs from the sealed candidate weights")
        slots = {r["slot"] for r in rows}
        if len(slots) != 16 or slots != set(range(SLOT_START[case], SLOT_START[case] + 16)):
            raise ValueError("Missing, duplicated, or unexpected selection slots")
        paired = {}
        for row in rows:
            expected = {"split": "development", "case": case, "cohort_id": plan["cohort_id"], "cell": cell,
                        "condition": condition, "inference_batching": "fixed16", "task_name": batch_spec["task"],
                        "input_prompt": batch_spec["input_prompt"], "effective_prompt": batch_spec["effective_prompt"],
                        "selected_goal": batch_spec["goal"]}
            if any(row.get(k) != v for k, v in expected.items()) or row.get("ok") is not True:
                raise ValueError("Episode differs from the declared development evaluation")
            layout = f"{'T' if case == 'temporal' else 'S'}-D{row['slot'] % 4 + 1:02d}"
            if row["layout_id"] != layout or type(row.get("policy_seed")) is not int or row["policy_seed"] < 0:
                raise ValueError("Episode does not use a valid development layout/seed")
            if type(row.get("success")) is not bool or row["strict_final"][batch_spec["goal"]] != row["success"]:
                raise ValueError("Success differs from the selected strict goal")
            obs, metrics_path = Path(row["obs"]).resolve(), Path(row["metrics"]).resolve()
            if not obs.is_relative_to(batch) or not metrics_path.is_relative_to(batch):
                raise ValueError("Evaluation observations/metrics belong to another batch")
            for p, digest in ((obs, row["obs_sha256"]), (metrics_path, row["metrics_sha256"])):
                if sha(p) != digest:
                    raise ValueError("Evaluation observation or metric changed")
                inputs[str(p)] = digest
            with np.load(obs, allow_pickle=False) as z:
                if int(z["seed"][0]) != row["policy_seed"]:
                    raise ValueError("Recorded policy seed differs from the episode row")
            metrics = read(metrics_path)
            e = row["env"]
            if metrics["selected_goal"] != row["selected_goal"] or bool(metrics["strict_final"][row["selected_goal"]][e]) != row["success"] or metrics["layout"]["layout_ids"][e] != layout:
                raise ValueError("Raw strict metrics differ from episode scoring")
            paired[row["slot"]] = (layout, row["policy_seed"])
        if paired_reference is not None and paired != paired_reference:
            raise ValueError("Candidates/goals do not share paired slots, layouts, and policy seeds")
        paired_reference = paired
        successes = sum(r["success"] for r in rows)
        if receipt["successes"] != successes:
            raise ValueError("Batch success count differs from strict episode outcomes")
        records[key] = rows
        for p in (batch / "completed.json", batch / "episodes.json"):
            receipts[str(p)] = sha(p)
        costs.append({"cell": cell, "condition": condition, "episodes": 16,
                      **{k: receipt[k] for k in ("wall_seconds", "execution_seconds", "queue_seconds") if k in receipt}})
    if set(records) != {(cell, c) for cell in ("original", "primary_preservation") for c in active}:
        raise ValueError("The complete two-by-two comparison is required")
    scores = []
    for condition, cid in active.items():
        candidate = candidates[cid]
        counts = {cell: sum(r["success"] for r in records[cell, condition]) for cell in ("original", "primary_preservation")}
        scores.append({"candidate_id": cid, "condition": condition, "successes_by_goal": counts,
                       "total_strict_successes": sum(counts.values()), "trials": 32,
                       "balanced_validation_action_mse": candidate["balanced_validation_action_mse"],
                       "learning_rate": candidate["learning_rate"]})
    winner = min(scores, key=lambda c: (-c["total_strict_successes"], c["balanced_validation_action_mse"], c["learning_rate"]))
    return winner, {"scores": scores, "batch_costs": costs, "source_receipt_sha256": receipts,
                    "source_observation_metric_sha256": inputs, "evaluation_plan_sha256": sha(root / "plan.json"),
                    "evaluation_identity_sha256": sha(root / "identity.json"), "global_freeze_sha256": sha(freeze)}


def select(adaptation, evaluation_root):
    adaptation, evaluation_root = Path(adaptation).resolve(), Path(evaluation_root).resolve()
    out = adaptation / "lora-selected"
    temporary = adaptation / ".lora-selected.inprogress"
    if out.exists() or temporary.exists():
        raise FileExistsError("Preserve existing selected/partial export; no overwrite")
    nom, candidates, _ = nomination(adaptation)
    winner, evidence = verify_evaluation(evaluation_root, adaptation, nom, candidates)
    candidate = candidates[winner["candidate_id"]]
    temporary.mkdir()
    shutil.copytree(candidate["workspace"], temporary / "workspace", symlinks=True)
    shutil.copy2(candidate["training_path"], temporary / "training.json")
    result = {"case": nom["case"], "split": "development", "status": "complete", "selection_rule": RULE,
              "selected_candidate_id": winner["candidate_id"], "selected_learning_rate": winner["learning_rate"],
              "selected_source_workspace": candidate["workspace"], "selected_training_sha256": candidate["training_sha256"],
              "dataset_manifest_sha256": nom["dataset_manifest_sha256"], "spec_sha256": nom["spec_sha256"],
              "nomination_sha256": sha(adaptation / "lora-sweep/nomination.json"), "evaluation_root": str(evaluation_root),
              "all_training_costs": nom.get("costs", {}), "failed_attempts": nom.get("failed_attempts", []),
              "heldout_used": False, "created_at": time.time(), **evidence}
    dump(temporary / "selection.json", result)
    source_files = workspace_files(candidate["workspace"])
    for source in source_files:
        relative = source.relative_to(Path(candidate["workspace"]).resolve())
        if sha(temporary / "workspace" / relative) != candidate["workspace_artifacts"][str(source)]:
            raise ValueError("Selected workspace copy differs from its sealed source")
    if sha(temporary / "training.json") != candidate["training_sha256"]:
        raise ValueError("Selected training receipt differs from its source")
    artifacts = {}
    for path in [temporary / "training.json", temporary / "selection.json", *workspace_files(temporary / "workspace")]:
        artifacts[str(out / path.relative_to(temporary))] = sha(path)
    dump(temporary / "seal.json", {"case": nom["case"], "split": "development", "selected_candidate_id": winner["candidate_id"],
                                   "artifacts": artifacts, "source_workspace_artifacts": candidate["workspace_artifacts"],
                                   "note": "Copied winner of the pre-heldout development selection; initial and sweep fits preserved."})
    temporary.rename(out)
    print(json.dumps({"selected": winner["candidate_id"], "workspace": str(out / "workspace"), "source_strict_successes": winner["total_strict_successes"], "trials": 32}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--adaptation", required=True); p.add_argument("--output", required=True)
    s = commands.add_parser("select")
    s.add_argument("--adaptation", required=True); s.add_argument("--evaluation-root", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.adaptation, args.output)
    else:
        select(args.adaptation, args.evaluation_root)


if __name__ == "__main__":
    main()
