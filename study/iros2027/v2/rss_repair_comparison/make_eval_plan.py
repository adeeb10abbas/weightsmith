"""Freeze both cases and all fitted candidates before any comparison test run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
LAYER_NAMES = {f"language.blocks.{i:02d}.mlp.linear" for i in range(8, 17)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def plan_payload_sha256(plan):
    """Bind the complete case plan without its circular global-manifest pointer."""
    payload = {k: v for k, v in plan.items() if k != "global_freeze"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_sources():
    """Freeze both cases' execution/observation code before either test starts."""
    from weightsmith import config as C
    files = [HERE / "worker.py", HERE / "protocol.json", HERE / "rewrites.py",
             Path(__file__), HERE / "prepare_shared.py", HERE / "adapt_case.py",
             HERE / "edit/pipeline.py", HERE / "lora/train_lora.py", HERE / "lora/sweep.py", HERE / "lora/select_candidate.py",
             C.BASE_WS / "workspace.json", REPO / "third_party/pi05_droid_jointpos.sha256"]
    files += [REPO / "weightsmith" / f for f in (
        "config.py", "pool.py", "rollout.py", "batched_server.py", "batched_client.py", "acts_worker.py", "distill.py",
        "lab/openpi_probe.py", "lab/openpi_vlm_full.py", "lab/openpi_head.py")]
    files += sorted((HERE / "runtime").rglob("*.py"))
    files += sorted((HERE / "runtime/layouts").glob("*.json"))
    openpi_sources = sorted((C.OPENPI_DIR / "src/openpi").rglob("*.py"))
    if not openpi_sources:
        raise ValueError("OpenPI checkpoint/model source tree is unavailable")
    files += openpi_sources
    robolab_package = C.ROBOLAB / "robolab"
    robolab_sources = sorted(p for p in robolab_package.rglob("*.py")
                            if p.relative_to(robolab_package).parts[0] != "tasks")
    if not robolab_sources:
        raise ValueError("RoboLab execution/scoring source tree is unavailable")
    files += robolab_sources
    files += [C.ROBOLAB / "policies/pi0_family/client.py"]
    files += [C.ROBOLAB / "assets/scenes" / name for name in
              ("rubiks_cube_banana_bowl.usda", "mustard_raisin_box.usda")]
    checkpoint_metadata = C.CHECKPOINT / "params/_METADATA"
    if not checkpoint_metadata.is_file():
        raise ValueError("Checkpoint parameter metadata is unavailable")
    files.append(checkpoint_metadata)
    return sorted(set(p.resolve() for p in files))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_selected_lora(adaptation, case):
    """Recompute the development winner once before constructing heldout plans."""
    selector = load_module("heldout_candidate_selector", HERE / "lora/select_candidate.py")
    selected = adaptation / "lora-selected"
    selection_path, seal_path = selected / "selection.json", selected / "seal.json"
    selection, seal = json.loads(selection_path.read_text()), json.loads(seal_path.read_text())
    nom, candidates, _ = selector.nomination(adaptation)
    if nom["case"] != case or selection.get("case") != case or selection.get("split") != "development" or selection.get("heldout_used") is not False:
        raise ValueError("Selected LoRA must come from this case's development comparison")
    evaluation = Path(selection["evaluation_root"]).resolve()
    winner, evidence = selector.verify_evaluation(evaluation, adaptation, nom, candidates)
    candidate = candidates[winner["candidate_id"]]
    if (selection.get("status") != "complete" or selection.get("selection_rule") != selector.RULE
            or selection.get("selected_candidate_id") != winner["candidate_id"]
            or selection.get("selected_learning_rate") != winner["learning_rate"]
            or selection.get("selected_source_workspace") != candidate["workspace"]
            or selection.get("selected_training_sha256") != candidate["training_sha256"]
            or selection.get("dataset_manifest_sha256") != nom["dataset_manifest_sha256"]
            or selection.get("spec_sha256") != nom["spec_sha256"]
            or selection.get("nomination_sha256") != sha(adaptation / "lora-sweep/nomination.json")
            or any(selection.get(key) != value for key, value in evidence.items())):
        raise ValueError("Selected LoRA receipt differs from the recomputed development winner/evidence")
    expected = {str(p): sha(p) for p in [selected / "training.json", selection_path, *selector.workspace_files(selected / "workspace")]}
    if (seal.get("case") != case or seal.get("split") != "development" or seal.get("selected_candidate_id") != winner["candidate_id"]
            or seal.get("artifacts") != expected or seal.get("source_workspace_artifacts") != candidate["workspace_artifacts"]):
        raise ValueError("Selected LoRA export differs from its seal")
    if sha(selected / "training.json") != candidate["training_sha256"]:
        raise ValueError("Selected training report differs from the nominated source fit")
    for source, checksum in candidate["workspace_artifacts"].items():
        rel = Path(source).relative_to(Path(candidate["workspace"]))
        if sha(selected / "workspace" / rel) != checksum:
            raise ValueError("Selected merged weights differ from the evaluated source candidate")
    artifacts = {**expected, **candidate["workspace_artifacts"], str(seal_path): sha(seal_path)}
    paths = [adaptation / "lora-sweep/nomination.json", Path(nom["source_freeze_path"]), Path(nom["source_identity_path"]),
             evaluation / "plan.json", evaluation / "status.json", evaluation / "identity.json"]
    plan = json.loads((evaluation / "plan.json").read_text())
    paths.append(Path(plan["global_freeze"]["path"]))
    for entry in nom["candidates"]:
        paths += [Path(entry["training_path"]), Path(entry["seal_path"])]
        if entry.get("log_path"):
            paths.append(Path(entry["log_path"]))
    artifacts.update(evidence["source_receipt_sha256"])
    # Raw selection NPZ files were checked above. Their digests remain in the
    # sealed selection receipt; do not rehash these large files every test batch.
    artifacts.update({p: v for p, v in evidence["source_observation_metric_sha256"].items() if Path(p).suffix != ".npz"})
    for path in paths:
        artifacts[str(path.resolve())] = sha(path)
    return {"selection": selection, "seal": seal, "artifacts": artifacts}


def validate_case(adaptation, case, protocol):
    status = json.loads((adaptation / "status.json").read_text())
    if status.get("status") != "complete" or status.get("case") != case:
        raise ValueError("Both adaptation procedures must finish before any heldout plan")
    spec_path = adaptation / "spec.json"
    spec = json.loads(spec_path.read_text())
    dataset_path = adaptation / "lora-dataset/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    fit = json.loads((adaptation / "lora-selected/training.json").read_text())
    edit = json.loads((adaptation / "edit-fit/solve.json").read_text())
    if spec.get("case") != case or spec.get("split") != "development" or dataset.get("split") != "development":
        raise ValueError("Adaptation artifacts must identify this case's development inputs")
    if not spec.get("cohort_id") or any(x.get("cohort_id") != spec["cohort_id"] for x in (dataset, fit, edit)):
        raise ValueError("Adaptation cohort identity differs")
    dataset_sha = sha(dataset_path)
    if edit["teacher_action_parity"]["status"] != "pass" or edit["replay_parity"]["status"] != "pass":
        raise ValueError("Both replay and cross-method teacher parity must pass")
    if (edit["replay_parity"].get("mode") != "exact_pool_bucket"
            or edit["replay_parity"].get("tolerance") != 1e-6
            or edit["replay_parity"].get("dataset_sha256") != dataset_sha
            or edit["teacher_action_parity"].get("tolerance") != .01):
        raise ValueError("Exact live-bucket replay and singleton teacher gates must use the declared contracts")
    if edit["teacher_action_parity"].get("dataset_sha256") != dataset_sha:
        raise ValueError("Teacher parity belongs to different LoRA targets")
    if fit["dataset_manifest_sha256"] != dataset_sha:
        raise ValueError("LoRA fit does not match frozen targets")
    if any(x["spec_sha256"] != sha(spec_path) for x in (edit, dataset)):
        raise ValueError("Methods do not match the current frozen development specification")
    if edit["source_observation_sha256"] != dataset["source_observation_sha256"] or dataset["source_observation_sha256"] != spec["source_observation_sha256"]:
        raise ValueError("Methods do not match the current frozen source observations")
    base = Path(edit["base_workspace"]).resolve()
    if Path(dataset["base_workspace"]).resolve() != base:
        raise ValueError("Methods identify different base workspaces")
    base_sha = sha(base / "workspace.json")
    if edit["base_manifest_sha256"] != base_sha or dataset["base_workspace_manifest_sha256"] != base_sha:
        raise ValueError("Base workspace identity differs from adaptation")
    recipe = protocol["adaptation"]
    lr = recipe["lora"]
    if (fit["rank"] != recipe["rank"] or fit["layers"] != recipe["layers"] or fit["steps"] != lr["steps"]
            or fit["learning_rate"] not in lr["learning_rate_candidates"] or fit["seed"] != lr["seed"] or fit["weight_decay"] != 0):
        raise ValueError("LoRA recipe differs from the declared primary comparison")
    cf = recipe["closed_form"]
    if (edit.get("rank_cap") != recipe["rank"] or edit.get("pooling") != "global"
            or edit.get("hold_weight") != cf["preservation_weight"] or edit.get("validation_used_in_fit") is not False
            or [r["layer"] for r in edit["layers"]] != recipe["layers"]
            or any(r["ridge_relative"] != cf["ridge_relative"] or r["rank"] > recipe["rank"] for r in edit["layers"])):
        raise ValueError("Closed-form recipe differs from the declared primary comparison")
    return dataset, fit, edit, base


def make(run_root, output):
    root, output = Path(run_root).resolve(), Path(output).resolve()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if sha(HERE / "rewrites.py") != protocol["router_sha256"]:
        raise ValueError("Rewrite rules changed since protocol declaration")
    if output.exists():
        raise FileExistsError(output)
    router = load_module("heldout_rewrite_router", HERE / "rewrites.py")
    frozen, cases = {}, {}
    common_base_sha = None
    for case in ["temporal", "spatial"]:
        adaptation = root / f"{case}-adaptation"
        dataset, fit, edit, base = validate_case(adaptation, case, protocol)
        selected = verify_selected_lora(adaptation, case)
        frozen.update(selected["artifacts"])
        base_manifest = json.loads((base / "workspace.json").read_text())
        if common_base_sha is not None and sha(base / "workspace.json") != common_base_sha:
            raise ValueError("The two cases do not share the same frozen base checkpoint identity")
        common_base_sha = sha(base / "workspace.json")
        seals_path = adaptation / "candidate-seals.json"
        seals = json.loads(seals_path.read_text())
        if set(seals) != {"closed_form", "lora"}:
            raise ValueError("Both candidate stage seals are required")
        candidates = {"closed_form": adaptation / "edit-fit/workspace",
                      "lora": adaptation / "lora-selected/workspace"}
        for method, workspace in candidates.items():
            manifest = json.loads((workspace / "workspace.json").read_text())
            if set(manifest["tensors"]) != LAYER_NAMES or manifest["base_vlm_sha256"] != base_manifest["base_vlm_sha256"]:
                raise ValueError("Candidate modules or base identity differ")
            tensors = sorted((workspace / "edits").glob("*.npy"))
            expected = {entry["file"] for entry in manifest["tensors"].values()}
            if len(tensors) != 9 or {p.name for p in tensors} != expected:
                raise ValueError("Expected nine edited MLP output projections")
            artifacts = {str(p): sha(p) for p in [workspace / "workspace.json", *tensors]}
            if method == "closed_form":
                if seals[method].get("artifacts") != artifacts:
                    raise ValueError("closed_form candidate differs from its immediate post-fit stage seal")
                if seals[method].get("sealed_after_stage") != "edit-solve":
                    raise ValueError("Candidate seal names the wrong adaptation stage")
                stage_log = adaptation / "edit-solve.log"
                if seals[method].get("stage_log_sha256") != sha(stage_log):
                    raise ValueError("Candidate stage log differs from its seal")
                frozen[str(stage_log)] = sha(stage_log)
                for row in edit["layers"]:
                    path = workspace / "edits" / manifest["tensors"][row["tensor"]]["file"]
                    if sha(path) != row["edited_sha256"]:
                        raise ValueError("Closed-form candidate differs from solve receipt")
            for path in [workspace / "workspace.json", *tensors]:
                frozen[str(path)] = sha(path)
        for name in sorted(LAYER_NAMES):
            path = base / "base" / base_manifest["tensors"][name]["file"]
            frozen[str(path)] = sha(path)
        for path in [seals_path, adaptation / "status.json", adaptation / "spec.json", adaptation / "lora-dataset/dataset.json",
                     adaptation / "lora-selected/training.json", adaptation / "edit-fit/solve.json"]:
            frozen[str(path)] = sha(path)
        batches, dispatches = [], []
        for i, cell in enumerate(protocol["cases"][case]["cells"]):
            dispatch_record = {"cell": cell["id"]}
            for mode, key in (("semantic", "rewrite"), ("exact", "exact_rewrite")):
                start = time.perf_counter_ns()
                dispatch = router.route(cell["prompt"], mode=mode)
                seconds = (time.perf_counter_ns() - start) / 1e9
                if dispatch != cell[key]:
                    raise ValueError(f"Stored {mode} rewrite differs from raw-prompt dispatch for {case}/{cell['id']}")
                dispatch_record[mode] = {"result": dispatch, "cpu_wall_seconds": seconds}
            dispatches.append(dispatch_record)
            conditions = protocol["conditions"]
            # Rotate order to reduce fixed condition/order confounding.
            conditions = conditions[i % 4:] + conditions[:i % 4]
            for condition in conditions:
                batches.append({"cell": cell["id"], "condition": condition, "task": cell["task"],
                                "goal": cell["goal"], "input_prompt": cell["prompt"],
                                "effective_prompt": dispatch_record["semantic"]["result"]["effective_prompt"] if condition == "saved_rewrite" else cell["prompt"],
                                "workspace": str(candidates[condition]) if condition in candidates else None,
                                "layouts": str(HERE / "runtime/layouts" / f"{case}_heldout.json"),
                                "slot_start": protocol["heldout"]["slot_start"][case], "n": 16})
        cases[case] = {"cohort_id": "rss-repair-comparison-20261117-heldout-v1", "split": "heldout",
                       "case": case, "batches": batches, "protocol_sha256": sha(HERE / "protocol.json"),
                       "inference_batching": protocol["heldout"]["inference_batching"],
                       "lora_selection": {"candidate_id": selected["selection"]["selected_candidate_id"],
                                          "learning_rate": selected["selection"]["selected_learning_rate"],
                                          "selection_sha256": sha(adaptation / "lora-selected/selection.json")},
                       "rewrite_dispatch": {"records": dispatches, "calls": 2 * len(dispatches),
                                            "cpu_wall_seconds": sum(d[m]["cpu_wall_seconds"] for d in dispatches for m in ("semantic", "exact")),
                                            "note": "One CPU dispatch per raw prompt and mode at plan creation; lightweight timing, not a latency benchmark."}}
    for path in runtime_sources():
        frozen[str(path)] = sha(path)
    output.mkdir(parents=True)
    freeze = output / "freeze.json"
    freeze.write_text(json.dumps({"artifacts": frozen, "plan_sha256": {case: plan_payload_sha256(plan) for case, plan in cases.items()},
                                  "note": "Both complete case-plan payloads and all candidates frozen before first heldout call; plan digest excludes only global_freeze pointer."}, indent=2) + "\n")
    for case, plan in cases.items():
        plan["global_freeze"] = {"path": str(freeze), "sha256": sha(freeze)}
        (output / f"{case}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(output), "planned_episodes": 640, "candidate_tensors": 36}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    make(a.run_root, a.output)
