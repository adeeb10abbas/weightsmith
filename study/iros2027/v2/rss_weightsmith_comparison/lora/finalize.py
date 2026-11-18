"""Seal all three declared LoRA candidates for external development selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

spec = importlib.util.spec_from_file_location("historical_lora", Path(__file__).with_name("historical_lora.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    from weightsmith.edits import fingerprint
    root = Path(args.project).resolve(); adaptation = root / "lora-adaptation"
    registry = adaptation / "candidates.json"
    if registry.exists(): raise FileExistsError(registry)
    dataset = root / "lora-preparation"
    source = json.loads((dataset / "source-seal.json").read_text())
    H.verify_artifacts(source["artifacts"])
    meta, _ = H.load_dataset(dataset)
    audit = json.loads((adaptation / "token-correspondence.json").read_text())
    if audit["status"] != "pass" or len(audit["rows"]) != 12:
        raise ValueError("Token correspondence audit did not pass")
    common_files = [root / "lora-source.json", dataset / "source-seal.json", dataset / "dataset.json",
                    root / "lora-smoke/smoke.json", adaptation / "numerical-boundary.json",
                    adaptation / "token-correspondence.json", H.HERE / "audit_tokens.py", Path(__file__)]
    common_files.extend(dataset / r["file"] for r in meta["records"])
    common = {**source["artifacts"], **{str(p.resolve()): H.sha(p) for p in common_files}}
    candidates = []
    for name, rate in [("lr-3e-5", .00003), ("lr-1e-4", .0001), ("lr-1e-3", .001)]:
        path = adaptation / name
        report = json.loads((path / "training.json").read_text())
        seal = json.loads((path / "seal.json").read_text())
        progress = json.loads((path / "progress.json").read_text())
        expected = {"split": "development", "rank": 72, "layers": list(range(17)), "steps": 200,
                    "learning_rate": rate, "seed": 17, "train_inputs": 6, "validation_inputs": 0,
                    "trainable_parameters": 22560768, "zero_start_exact": True,
                    "dataset_sha256": H.sha(dataset / "dataset.json"), "source_code_sha256": H.sha(H.__file__),
                    "utility_sha256": H.sha(H.UTILITY), "closed_loop_evaluation": "not performed"}
        if any(report[k] != value for k, value in expected.items()) or progress["stage"] != "complete":
            raise ValueError(f"Incomplete or nonregistered candidate: {name}")
        if report["history"][-1]["step"] != 200 or not all(np.isfinite(r["flow_loss"]+r["gradient_norm"]) for r in report["history"]):
            raise ValueError(f"Invalid optimization history: {name}")
        H.verify_artifacts(seal["artifacts"])
        workspace = path / "workspace"
        if len(list((workspace / "edits").glob("*.npy"))) != 17:
            raise ValueError(f"Wrong module count: {name}")
        candidates.append({"candidate_id": seal["candidate_id"], "learning_rate": rate,
                           "workspace": str(workspace), "fingerprint": fingerprint(workspace / "edits"),
                           "training_path": str(path / "training.json"), "seal_path": str(path / "seal.json"),
                           "recorded_fitting_wall_seconds": report["wall_seconds"],
                           "optimization_memory_stats": report["memory_stats"],
                           "artifacts": {**common, **seal["artifacts"], str(path / "seal.json"): H.sha(path / "seal.json")}})
    H.dump(registry, {"cohort_id": H.COHORT, "split": "development", "rank": 72, "layers": list(range(17)),
                      "steps": 200, "source_observations": source["items"], "candidates": candidates,
                      "selected_candidate": None, "selection": "Fresh development original-request success; tie chooses smaller learning rate; no fitting-metric selection",
                      "source_config_seal": {"path": str(root / "lora-source.json"), "sha256": H.sha(root / "lora-source.json")},
                      "numerical_boundary": {"path": str(adaptation / "numerical-boundary.json"), "sha256": H.sha(adaptation / "numerical-boundary.json")},
                      "token_correspondence": {"path": str(adaptation / "token-correspondence.json"), "sha256": H.sha(adaptation / "token-correspondence.json")},
                      "cost_scope": {"new_policy_episodes": 0, "optimizer_steps": 600, "additional_smoke_steps": 1,
                                     "source_observations": 6, "new_native_replay_calls": 12,
                                     "source_preparation_recorded_wall_seconds": meta["wall_seconds"],
                                     "fitting_recorded_wall_seconds": sum(c["recorded_fitting_wall_seconds"] for c in candidates),
                                     "note": "Recorded stage intervals are not GPU compute or complete historical acquisition/search cost. Memory was sampled after optimization, before fitting diagnostics/export. Separate development selection adds policy trials."}})
    status = {"status": "complete", "stage": "complete", "completed_candidates": [c["candidate_id"] for c in candidates],
              "current_candidate": None, "updated_unix": time.time(), "registry_path": str(registry),
              "registry_sha256": H.sha(registry), "gpu": 0, "selection_performed": False}
    H.dump(adaptation / "status.json", status)
    print(json.dumps({"registry": str(registry), "sha256": H.sha(registry),
                      "fingerprints": {c["candidate_id"]: c["fingerprint"] for c in candidates}}), flush=True)


if __name__ == "__main__": main()
