"""Verify all 96 completed recovery episodes and summarize every frozen arm."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / "projects/rss-historical-recovery-20261118"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("recovery_analysis_worker", HERE.parent / "rss_repair_comparison/worker.py")
stats = load("recovery_analysis_stats", HERE.parent / "rss_repair_comparison/analysis/report.py")


def main():
    plan = stats.read(ROOT / "freeze/temporal-plan.json")
    worker.verify_global_freeze(plan)
    evaluation = ROOT / "evaluation"
    state = stats.read(evaluation / "status.json")
    if state["status"] != "complete" or state["completed_episodes"] != 96 or state["completed_batches"] != 6:
        raise ValueError("All six arms must be complete; no partial outcome report")
    if stats.read(evaluation / "plan.json") != plan:
        raise ValueError("Executed plan differs from frozen plan")
    identity = stats.read(evaluation / "identity.json")
    for file, expected in identity.items():
        if stats.sha(file) != expected:
            raise ValueError(f"Frozen execution identity changed: {file}")
    conditions = [b["condition"] for b in plan["batches"]]
    groups, receipts, checked_metrics = {}, {}, {}
    all_files = {}
    for index, batch in enumerate(plan["batches"]):
        path = evaluation / f"batch-{index:02d}-{batch['cell']}-{batch['condition']}"
        receipt = stats.read(path / "completed.json")
        if receipt["episodes_sha256"] != stats.sha(path / "episodes.json"):
            raise ValueError("Episode receipt mismatch")
        if (receipt["batch"], receipt["condition"], receipt["n"]) != (index, batch["condition"], 16):
            raise ValueError("Completed batch identity differs from plan")
        rows = sorted(stats.read(path / "episodes.json"), key=lambda row: row["slot"])
        if [r["slot"] for r in rows] != list(range(batch["slot_start"], batch["slot_start"]+16)):
            raise ValueError("Missing or duplicated episode slots")
        if len(list(path.glob("sim-*.log"))) != 1:
            raise ValueError("Expected exactly one simulator attempt per arm")
        for row in rows:
            if (row["cohort_id"], row["split"], row["case"], row["condition"], row["input_prompt"], row["effective_prompt"]) != (
                    plan["cohort_id"], "development", "temporal", batch["condition"], batch["input_prompt"], batch["effective_prompt"]):
                raise ValueError("Episode identity or instruction differs from plan")
            if not row["ok"] or row["selected_goal"] != "cube_then_banana" or row["inference_batching"] != "fixed16":
                raise ValueError("Episode execution differs from protocol")
            if row["success"] != row["strict_final"]["cube_then_banana"]:
                raise ValueError("Primary outcome differs from strict predicate")
            if stats.sha(row["obs"]) != row["obs_sha256"]:
                raise ValueError("Raw observation artifact changed")
            metrics = row["metrics"]
            if metrics not in checked_metrics:
                checked_metrics[metrics] = stats.sha(metrics)
            if checked_metrics[metrics] != row["metrics_sha256"]:
                raise ValueError("Scoring metrics changed")
            all_files[row["obs"]] = row["obs_sha256"]
        if receipt["successes"] != sum(r["success"] for r in rows):
            raise ValueError("Receipt outcome count differs from episodes")
        groups[batch["condition"]] = rows
        receipts[batch["condition"]] = receipt
        for file in [path / "episodes.json", path / "completed.json"]:
            all_files[str(file)] = stats.sha(file)
    primary, secondary, paired, cameras = [], [], [], []
    for condition, rows in groups.items():
        successes = sum(r["success"] for r in rows)
        primary.append({"condition": condition, "successes": successes, "n": 16,
                        "success_rate": successes / 16,
                        **stats.clustered_interval([r["success"] for r in rows], [r["layout_id"] for r in rows])})
        secondary.append({"condition": condition, "n": 16,
                          "stock_final_successes": sum(r["stock_final"]["cube_then_banana"] for r in rows),
                          "stock_ever_successes": sum(r["stock_ever"]["cube_then_banana"] for r in rows),
                          "note": "Secondary unchanged stock-compatible scores on strict-runtime trajectories, not historical-runtime replication."})
    for reference, condition in itertools.combinations(conditions, 2):
        a, b = groups[condition], groups[reference]
        for x, y in zip(a, b, strict=True):
            if (x["slot"], x["layout_id"], x["policy_seed"]) != (y["slot"], y["layout_id"], y["policy_seed"]):
                raise ValueError("Paired episode identity mismatch")
            cameras.append({"condition": condition, "reference": reference, "slot": x["slot"],
                            "identical_first_cameras": x["first_camera_sha256"] == y["first_camera_sha256"], "excluded": False})
        diff = [int(x["success"])-int(y["success"]) for x, y in zip(a, b, strict=True)]
        paired.append({"condition": condition, "reference": reference, "n": 16,
                       "fixed": sum(x["success"] and not y["success"] for x, y in zip(a, b)),
                       "broken": sum(not x["success"] and y["success"] for x, y in zip(a, b)),
                       "both_success": sum(x["success"] and y["success"] for x, y in zip(a, b)),
                       "both_failure": sum(not x["success"] and not y["success"] for x, y in zip(a, b)),
                       "success_rate_difference": float(np.mean(diff)),
                       **stats.clustered_interval(diff, [r["layout_id"] for r in a])})
    report = {"cohort_id": plan["cohort_id"], "mode": "completed_development_artifact_recovery",
              "verified_policy_episodes": 96, "primary": primary, "secondary": secondary, "paired": paired,
              "pairing_diagnostics": {"comparisons": len(cameras),
                  "first_camera_mismatches": sum(not x["identical_first_cameras"] for x in cameras),
                  "automatically_excluded_episodes": 0},
              "costs": [{"condition": c, "execution_seconds": r["execution_seconds"],
                         "queue_seconds": r["queue_seconds"], "simulator_attempts": 1} for c, r in receipts.items()],
              "statistics": "Descriptive 95% layout-cluster percentile intervals, 20000 draws, seed17; eight clusters with two noise replicates retained together. No p values, equivalence or confirmatory claims.",
              "limitations": stats.read(HERE / "protocol.json")["limitations"],
              "method_comparison": "Candidates used different construction data, objectives and budgets. An artifact advantage does not establish algorithm superiority.",
              "analysis_code_sha256": stats.sha(__file__), "statistics_source_sha256": stats.sha(stats.__file__),
              "source_receipts": {**all_files, **checked_metrics}}
    output = ROOT / "analysis-v1"
    output.mkdir(exist_ok=False)
    stats.write(output / "report.json", report)
    for key in ["primary", "secondary", "paired", "costs"]:
        stats.write_csv(output / f"{key}.csv", report[key])
    stats.write_csv(output / "pairing_diagnostics.csv", cameras)
    print(json.dumps({"output": str(output), "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
