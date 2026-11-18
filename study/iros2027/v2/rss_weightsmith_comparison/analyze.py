"""Verify complete comparison receipts, then report all conditions and contrasts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from common import HERE, PRIOR, ROOT, load, protocol, read, save, sha, validate_plan


def collect(root, expected_split):
    root = Path(root); p = protocol()
    plan = read(root / "plan.json"); validate_plan(plan)
    if plan["split"] != expected_split:
        raise ValueError("Wrong evaluation split")
    worker = load("weightsmith_comparison_receipt_worker", PRIOR / "worker.py")
    worker.verify_global_freeze(plan)
    state = read(root / "status.json")
    total = 48 if expected_split == "development" else 320
    if (state["status"] != "complete" or state["completed_episodes"] != total
            or state["completed_batches"] != len(plan["batches"])):
        raise ValueError("Every declared batch must complete before selection or outcome analysis")
    identity = read(root / "identity.json")
    for path, checksum in identity.items():
        if sha(path) != checksum:
            raise ValueError(f"Execution identity changed: {path}")
    groups, receipts, artifacts, checked_metrics = defaultdict(list), [], {}, {}
    for number, b in enumerate(plan["batches"]):
        path = root / f"batch-{number:02d}-{b['cell']}-{b['condition']}"
        receipt = read(path / "completed.json"); rows = read(path / "episodes.json")
        if (receipt["episodes_sha256"] != sha(path / "episodes.json")
                or (receipt["batch"], receipt["cell"], receipt["condition"], receipt["n"]) != (number, b["cell"], b["condition"], 16)
                or sorted(r["slot"] for r in rows) != list(range(b["slot_start"], b["slot_start"] + 16))
                or len(list(path.glob("sim-*.log"))) != 1):
            raise ValueError("Completed batch identity, count, or simulator-attempt contract differs")
        server = json.loads(receipt["server"])
        if server.get("workspace") != b["workspace"]:
            raise ValueError("Server loaded a different policy workspace")
        for r in rows:
            if (r["cohort_id"], r["case"], r["split"], r["condition"], r["cell"], r["input_prompt"], r["effective_prompt"]) != (
                    p["cohort_id"], "temporal", expected_split, b["condition"], b["cell"], b["input_prompt"], b["effective_prompt"]):
                raise ValueError("Episode identity differs from sealed plan")
            if (not r["ok"] or r["selected_goal"] != p["goal"] or r["inference_batching"] != "fixed16"
                    or r["success"] != r["strict_final"][p["goal"]]):
                raise ValueError("Primary scoring or execution differs from protocol")
            if sha(r["obs"]) != r["obs_sha256"]:
                raise ValueError("Observation changed")
            artifacts[r["obs"]] = r["obs_sha256"]
            if r["metrics"] not in checked_metrics:
                checked_metrics[r["metrics"]] = sha(r["metrics"])
            if checked_metrics[r["metrics"]] != r["metrics_sha256"]:
                raise ValueError("Scoring metrics changed")
            metrics = read(r["metrics"])
            if metrics["reset_calls"] != 2 or metrics["layout"]["sha256"] != sha(b["layouts"]):
                raise ValueError("Reset receipt/layout identity mismatch")
            layouts = read(b["layouts"])["layouts"]
            if r["layout_id"] != layouts[r["slot"] % len(layouts)]["id"]:
                raise ValueError("Recorded layout differs from slot assignment")
            groups[b["condition"]].append(r)
        if sum(r["success"] for r in rows) != receipt["successes"]:
            raise ValueError("Receipt successes do not match episodes")
        receipts.append(receipt)
        for source in (path / "episodes.json", path / "completed.json"):
            artifacts[str(source)] = sha(source)
    n = 16 if expected_split == "development" else 64
    clusters = 8 if expected_split == "development" else 16
    replicates = 2 if expected_split == "development" else 4
    for rows in groups.values():
        rows.sort(key=lambda r: r["slot"])
        if len(rows) != n or len({r["slot"] for r in rows}) != n:
            raise ValueError("Missing or duplicate condition trials")
        count = Counter(r["layout_id"] for r in rows)
        if len(count) != clusters or set(count.values()) != {replicates}:
            raise ValueError("Expected balanced layout/noise design")
    artifacts.update(checked_metrics)
    for source in (root / "plan.json", root / "status.json", root / "identity.json", Path(plan["global_freeze"]["path"])):
        artifacts[str(source)] = sha(source)
    return plan, dict(groups), receipts, artifacts


def clustered_interval(values, layouts, *, clusters=16, replicates=4, draws=20000):
    groups = defaultdict(list)
    for value, layout in zip(values, layouts, strict=True):
        groups[layout].append(float(value))
    if len(groups) != clusters or set(map(len, groups.values())) != {replicates}:
        raise ValueError("Clustered interval requires the declared balanced layout design")
    means = np.array([np.mean(groups[key]) for key in sorted(groups)])
    selected = np.random.default_rng(17).integers(0, clusters, size=(draws, clusters))
    low, high = np.quantile(means[selected].mean(axis=1), [.025, .975])
    return {"ci_low": float(low), "ci_high": float(high), "layout_clusters": clusters,
            "degenerate_ci": bool(np.isclose(low, high, rtol=0, atol=1e-12))}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)


def summarize():
    p = protocol(); output = ROOT / "analysis-v1"
    if output.exists():
        raise FileExistsError(output)
    plan, groups, receipts, artifacts = collect(ROOT / "evaluation", "heldout")
    if set(groups) != set(p["conditions"]):
        raise ValueError("Missing comparison arms")
    primary, secondary, layouts, pairs, camera_rows = [], [], [], [], []
    for condition in p["conditions"]:
        rows = groups[condition]; values = [r["success"] for r in rows]; ids = [r["layout_id"] for r in rows]
        primary.append({"condition": condition, "successes": sum(values), "n": 64, "success_rate": sum(values) / 64,
                        **clustered_interval(values, ids)})
        secondary.append({"condition": condition, "n": 64,
                          "stock_final_successes": sum(r["stock_final"][p["goal"]] for r in rows),
                          "stock_ever_successes": sum(r["stock_ever"][p["goal"]] for r in rows)})
        for layout in sorted(set(ids)):
            batch = [r for r in rows if r["layout_id"] == layout]
            layouts.append({"condition": condition, "layout_id": layout, "successes": sum(r["success"] for r in batch), "n": 4})
    # Primary contrast first, followed by all remaining condition pairs.
    contrasts = [("weightsmith", "teacher_action_lora"), ("weightsmith", "base"),
                 ("weightsmith", "matched_random"), ("weightsmith", "clarified_input")]
    seen = {frozenset(pair) for pair in contrasts}
    contrasts += [(a, b) for a, b in itertools.combinations(p["conditions"], 2) if frozenset((a, b)) not in seen]
    for condition, reference in contrasts:
        a, b = groups[condition], groups[reference]
        for x, y in zip(a, b, strict=True):
            if (x["slot"], x["layout_id"], x["policy_seed"]) != (y["slot"], y["layout_id"], y["policy_seed"]):
                raise ValueError("Paired conditions differ in slot, layout, or noise identity")
            camera_rows.append({"condition": condition, "reference": reference, "slot": x["slot"],
                                "identical_first_cameras": x["first_camera_sha256"] == y["first_camera_sha256"], "excluded": False})
        differences = [int(x["success"]) - int(y["success"]) for x, y in zip(a, b)]
        pairs.append({"condition": condition, "reference": reference,
                      "primary_estimand": (condition, reference) == ("weightsmith", "teacher_action_lora"),
                      "n": 64, "fixed": sum(x["success"] and not y["success"] for x, y in zip(a, b)),
                      "broken": sum(y["success"] and not x["success"] for x, y in zip(a, b)),
                      "both_success": sum(x["success"] and y["success"] for x, y in zip(a, b)),
                      "both_failure": sum(not x["success"] and not y["success"] for x, y in zip(a, b)),
                      "success_rate_difference": float(np.mean(differences)),
                      **clustered_interval(differences, [r["layout_id"] for r in a])})
    costs = [{"split": "heldout", "condition": r["condition"], "batch": r["batch"], "policy_episodes": r["n"],
              "execution_seconds": r["execution_seconds"], "queue_seconds": r["queue_seconds"], "simulator_attempts": 1}
             for r in receipts]
    _, _, selection_receipts, _ = collect(ROOT / "selection", "development")
    costs += [{"split": "development", "condition": r["condition"], "batch": r["batch"], "policy_episodes": r["n"],
               "execution_seconds": r["execution_seconds"], "queue_seconds": r["queue_seconds"], "simulator_attempts": 1}
              for r in selection_receipts]
    registry = read(ROOT / "lora-adaptation/candidates.json")
    report = {"cohort_id": p["cohort_id"], "status": "complete_and_verified", "new_test_policy_episodes": 320,
              "new_selection_policy_episodes": 48, "reused_source_observations": 6, "historical_episodes_pooled": 0,
              "primary_estimand": p["primary_estimand"], "primary": primary, "secondary": secondary,
              "paired": pairs, "per_layout": layouts, "costs": costs, "selection": read(ROOT / "selection.json"),
              "training_reports": {c["candidate_id"]: read(c["training_path"]) for c in registry["candidates"]},
              "teacher_target_numerical_boundary": read(ROOT / "lora-adaptation/numerical-boundary.json"),
              "archived_capture_parity": read(ROOT / "design/sources.json")["archived_capture_parity"],
              "cost_boundaries": "All three LoRA training runs and selection trials count. Historical construction/search, human clarification, and decision selection are reused and not declared free. Wall components are not GPU-hours or an end-to-end speed claim.",
              "statistics": p["statistics"], "limitations": p["limitations"],
              "pairing_diagnostics": {"compared_pairs": len(camera_rows), "first_camera_mismatches": sum(not r["identical_first_cameras"] for r in camera_rows), "automatically_excluded_episodes": 0},
              "source_receipts": artifacts, "analysis_code_sha256": sha(__file__), "protocol_sha256": sha(HERE / "protocol.json")}
    output.mkdir()
    save(output / "report.json", report)
    for name, rows in (("primary", primary), ("secondary", secondary), ("paired", pairs), ("per_layout", layouts),
                       ("costs", costs), ("pairing_diagnostics", camera_rows)):
        write_csv(output / f"{name}.csv", rows)
    print(json.dumps({"report": str(output / "report.json"), "primary": primary, "primary_comparison": pairs[0]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    summarize()
