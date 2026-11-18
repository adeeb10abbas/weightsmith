"""Verify the complete 384-trial target ablation before exposing outcomes.

No partial-outcome mode: both worker completion gates are checked first.
Raw observations, metric arrays, candidate seals and runtime identities are
verified. All registered cells and contrasts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np

from pipeline import HERE, PRIOR, METHODS, dump, read, require, sha, verify_map, module
from plan import check_plan


def interval(values, layouts, draws=20000, seed=17):
    groups = defaultdict(list)
    for value, layout in zip(values, layouts, strict=True):
        groups[layout].append(float(value))
    require(len(groups) == 8 and len(set(map(len, groups.values()))) == 1,
            "Expected eight equally sized layout clusters")
    means = np.asarray([np.mean(groups[k]) for k in sorted(groups)])
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, 8, size=(draws, 8))].mean(1)
    lo, hi = np.quantile(sampled, [.025, .975])
    return {"ci_low": float(lo), "ci_high": float(hi), "layout_clusters": 8,
            "degenerate_ci": bool(np.isclose(lo, hi, rtol=0, atol=1e-12)),
            "interpretation": "Exploratory percentile interval over eight layouts; a degenerate interval is not certainty or equivalence."}


def validate_row(row, cell, case, condition, protocol, layouts, metrics, obs):
    require((row["case"], row["cell"], row["condition"], row["cohort_id"], row["split"]) ==
            (case, cell["id"], condition, protocol["cohort"], "heldout"), "Episode identity differs")
    require(row["inference_batching"] == "fixed16" and row["ok"] is True, "Invalid inference/episode")
    require(row["input_prompt"] == row["effective_prompt"] == cell["prompt"], "Prompt differs")
    require(row["task_name"] == cell["task"], "Task differs")
    require(row["selected_goal"] == metrics["selected_goal"] == cell["goal"], "Selected goal differs")
    env = row["env"]
    require(env == row["slot"] % 16 and row["group"] == row["slot"] // 16, "Environment/slot differs")
    expected_layout = layouts["layouts"][row["slot"] % 8]["id"]
    require(row["layout_id"] == metrics["layout"]["layout_ids"][env] == expected_layout, "Layout differs")
    for kind in ("strict_final", "strict_ever", "stock_final", "stock_ever"):
        values = {key: bool(value[env]) for key, value in metrics[kind].items()}
        require(row[kind] == values and all(type(v) is bool for v in row[kind].values()),
                "Episode predicates differ from raw metric arrays")
    require(type(row["success"]) is bool and row["success"] == row["strict_final"][cell["goal"]],
            "Success differs from selected strict predicate")
    require(row["policy_seed"] == int(obs["seed"][0]) == ((row["slot"] // 16) * 64 + env) * 100,
            "Policy-noise stream differs")
    require(row["decision_count"] == len(obs["seed"]) > 0, "Decision count differs")
    for camera in ("base_img", "wrist_img"):
        require(row["first_camera_sha256"][camera] == hashlib.sha256(obs[camera][0].tobytes()).hexdigest(),
                "First-camera identity differs")


def collect(run_root):
    run_root = Path(run_root).resolve()
    roots = {case: run_root / f"{case}-heldout-v1" for case in ("temporal", "spatial")}
    # Deliberately gate both cases before reading any outcome or receipt.
    for case, root in roots.items():
        status = read(root / "status.json")
        require(status["status"] == "complete" and status["completed_episodes"] == 192
                and status["completed_batches"] == 12, f"{case} is not complete; outcomes remain unopened")
    protocol = read(HERE / "protocol.json")
    require(protocol["heldout"]["planned_total"] == 384 and protocol["conditions"] == list(METHODS),
            "Unexpected protocol accounting")
    worker = module("target_ablation_verification_worker", PRIOR / "worker.py")
    rows, receipts, sources = [], [], {str(HERE / "protocol.json"): sha(HERE / "protocol.json")}
    metric_cache = {}
    for case, root in roots.items():
        plan, identity = read(root / "plan.json"), read(root / "identity.json")
        check_plan(plan, protocol)
        require(plan["protocol_sha256"] == sha(HERE / "protocol.json"), "Changed protocol")
        frozen = worker.verify_global_freeze(plan)
        require(all(identity.get(path) == value for path, value in frozen.items()), "Worker omitted frozen identity")
        verify_map(identity)
        sources.update(identity)
        for name in ("plan.json", "identity.json", "status.json"):
            sources[str(root / name)] = sha(root / name)
        expected_batches = set()
        cells = {cell["id"]: cell for cell in protocol["cases"][case]["cells"]}
        for number, spec in enumerate(plan["batches"]):
            cell, condition = spec["cell"], spec["condition"]
            batch = root / f"batch-{number:02d}-{cell}-{condition}"
            expected_batches.add(batch)
            receipt, episodes = read(batch / "completed.json"), read(batch / "episodes.json")
            require((receipt["batch"], receipt["cell"], receipt["condition"], receipt["n"]) ==
                    (number, cell, condition, 16), "Receipt identity differs")
            require(receipt["episodes_sha256"] == sha(batch / "episodes.json"), "Episode checksum differs")
            start = protocol["heldout"]["slot_start"][case]
            require(sorted(row["slot"] for row in episodes) == list(range(start, start + 16)), "Slots missing or duplicated")
            layout = read(spec["layouts"])
            require(layout["cohort"] == protocol["cohort"] and layout["case"] == case, "Wrong layout cohort")
            for row in episodes:
                for key in ("metrics", "obs"):
                    path = Path(row[key]).resolve()
                    require(path.is_relative_to(batch), "Episode artifact escapes its new batch directory")
                    require(sha(path) == row[key + "_sha256"], f"Raw {key} checksum differs")
                    sources[str(path)] = row[key + "_sha256"]
                if row["metrics"] not in metric_cache:
                    metric_cache[row["metrics"]] = read(row["metrics"])
                with np.load(row["obs"], allow_pickle=False) as observations:
                    validate_row(row, cells[cell], case, condition, protocol, layout,
                                 metric_cache[row["metrics"]], observations)
            require(receipt["successes"] == sum(row["success"] for row in episodes), "Receipt success differs")
            require(Counter(r["layout_id"] for r in episodes) == Counter({r["id"]: 2 for r in layout["layouts"]}),
                    "Expected two replicates per layout")
            rows.extend(episodes)
            receipts.append({"case": case, "path": str(batch / "completed.json"), **receipt})
            for name in ("completed.json", "episodes.json"):
                sources[str(batch / name)] = sha(batch / name)
        require(set(root.glob("batch-*")) == expected_batches, "Unexpected batch attempts require diagnosis")
    keys = [(r["case"], r["cell"], r["condition"], r["slot"]) for r in rows]
    require(len(rows) == len(set(keys)) == 384, "Expected 384 distinct new policy episodes")
    return protocol, rows, receipts, sources


def paired(candidate, reference):
    require(len(candidate) == len(reference) == 16, "Expected sixteen paired episodes")
    for a, b in zip(candidate, reference, strict=True):
        require((a["slot"], a["layout_id"], a["policy_seed"]) ==
                (b["slot"], b["layout_id"], b["policy_seed"]), "Paired identity differs")
    values = [int(a["success"]) - int(b["success"]) for a, b in zip(candidate, reference)]
    return {"n": 16, "fixed": sum(a["success"] and not b["success"] for a, b in zip(candidate, reference)),
            "broken": sum(b["success"] and not a["success"] for a, b in zip(candidate, reference)),
            "both_success": sum(a["success"] and b["success"] for a, b in zip(candidate, reference)),
            "both_failure": sum(not a["success"] and not b["success"] for a, b in zip(candidate, reference)),
            "success_rate_difference": float(np.mean(values)),
            **interval(values, [r["layout_id"] for r in candidate])}


def summarize(protocol, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["cell"], row["condition"]].append(row)
    for value in groups.values():
        value.sort(key=lambda r: r["slot"])
    primary, comparisons, collisions, cameras = [], [], [], []
    for case, definition in protocol["cases"].items():
        for cell in definition["cells"]:
            cid = cell["id"]
            for condition in METHODS:
                group = groups[case, cid, condition]
                require(len(group) == 16, "Missing registered outcomes")
                primary.append({"case": case, "cell": cid, "condition": condition,
                    "selected_goal": cell["goal"], "role": cell.get("role", "target_repair"),
                    "successes": sum(r["success"] for r in group), "n": 16,
                    "success_rate": float(np.mean([r["success"] for r in group])),
                    "stock_compatible_successes": sum(r["stock_final"][cell["goal"]] for r in group),
                    **interval([r["success"] for r in group], [r["layout_id"] for r in group])})
                for goal in group[0]["strict_final"]:
                    if goal == cell["goal"]:
                        continue
                    collisions.append({"case": case, "cell": cid, "condition": condition, "competing_goal": goal,
                        "competing_final": sum(r["strict_final"][goal] for r in group),
                        "competing_only_final": sum(r["strict_final"][goal] and not r["success"] for r in group),
                        "selected_and_competing_final": sum(r["strict_final"][goal] and r["success"] for r in group),
                        "competing_ever": sum(r["strict_ever"][goal] for r in group), "n": 16,
                        "ever_note": "Ever may occur at another time or as a prerequisite; not a simultaneous collision."})
            for condition, reference in protocol["reporting"]["comparisons"]:
                a, b = groups[case, cid, condition], groups[case, cid, reference]
                comparisons.append({"case": case, "cell": cid, "condition": condition, "reference": reference,
                                    **paired(a, b)})
                for ar, br in zip(a, b):
                    cameras.append({"case": case, "cell": cid, "condition": condition, "reference": reference,
                        "slot": ar["slot"], "layout_id": ar["layout_id"], "excluded": False,
                        "base_camera_identical": ar["first_camera_sha256"]["base_img"] == br["first_camera_sha256"]["base_img"],
                        "wrist_camera_identical": ar["first_camera_sha256"]["wrist_img"] == br["first_camera_sha256"]["wrist_img"]})
    require(len(primary) == 24 and len(comparisons) == 30, "Incomplete registered table")
    return {"primary": primary, "paired": comparisons, "goal_diagnostics": collisions, "camera_pairing": cameras,
            "camera_note": "Paired intended layouts and noise streams. Reset/render discrepancies retained without exclusions.",
            "interpretation": protocol["reporting"]}


def costs(run_root, receipts):
    constructions = []
    for case in ("temporal", "spatial"):
        root = Path(run_root) / f"{case}-construction-v1"
        status = read(root / "status.json")
        norms = {method: read(root / method / "fit.json")["layers"] for method in METHODS if method != "base"}
        constructions.append({"case": case, "whole_cpu_construction_seconds": status["wall_seconds"],
                              "layer_fit_export_seconds": {m: sum(r["wall_seconds"] for r in v) for m, v in norms.items()},
                              "target_and_update_norms": norms})
    return {"new_test_episodes": 384, "new_development_policy_episodes": 0,
            "reused_development": "Same sealed 12 repair and 12 preservation decisions per case; eight validation decisions excluded from fitting. Acquisition and prior qualification costs are reused, not zero.",
            "cpu_constructions": constructions,
            "sum_batch_execution_seconds": sum(r["execution_seconds"] for r in receipts),
            "sum_batch_queue_seconds": sum(r["queue_seconds"] for r in receipts),
            "sum_batch_wrapper_wall_seconds": sum(r["wall_seconds"] for r in receipts),
            "note": "Sums are lane work and queue accounting, not elapsed parallel wall time; fitting subtimes exclude loading and target construction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an analysis attempt")
    protocol, rows, receipts, sources = collect(args.run_root)
    report = {"mode": "complete_target_ablation", "complete_and_verified": True,
              "cohort": protocol["cohort"], "new_policy_episodes": len(rows),
              "verification": "All384 receipts, raw observation hashes, metric arrays, cameras, noise streams, plans and frozen source/candidate bytes checked.",
              **summarize(protocol, rows), "costs": costs(args.run_root, receipts), "source_sha256": sources}
    args.output.mkdir(parents=True)
    dump(args.output / "report.json", report)
    dump(args.output / "episodes.json", rows)
    dump(args.output / "receipts.json", receipts)
    for key in ("primary", "paired", "goal_diagnostics", "camera_pairing"):
        with (args.output / f"{key}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[key][0]))
            writer.writeheader()
            writer.writerows(report[key])
    print(f"Verified384/384 new policy episodes; report written to {args.output}")


if __name__ == "__main__":
    main()
