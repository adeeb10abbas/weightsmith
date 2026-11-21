#!/usr/bin/env python3
"""CPU-only fit of fixed deployed updates on included and excluded source states.

This is a local linear diagnostic at frozen original activations. It does not
run an edited network, refit an update, or measure robot behavior.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np

import update_agreement as U

REPO = U.REPO
STUDY = REPO / "study/iros2027/v2/rss_few_pair_validation_20261120"
sys.path[:0] = [str(REPO), str(STUDY)]
from adaptation import recipe
from weightsmith import distill

OUTPUT = REPO / "projects/rss-few-pair-recovery-20261120/diagnostics/local-activation-fit"


def metrics(error_squared: float, target_squared: float) -> dict:
    if not math.isfinite(error_squared) or not math.isfinite(target_squared):
        raise ValueError("Nonfinite activation-fit metric")
    if error_squared < 0 or target_squared <= 0:
        raise ValueError("Invalid error or zero target energy")
    ratio_squared = error_squared / target_squared
    return {"error_squared": error_squared, "target_squared": target_squared,
            "relative_error": math.sqrt(ratio_squared),
            "target_energy_reduction": 1.0 - ratio_squared}


def local_systems(original: dict, teacher: dict, slots: list, layer: int):
    """Use the original construction function on one state at a time."""
    systems = []
    for i in range(original[recipe.ARRAYS[0]].shape[0]):
        systems.append(distill.build_keys_values(
            {key: value[i:i+1] for key, value in original.items()},
            {key: value[i:i+1] for key, value in teacher.items()},
            layers=[layer], mode="both", alpha=1.0, slot_map=slots,
            cells=distill.IMAGE_CELLS, capture_layers=recipe.LAYERS)[layer])
    return np.stack([k for k, _ in systems]), np.stack([v for _, v in systems])


def check_pooled_identity(keys, values, indices, expected):
    pooled_keys = keys[indices].mean(axis=0)
    pooled_values = values[indices].mean(axis=0)
    key_error = float(np.max(np.abs(pooled_keys - expected[0])))
    value_error = float(np.max(np.abs(pooled_values - expected[1])))
    # The only permitted difference is float64 reassociation of mean and
    # subtraction for values. Keys use exactly the same ordered mean.
    if key_error != 0.0 or value_error > 1e-12:
        raise ValueError(f"Pooled construction identity failed: {key_error}, {value_error}")
    return {"keys_max_abs_error": key_error, "values_max_abs_error": value_error,
            "values_tolerance": 1e-12}


def per_state_metrics(prediction, targets):
    if prediction.shape != targets.shape or prediction.ndim != 3:
        raise ValueError("Expected equal [state, row, output] arrays")
    return [metrics(float(np.sum((p-v)**2)), float(np.sum(v**2)))
            for p, v in zip(prediction, targets, strict=True)]


def aggregate(rows):
    return metrics(sum(r["error_squared"] for r in rows),
                   sum(r["target_squared"] for r in rows))


def load_captures(item, pins):
    registry = U.read(item["registry"])
    indices, arrays = {}, {}
    for kind, directory in registry["captures"].items():
        directory = Path(directory)
        indices[kind] = U.read(directory / "index.json")
        with np.load(directory / "acts.npz", allow_pickle=False) as data:
            arrays[kind] = {key: data[key] for key in recipe.ARRAYS}
        for value in arrays[kind].values():
            if value.shape[:2] != (9, 17) or not np.isfinite(value).all():
                raise ValueError("Expected nine finite captured states and 17 layers")
    slots, alignment = recipe.capture_alignment(indices["original"], indices["teacher"], expected_count=9)
    parity = U.read(Path(registry["gate"]["native_and_instrumented_parity"]))
    if parity["status"] != "pass" or list(map(list, slots)) != parity["slot_map"]:
        raise ValueError("Capture alignment differs from the original construction gate")
    manifest = U.read(Path(registry["source_manifest"]))
    pair_ids = [row["pair_id"] for row in manifest["pairs"]]
    spec = U.read(Path(manifest["spec"]))
    U.add_pin(pins, manifest["spec"])
    for i, row in enumerate(spec["items"]):
        if row["pair_id"] != pair_ids[i]:
            raise ValueError("Source/capture ordering differs")
        for kind in ("original", "teacher"):
            captured = indices[kind]["items"][i]
            if captured["episode"] != f"item-{i:04d}" or captured["k"] != row["decision_index"]:
                raise ValueError("Captured decision differs from source manifest")
    return arrays, slots, alignment, pair_ids


def compute_goal(goal, item, pins):
    arrays, slots, alignment, pair_ids = load_captures(item, pins)
    entries = item["entries"]
    original, teacher = arrays["original"], arrays["teacher"]
    # Reuse the exact construction wrapper. Its results are verification
    # references only; no solver or merge function is called.
    pooled = {g: recipe.pooled_key_values(original, teacher,
              entries[g]["candidate"]["source_indices"], slots) for g in U.GROUPS}
    rows, pooled_rows, identity_rows = [], [], []
    for layer in recipe.LAYERS:
        keys, targets = local_systems(original, teacher, slots, layer)
        if keys.shape[:2] != (9, 72) or targets.shape[:2] != (9, 72):
            raise ValueError("Expected all nine states and the historical 72 rows")
        base = np.load(entries["g0"]["paths"][layer]["base"], mmap_mode="r", allow_pickle=False)
        for group in U.GROUPS:
            entry = entries[group]
            included = entry["candidate"]["source_indices"]
            identity_rows.append({"goal": goal, "group": group, "layer": layer,
                **check_pooled_identity(keys, targets, included, pooled[group][layer])})
            path = entry["paths"][layer]
            edited = np.load(path["edited"], mmap_mode="r", allow_pickle=False)
            if base.dtype != np.float32 or edited.dtype != np.float32 or edited.shape != base.shape:
                raise ValueError("Expected effective BF16 deployed weights stored as float32")
            delta = np.subtract(edited, base, dtype=np.float64)
            prediction = (keys.reshape(-1, keys.shape[-1]) @ delta).reshape(targets.shape)
            values = per_state_metrics(prediction, targets)
            for state, value in enumerate(values):
                rows.append({"goal": goal, "group": group, "layer": layer,
                    "state_index": state, "pair_id": pair_ids[state],
                    "split": "included_three" if state in included else "excluded_six", **value})
            k, v = pooled[group][layer]
            # This reproduces the mean-state construction objective after
            # deployment rounding. The receipt uses the pre-rounding solve.
            pooled_rows.append({"goal": goal, "group": group, "layer": layer,
                "construction_pre_rounding_relative_error": entry["candidate"]["layers"][layer]["fit_relative_error"],
                **metrics(float(np.sum((k @ delta-v)**2)), float(np.sum(v**2)))})
            del edited, delta, prediction
        del base, keys, targets
        print(json.dumps({"goal": goal, "completed_layer": layer}), flush=True)
    overall, states, layers = [], [], []
    for group in U.GROUPS:
        subset = [r for r in rows if r["group"] == group]
        for split in ("included_three", "excluded_six"):
            selected = [r for r in subset if r["split"] == split]
            overall.append({"goal": goal, "group": group, "split": split,
                "states": 3 if split == "included_three" else 6, "layers": 17,
                **aggregate(selected)})
            for layer in recipe.LAYERS:
                layers.append({"goal": goal, "group": group, "split": split,
                    "layer": layer, **aggregate([r for r in selected if r["layer"] == layer])})
        for i in range(9):
            selected = [r for r in subset if r["state_index"] == i]
            states.append({"goal": goal, "group": group, "state_index": i,
                "pair_id": pair_ids[i], "split": selected[0]["split"], **aggregate(selected)})
    return {"overall": overall, "per_state": states, "per_layer": layers,
            "per_state_layer": rows, "pooled_construction_fit": pooled_rows,
            "pooled_identity": identity_rows, "alignment": alignment,
            "pair_ids": pair_ids, "source_groups": {
                g: entries[g]["candidate"]["source_indices"] for g in U.GROUPS}}


def main():
    start = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    pins, goals = U.prepare()
    for path in (Path(__file__), Path(U.__file__), Path(recipe.__file__), Path(distill.__file__)):
        U.add_pin(pins, path.resolve())
    for item in goals.values():
        manifest = U.read(Path(U.read(item["registry"])["source_manifest"]))
        U.add_pin(pins, manifest["spec"])
    definition = {
        "status": "frozen_before_computation", "goals": list(U.GOALS), "groups": list(U.GROUPS),
        "layers": list(recipe.LAYERS), "rows": "40 original aligned prompt/template/state positions plus 32 image cells",
        "primary_metric": "sqrt(sum_states,layers ||K_i,l (edited_l-base_l)-V_i,l||_F^2 / sum_states,layers ||V_i,l||_F^2)",
        "target": "V_i,l = (teacher-original residual output)_i,l - (teacher-original residual output)_i,l-1; previous gap zero at layer 0",
        "null_baseline": "Zero weight update has relative_error 1 and target_energy_reduction 0",
        "scope": "Local one-step fit at frozen original keys; no edited-network forward propagation or behavioral generalization",
        "groups_compared": "Own three included construction states versus the other six qualified source states from the same goal and instruction pair",
        "nonindependence": "Excluded states are reused across edits; the six edits and all layer/state rows are not independent statistical replications",
        "precision": "Float64 products of effective BF16 deployed weights; construction receipt error is pre-rounding and shown separately",
        "new_policy_episodes": 0, "gpu_used": False, "max_blas_threads": 4,
        "refitting": False, "selection": False, "artifacts": pins}
    U.write_json(OUTPUT / "definition.json", definition)
    U.verify(pins)
    print(json.dumps({"preflight_hashes_verified": len(pins)}), flush=True)
    results = {goal: compute_goal(goal, item, pins) for goal, item in goals.items()}
    U.verify(pins)
    report = {**definition, "status": "complete_and_verified", "artifacts": pins,
              "hashes_verified_before_and_after": True, "results": results,
              "wall_seconds": time.time()-start}
    paths = []
    for key in ("overall", "per_state", "per_layer", "per_state_layer",
                "pooled_construction_fit", "pooled_identity"):
        path = OUTPUT / (key.replace("_", "-") + ".csv")
        rows = [row for goal in U.GOALS for row in results[goal][key]]
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        paths.append(path)
    U.write_json(OUTPUT / "report.json", report)
    paths.extend([OUTPUT / "report.json", OUTPUT / "definition.json"])
    U.write_json(OUTPUT / "receipt.json", {"status": "complete_and_verified",
        "new_policy_episodes": 0, "gpu_used": False,
        "artifacts": {str(p): U.sha(p) for p in paths}})
    print(json.dumps({"overall": [r for goal in U.GOALS for r in results[goal]["overall"]],
                      "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
