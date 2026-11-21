#!/usr/bin/env python3
"""Read-only comparison of every deployed source-group weight update.

This exploratory diagnostic measures agreement of updates, not policy efficacy.
No source group is selected and no policy, simulator, or GPU is loaded.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

# Set before importing numpy, even if the caller's environment requests more.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"
import numpy as np

REPO = Path(__file__).resolve().parents[5]
OLDROOT = REPO / "projects/rss-few-pair-validation-20261120"
OUTPUT = REPO / "projects/rss-few-pair-recovery-20261120/diagnostics/update-agreement"
GOALS = ("temporal", "cube_behind")
GROUPS = ("g0", "g1", "g2")
PAIRS = tuple(itertools.combinations(GROUPS, 2))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def add_pin(pins: dict, path: Path | str, expected: str | None = None):
    path = Path(path)
    key = str(path.absolute())
    expected = expected or sha(path)
    if key in pins and pins[key] != expected:
        raise ValueError(f"Conflicting artifact hashes: {path}")
    pins[key] = expected


def verify(pins: dict):
    for path, expected in pins.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Artifact changed: {path}")


def prepare():
    pins = {}
    goals = {}
    add_pin(pins, Path(__file__).resolve())
    for goal in GOALS:
        registry_path = OLDROOT / goal / "adaptation/weightsmith/candidates.json"
        add_pin(pins, registry_path)
        registry = read(registry_path)
        if registry["status"] != "complete" or registry["goal_id"] != goal:
            raise ValueError(f"Incomplete registry: {goal}")
        if registry["heldout_used"] or registry["development_outcomes_used"]:
            raise ValueError("Update construction used evaluation outcomes")
        for path, expected in registry["artifacts"].items():
            add_pin(pins, path, expected)
        candidates = {c["group_id"]: c for c in registry["candidates"]}
        if len(registry["candidates"]) != 3 or set(candidates) != set(GROUPS):
            raise ValueError("Expected all three groups without selection")
        used_pairs = set()
        entries = {}
        for group in GROUPS:
            candidate = candidates[group]
            workspace = Path(candidate["workspace"])
            construction = workspace.parent / "construction.json"
            add_pin(pins, construction)
            if read(construction) != candidate:
                raise ValueError("Construction record differs from registry")
            if candidate["source_pairs"] != 3 or candidate["status"] != "complete":
                raise ValueError("Candidate does not contain three source pairs")
            if len(candidate["pair_ids"]) != 3 or used_pairs.intersection(candidate["pair_ids"]):
                raise ValueError("Source groups are not disjoint triplets")
            used_pairs.update(candidate["pair_ids"])
            add_pin(pins, candidate["source_manifest"], candidate["source_manifest_sha256"])
            for path, expected in candidate["artifacts"].items():
                add_pin(pins, path, expected)
            description = read(workspace / "workspace.json")
            layers = {layer["layer"]: layer for layer in candidate["layers"]}
            if len(candidate["layers"]) != 17 or set(layers) != set(range(17)):
                raise ValueError("Expected precisely layers 0 through 16")
            paths = {}
            for layer_id, layer in layers.items():
                entry = description["tensors"][layer["tensor"]]
                base = workspace / "base" / entry["file"]
                edited = workspace / "edits" / entry["file"]
                # Deduplicate shared base files while checking the candidate's pin.
                add_pin(pins, base.resolve(), layer["base_sha256"])
                add_pin(pins, edited, layer["edited_sha256"])
                paths[layer_id] = {"base": base.resolve(), "edited": edited,
                                   "base_sha256": layer["base_sha256"], "shape": entry["shape"]}
            entries[group] = {"candidate": candidate, "paths": paths}
        for layer in range(17):
            if len({entries[g]["paths"][layer]["base_sha256"] for g in GROUPS}) != 1:
                raise ValueError("Groups do not share base weights")
        goals[goal] = {"registry": registry_path, "entries": entries}
    return pins, goals


def pair_metrics(norm2: dict, dots: dict, goal: str, layer: int | str):
    rows = []
    for a, b in PAIRS:
        na, nb = math.sqrt(norm2[a]), math.sqrt(norm2[b])
        value = dots[(a, b)] / (na * nb) if na and nb else None
        if value is not None and not -1.0000000001 <= value <= 1.0000000001:
            raise ValueError("Invalid cosine")
        rows.append({"goal": goal, "layer": layer, "group_a": a, "group_b": b,
                     "cosine": value, "norm_a": na, "norm_b": nb,
                     "norm_ratio_a_over_b": na / nb if nb else None,
                     "inner_product": dots[(a, b)]})
    return rows


def compute_goal(goal: str, entries: dict):
    accumulated_norm2 = {g: 0.0 for g in GROUPS}
    accumulated_dots = {p: 0.0 for p in PAIRS}
    rows = []
    for layer in range(17):
        base = np.load(entries["g0"]["paths"][layer]["base"], mmap_mode="r", allow_pickle=False)
        deltas = {}
        for group in GROUPS:
            paths = entries[group]["paths"][layer]
            edited = np.load(paths["edited"], mmap_mode="r", allow_pickle=False)
            if list(base.shape) != paths["shape"] or edited.shape != base.shape:
                raise ValueError("Mismatched weight shapes")
            if base.dtype != np.float32 or edited.dtype != np.float32:
                raise ValueError("Expected deployed effective-bfloat16 weights stored as float32")
            delta = np.subtract(edited, base, dtype=np.float64).reshape(-1)
            if not np.isfinite(delta).all():
                raise ValueError("Nonfinite deployed weight update")
            deltas[group] = delta
            del edited
        norm2 = {g: float(np.dot(deltas[g], deltas[g])) for g in GROUPS}
        dots = {p: float(np.dot(deltas[p[0]], deltas[p[1]])) for p in PAIRS}
        rows.extend(pair_metrics(norm2, dots, goal, layer))
        for group in GROUPS:
            accumulated_norm2[group] += norm2[group]
        for pair in PAIRS:
            accumulated_dots[pair] += dots[pair]
        del base, deltas, delta
        print(json.dumps({"goal": goal, "completed_layer": layer}), flush=True)
    return rows, pair_metrics(accumulated_norm2, accumulated_dots, goal, "all_0_16")


def write_json(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    start = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    pins, goals = prepare()
    verify(pins)
    print(json.dumps({"preflight_hashes_verified": len(pins)}), flush=True)
    layer_rows, overall_rows, provenance = [], [], {}
    for goal, item in goals.items():
        layers, overall = compute_goal(goal, item["entries"])
        layer_rows.extend(layers)
        overall_rows.extend(overall)
        provenance[goal] = {"registry": str(item["registry"]), "groups": {
            g: {key: item["entries"][g]["candidate"][key] for key in
                ("candidate_id", "workspace", "source_pairs", "pair_ids", "source_indices",
                 "source_manifest", "source_manifest_sha256")}
            for g in GROUPS}}
    verify(pins)
    report = {"status": "complete_and_verified", "diagnostic": "deployed_weight_update_agreement",
              "scope": "Exploratory source-group consistency; not policy efficacy or generalization.",
              "definition": "deltaW = stored edited weight minus stored base weight, computed in float64",
              "aggregation": "Overall cosine from summed layer inner products and squared norms, not mean cosine",
              "all_groups_reported": True, "source_group_selection": False,
              "new_policy_episodes": 0, "gpu_used": False, "max_blas_threads": 4,
              "hashes_verified_before_and_after": True, "artifacts": pins,
              "provenance": provenance, "overall": overall_rows, "per_layer": layer_rows,
              "wall_seconds": time.time() - start}
    for filename, rows in (("overall.csv", overall_rows), ("per-layer.csv", layer_rows)):
        with (OUTPUT / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(OUTPUT / "report.json", report)
    write_json(OUTPUT / "receipt.json", {"status": "complete_and_verified", "new_policy_episodes": 0,
               "artifacts": {str(p): sha(p) for p in (OUTPUT / "report.json", OUTPUT / "overall.csv",
                                                        OUTPUT / "per-layer.csv")}})
    print(json.dumps({"overall": overall_rows, "report": str(OUTPUT / "report.json"),
                      "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
