"""Validate the v2 packet and emit a development-only screen plan. Never launch jobs.

Requires PyYAML. Human semantic review and task implementation are explicit launch gates.
The plan retains all 12 registry entries, including reference cases that need no new GPU run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from provenance import sha256_file

# These IDs are semantic goal groups, not evidence of independent repair.
GOALS = {
    "T01": "cube_then_banana_in_bowl", "T04": "cube_then_banana_in_bowl",
    "T02": "cube_bowl_then_banana_plate", "T03": "cube_bowl_then_banana_plate",
    "S01": "cube_left_of_bowl", "S02": "cube_right_of_bowl",
    "S03": "mustard_right_of_raisin", "S04": "banana_left_of_plate",
    "B01": "cube_bowl_and_banana_plate", "B02": "cube_bowl_and_banana_plate",
    "B03": "red_left_blue_right_bins", "B04": "red_left_blue_right_bins",
}
CLUSTERS = {
    "cube_then_banana_in_bowl": "published_ctb",
    "cube_bowl_then_banana_plate": "cube_banana_two_destinations",
    "cube_bowl_and_banana_plate": "cube_banana_two_destinations",
    "cube_left_of_bowl": "cube_bowl_spatial", "cube_right_of_bowl": "cube_bowl_spatial",
    "mustard_right_of_raisin": "existing_cosmos_spatial",
    "banana_left_of_plate": "banana_plate_spatial", "red_left_blue_right_bins": "color_binding",
}


def task_classes(root: Path) -> set[str]:
    """Static class inventory, not proof of runtime discovery or physical competence."""
    names = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        names.update(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return names


def build_plan(packet: Path, available: set[str]) -> dict:
    manifest_path = packet / "IROS2027_SWEEP_MANIFEST.yaml"
    registry_path = packet / "IROS2027_PROMPT_PAIRS.csv"
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("version") != "v2_additive":
        raise ValueError("expected v2_additive packet")
    protocol = manifest["protocol"]
    if protocol["enable_holdout_readback_to_editor"] is not False or protocol["no_holdout_tuning"] is not True:
        raise ValueError("held-out isolation must remain enabled")
    if manifest["existing_evidence"]["no_overwrite"] is not True:
        raise ValueError("historical evidence must remain immutable")
    screen = manifest["screen"]
    with registry_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["case_id"] for row in rows]
    if len(ids) != screen["planned_pairs"] or len(set(ids)) != len(ids) or set(ids) != set(GOALS):
        raise ValueError("registry must contain each of the 12 prespecified IDs exactly once")
    families = Counter(row["family"] for row in rows)
    # The CSV's spatial_converse is the YAML's spatial_converse_goal_equivalence.
    expected_families = {"temporal_scope", "spatial_converse", "object_destination_binding"}
    if set(families) != expected_families or set(families.values()) != {screen["pairs_per_family"]}:
        raise ValueError("expected four wording pairs in each of three families")
    references = set(screen["existing_or_reference_pairs"])
    if references != {"T01", "S03"}:
        raise ValueError("v2 references must be T01 and S03")
    repeated = {frozenset(pair) for pair in screen["repeated_goal_alternatives"]}
    if repeated != {frozenset(pair) for pair in [("T01", "T04"), ("T02", "T03"), ("B01", "B02"), ("B03", "B04")]}:
        raise ValueError("goal grouping differs from this version's reviewed interpretation")
    starts = screen["trial_starts_per_pair"]
    if type(starts) is not int or starts <= 0 or starts % 32:
        raise ValueError("ABBA screen requires a positive multiple of 32 starts (16 environments per block)")
    cases = []
    for row in rows:
        case_id = row["case_id"]
        reference = case_id in references
        goal = GOALS[case_id]
        gates = ["human_semantic_review", "physical_competence_check", "runtime_and_scene_identity"]
        if row["task_implementation"] not in available:
            gates.append("task_implementation_and_predicate_validation")
        if reference:
            gates = ["historical_provenance_audit"]
        cases.append({
            **row, "semantic_goal_id": goal, "correlation_cluster": CLUSTERS[goal],
            "reuse_or_new": "duplicate" if reference else "new_claim",
            "action": "reuse_existing_evidence" if reference else "pending_development_screen",
            "claim": "Historical anchor; audit provenance without replay."
            if reference else "Test whether equivalent wording reveals a latent skill for this semantic goal.",
            "new_goal_candidate": not reference and case_id != "T04",
            "independent_repair_established": False, "human_review": "pending", "launch_ready": False,
            "blocking_gates": gates, "planned_episodes": 0 if reference else starts * 2,
            "batch_order": [] if reference else ["base_original", "base_explicit", "base_explicit", "base_original"],
        })
    candidate_goals = {c["semantic_goal_id"] for c in cases if c["new_goal_candidate"]}
    return {
        "schema_version": 1, "cohort_id": "iros2027-v2-B-screen", "phase": "development_only",
        "source_files_sha256": {p.name: sha256_file(p) for p in (manifest_path, registry_path)},
        "cases": cases, "registry_pairs": len(cases), "reference_pairs_reused": len(references),
        "planned_new_screen_episodes_upper_bound": sum(c["planned_episodes"] for c in cases),
        "candidate_distinct_new_goals": len(candidate_goals), "independent_repairs_established": 0,
        "launch_ready_cases": 0, "gpu_jobs_launched": 0,
        "execution": "planning only; no queue worker or simulator/held-out invocation exists in this script",
        "ordering": "For each 32-start unit: 16 matched starts A then B; another 16 matched starts B then A.",
        "scene_scope": "Fixed original scenes for screening; separate scene-randomized cohort remains to implement.",
        "remaining": ["Human review of goal equivalence and frame conventions", "Implement missing task predicates",
                      "Freeze scene/start identities and runtime hashes before dispatch",
                      "Validate decision-level diagnostics; success-gap alone is insufficient",
                      "Implement an append-only execution ledger and failure-preserving worker",
                      "Resolve one candidate in brief versus maximum two in manifest before edit construction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).parent / "packet")
    parser.add_argument("--task-source", type=Path, required=True, help="installed RoboLab task source directory")
    args = parser.parse_args()
    if not args.task_source.is_dir():
        parser.error("task source directory does not exist")
    print(json.dumps(build_plan(args.packet, task_classes(args.task_source)), indent=2))


if __name__ == "__main__":
    main()
