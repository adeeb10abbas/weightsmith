import shutil
from pathlib import Path

import pytest
from plan import build_plan

PACKET = Path(__file__).resolve().parents[1] / "packet"


def test_reference_reuse_grouping_and_review_gates():
    plan = build_plan(PACKET, {"RubiksCubeLeftOfBowlTask"})
    cases = {c["case_id"]: c for c in plan["cases"]}
    assert plan["registry_pairs"] == 12
    assert plan["planned_new_screen_episodes_upper_bound"] == 640
    assert plan["candidate_distinct_new_goals"] == 6
    assert plan["launch_ready_cases"] == 0
    assert cases["T01"]["planned_episodes"] == cases["S03"]["planned_episodes"] == 0
    assert cases["T02"]["semantic_goal_id"] == cases["T03"]["semantic_goal_id"]
    assert cases["T02"]["semantic_goal_id"] != cases["B01"]["semantic_goal_id"]
    assert cases["T02"]["correlation_cluster"] == cases["B01"]["correlation_cluster"]
    assert cases["S01"]["correlation_cluster"] == cases["S02"]["correlation_cluster"]
    assert not cases["T04"]["new_goal_candidate"]
    assert "task_implementation_and_predicate_validation" in cases["T02"]["blocking_gates"]
    assert "task_implementation_and_predicate_validation" not in cases["S01"]["blocking_gates"]
    assert all(not c["launch_ready"] for c in cases.values())


@pytest.mark.parametrize("old,new", [
    ("version: v2_additive", "version: old"),
    ("enable_holdout_readback_to_editor: false", "enable_holdout_readback_to_editor: true"),
    ("no_overwrite: true", "no_overwrite: false"),
    ("trial_starts_per_pair: 32", "trial_starts_per_pair: 16"),
])
def test_changed_protocol_is_rejected(tmp_path, old, new):
    packet = tmp_path / "packet"
    shutil.copytree(PACKET, packet)
    manifest = packet / "IROS2027_SWEEP_MANIFEST.yaml"
    manifest.write_text(manifest.read_text().replace(old, new))
    with pytest.raises(ValueError):
        build_plan(packet, set())


def test_duplicate_pair_cannot_hide_missing_screen(tmp_path):
    packet = tmp_path / "packet"
    shutil.copytree(PACKET, packet)
    registry = packet / "IROS2027_PROMPT_PAIRS.csv"
    registry.write_text(registry.read_text().replace("T04,", "T01,"))
    with pytest.raises(ValueError, match="exactly once"):
        build_plan(packet, set())
