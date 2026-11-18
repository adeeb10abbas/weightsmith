"""Create a new design, freeze development selection, then freeze final testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from common import HERE, PRIOR, REPO, ROOT, freeze_plan, protocol, read, save, select_winner, sha, test_batches, verify_registry, workspace_identity


def design():
    from common import make_layouts
    from weightsmith import config as C
    from weightsmith.edits import fingerprint
    p = protocol(); output = ROOT / "design"
    if output.exists():
        raise FileExistsError(output)
    checked = []
    ranges = [(p["development"]["slot_start"], 16), (p["heldout"]["slot_start"], 64)]
    for path in (REPO / "projects").glob("**/*plan*.json"):
        try:
            old = read(path)
        except (ValueError, OSError):
            continue
        if not isinstance(old, dict):
            continue
        for b in old.get("batches", []):
            if "slot_start" not in b:
                continue
            for start, n in ranges:
                if max(start, b["slot_start"]) < min(start + n, b["slot_start"] + b.get("n", 16)):
                    raise ValueError(f"Slot overlap: {path}")
        if old.get("batches"):
            checked.append(str(path.resolve()))
    layouts = {split: make_layouts(split) for split in ("development", "heldout")}
    canonical = lambda row: json.dumps(row["poses"], sort_keys=True)
    proposed = [canonical(row) for blob in layouts.values() for row in blob["layouts"]]
    if len(set(proposed)) != len(proposed):
        raise ValueError("Development and test layouts overlap")
    old_layouts = list((PRIOR / "runtime/layouts").glob("temporal_*.json"))
    old_layouts += list((REPO / "projects/rss-historical-recovery-20261118/freeze").glob("temporal_development.json"))
    for path in old_layouts:
        if set(proposed) & {canonical(row) for row in read(path)["layouts"]}:
            raise ValueError(f"A layout was previously used: {path}")
    identities = {}
    for condition, key in (("weightsmith", "historical_full"), ("matched_random", "matched_random")):
        entry = p[key]
        identities[condition] = workspace_identity(entry["workspace"], entry["fingerprint"])
    old_root = Path(p["source_project"])
    ancestry_paths = [old_root / "cands/ctl-rand/meta.json", old_root / "heldout/ctl1-90000-64/plan.json",
                      REPO / "study/results/opus/controls/ctl_rand.py", REPO / "study/results/opus/controls/ctl_rand_report.json"]
    random_metadata = read(ancestry_paths[0]); old_plan = read(ancestry_paths[1]); random_report = read(ancestry_paths[-1])
    if (random_metadata.get("from") != "v2mi-f"
            or old_plan["arms"]["rand"]["fingerprint"] != p["matched_random"]["fingerprint"]
            or fingerprint(old_root / "cands/v2mi-f/edits") != p["historical_full"]["fingerprint"]
            or [r["layer"] for r in random_report] != list(range(17))
            or any(r["reproduces_v2mi_f"] != 1.0 or r["rank"] != 72 for r in random_report)
            or any(abs(r["fro_edit"] - r["fro_rand"]) > 1e-10 for r in random_report)):
        raise ValueError("Random-control ancestry or exact full-edit construction check differs")
    random_chain = {"parent_candidate": "v2mi-f", "parent_fingerprint": p["historical_full"]["fingerprint"],
                    "random_fingerprint": p["matched_random"]["fingerprint"],
                    "all_17_layers_reproduced_saved_full_edit": True,
                    "matching": "Rank and singular values before BF16 storage rounding; one seed-0 random direction realization",
                    "verification": "Current parent fingerprint plus archived construction/evaluation metadata; no new randomized control fit",
                    "artifacts": {str(path): sha(path) for path in ancestry_paths}}
    capture = Path(p["source_project"]) / "data/acts/c2-multi/index.json"
    idx = read(capture)
    observations = []
    for i, (episode, decision) in enumerate(p["source_decisions"]):
        original, clarified = idx["items"][i], idx["items"][i + 12]
        if any((row["episode"], row["k"]) != (episode, decision) for row in (original, clarified)):
            raise ValueError("Construction states differ from the six historical decisions")
        if original["prompt"] != p["original_prompt"] or clarified["prompt"] != p["clarified_prompt"]:
            raise ValueError("Archived source instructions differ from protocol")
        source = Path(p["source_project"]) / "data/episodes" / episode / "obs.npz"
        observations.append({"episode": episode, "decision_index": decision, "obs_path": str(source),
                             "obs_sha256": sha(source), "policy_seed": original["seed"],
                             "original_capture_row": i, "teacher_capture_row": i + 12})
    checkpoint = {}
    for line in (REPO / "third_party/pi05_droid_jointpos.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (C.CHECKPOINT / relative).resolve()
        if sha(path) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        checkpoint[str(path)] = expected
    save(output / "protocol.json", p)
    for split, blob in layouts.items():
        save(output / f"{split}-layouts.json", blob)
    save(output / "historical-candidates.json", identities)
    save(output / "random-control-ancestry.json", random_chain)
    save(output / "sources.json", {"reuse_or_new": "Six reused historical development observations; no newly selected Weightsmith states",
                                    "capture_index": str(capture), "capture_index_sha256": sha(capture),
                                    "archived_capture_parity": idx.get("base_parity_max_abs_action_error"),
                                    "source_observations": observations, "checkpoint_artifacts": checkpoint})
    save(output / "design.json", {"cohort_id": p["cohort_id"], "protocol_sha256": sha(HERE / "protocol.json"),
                                   "development_policy_budget": 48, "heldout_policy_budget": 320,
                                   "slot_overlap_checked_plans": checked,
                                   "new_layouts_checked_against": [str(x) for x in old_layouts],
                                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                                   "geometry_validation": p["reset_gate"], "policy_outcomes_used_to_generate_layouts": False})
    print(json.dumps({"design": str(output), "source_decisions": len(observations), "development_episodes": 48, "test_episodes": 320}))


def design_artifacts():
    p = protocol()
    if read(ROOT / "design/protocol.json") != p or read(ROOT / "design/design.json")["protocol_sha256"] != sha(HERE / "protocol.json"):
        raise ValueError("Protocol changed after design declaration")
    sources = read(ROOT / "design/sources.json")
    artifacts = dict(sources["checkpoint_artifacts"])
    artifacts[sources["capture_index"]] = sources["capture_index_sha256"]
    artifacts.update({row["obs_path"]: row["obs_sha256"] for row in sources["source_observations"]})
    identities = read(ROOT / "design/historical-candidates.json")
    artifacts.update(read(ROOT / "design/random-control-ancestry.json")["artifacts"])
    for entry in identities.values():
        artifacts.update(entry["artifacts"]); artifacts.update(entry["base_artifacts"])
    for path, checksum in artifacts.items():
        if sha(path) != checksum:
            raise ValueError(f"Declared source or candidate changed: {path}")
    return artifacts, identities


def selection():
    p = protocol(); artifacts, _ = design_artifacts()
    candidates, candidate_artifacts = verify_registry()
    artifacts.update(candidate_artifacts)
    batches = [{"cell": "lora_selection", "condition": c["candidate_id"], "learning_rate": c["learning_rate"],
                "task": p["task"], "goal": p["goal"], "input_prompt": p["original_prompt"],
                "effective_prompt": p["original_prompt"], "workspace": c["workspace"],
                "layouts": str(ROOT / "design/development-layouts.json"),
                "slot_start": p["development"]["slot_start"], "n": 16}
               for c in sorted(candidates, key=lambda x: x["learning_rate"])]
    plan = {"cohort_id": p["cohort_id"], "case": "temporal", "split": "development", "inference_batching": "fixed16",
            "protocol_sha256": sha(HERE / "protocol.json"), "batches": batches, "max_policy_episodes": 48,
            "purpose": "Choose one of three fixed LoRA learning rates using fresh development executions only"}
    print(freeze_plan(ROOT / "selection-frozen", plan, artifacts))


def heldout():
    # Import lazily so the same verifier is used for selection and final reports.
    from analyze import collect
    p = protocol(); artifacts, historical = design_artifacts()
    candidates, candidate_artifacts = verify_registry(); artifacts.update(candidate_artifacts)
    _, groups, receipts, source_artifacts = collect(ROOT / "selection", expected_split="development")
    candidates_by_id = {c["candidate_id"]: c for c in candidates}
    rows = [{"candidate_id": key, "learning_rate": candidates_by_id[key]["learning_rate"],
             "n": len(values), "successes": sum(x["success"] for x in values)} for key, values in groups.items()]
    winner = select_winner(rows); selected = candidates_by_id[winner["candidate_id"]]
    selected_path = ROOT / "selection.json"
    if selected_path.exists():
        raise FileExistsError(selected_path)
    save(selected_path, {"split": "development", "status": "complete", "heldout_used": False,
                         "selection_rule": p["lora"]["selection"], "candidates": sorted(rows, key=lambda x: x["learning_rate"]),
                         "selected_candidate_id": winner["candidate_id"], "selected_learning_rate": winner["learning_rate"],
                         "selected_workspace": selected["workspace"], "registry_sha256": sha(ROOT / "lora-adaptation/candidates.json"),
                         "new_development_policy_episodes": 48, "source_receipts": source_artifacts})
    artifacts[str(selected_path)] = sha(selected_path)
    # Raw observations were verified by collect; retain hashes in selection.json
    # and avoid re-reading those large files before each heldout batch.
    artifacts.update({k: v for k, v in source_artifacts.items() if not k.endswith(".npz")})
    workspaces = {"weightsmith": historical["weightsmith"]["workspace"],
                  "matched_random": historical["matched_random"]["workspace"], "teacher_action_lora": selected["workspace"]}
    plan = {"cohort_id": p["cohort_id"], "case": "temporal", "split": "heldout", "inference_batching": "fixed16",
            "protocol_sha256": sha(HERE / "protocol.json"), "batches": test_batches(workspaces), "max_policy_episodes": 320,
            "selection_sha256": sha(selected_path), "primary_estimand": p["primary_estimand"]}
    print(freeze_plan(ROOT / "heldout-frozen", plan, artifacts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["design", "selection", "heldout"])
    args = parser.parse_args()
    {"design": design, "selection": selection, "heldout": heldout}[args.stage]()
