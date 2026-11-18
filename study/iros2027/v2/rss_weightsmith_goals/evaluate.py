"""Freeze every qualified goal and its comparison before any new test rollout."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "adaptation"))
from run import source_files, verify_gate
from prepare import sha
from report import verify_qualification


def read(path):
    return json.loads(Path(path).read_text())


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = read(HERE / "protocol.json")
    qualification_evidence = {}
    qualification, _ = verify_qualification(root, protocol, qualification_evidence)
    expected = [case for case in protocol["case_order"]
                if qualification["repeat"].get(case, {}).get("passes_gate")]
    if qualification["selected"] != expected or not expected:
        raise ValueError("No selected goal, or selection differs from the fixed case order")
    if output.exists():
        raise FileExistsError("Preserve an existing test freeze, including incomplete attempts")
    selected = list(expected)
    construction = {case: read(root / f"{case}-adaptation/status.json") for case in selected}
    if any(row.get("status") not in ("complete", "source_coverage_failed") for row in construction.values()):
        raise ValueError("Every qualified goal must have a terminal construction outcome before testing")
    expected = [case for case in expected if construction[case]["status"] == "complete"]
    if not expected:raise ValueError("No completed candidate construction")
    identities = {str(p): sha(p) for p in source_files()}
    identities.update(qualification_evidence)
    identities[str(root / "qualification-status.json")] = sha(root / "qualification-status.json")
    identities[str(root / "qualification-source-freeze.json")] = sha(root / "qualification-source-freeze.json")
    for case in selected:
        path = root / f"{case}-adaptation/status.json"
        identities[str(path)] = sha(path)
    for path, checksum in read(root / "qualification-source-freeze.json").items():
        if sha(path) != checksum:
            raise ValueError("A qualification source changed after its freeze")
        identities[path] = checksum
    plans = {}
    for case in expected:
        cfg = protocol["cases"][case]
        qualification_path = root / "qualification" / case / "qualified.json"
        identities.update(verify_gate(qualification_path, case))
        adaptation = root / f"{case}-adaptation"
        if read(adaptation / "status.json").get("status") != "complete":
            raise ValueError("Every selected goal's adaptation must complete before testing")
        seal = read(adaptation / "candidate-seal.json")
        if seal.get("case") != case or seal.get("split") != "development":
            raise ValueError("Candidate is from a different case or split")
        for name, checksum in seal["artifacts"].items():
            if sha(name) != checksum:
                raise ValueError("Candidate weights changed after adaptation")
            identities[name] = checksum
        for name, checksum in read(adaptation / "identity.json").items():
            if sha(name) != checksum:
                raise ValueError("An adaptation source changed after its freeze")
            identities[name] = checksum
        for relative in ("spec.json", "candidate-seal.json", "identity.json", "status.json",
                         "edit-fit/solve.json", "native-targets/dataset.json", "edit-prepared/prepared.json"):
            path = adaptation / relative
            identities[str(path)] = sha(path)
        if (seal["spec_sha256"] != sha(adaptation / "spec.json")
                or seal["solve_sha256"] != sha(adaptation / "edit-fit/solve.json")
                or seal["source_identity_sha256"] != sha(adaptation / "identity.json")):
            raise ValueError("Candidate seal does not identify the current adaptation")
        solve = read(adaptation / "edit-fit/solve.json")
        if solve["replay_parity"]["status"] != "pass" or solve["teacher_action_parity"]["status"] != "pass":
            raise ValueError("Recorded-input and teacher replay gates must pass")
        layout = HERE / "runtime/layouts" / f"{case}_heldout.json"
        layout_ids = [row["id"] for row in read(layout)["layouts"]]
        if len(set(layout_ids)) != 8 or read(layout)["split"] != "heldout":
            raise ValueError("Eight distinct new test layouts are required")
        batches = []
        conditions = ["base", "weightsmith"]
        for number, cell in enumerate(protocol["heldout"]["cells"]):
            prompt = cfg["unseen"][int(cell[-1])-1] if cell.startswith("unseen") else cfg[cell]
            for condition in conditions[number % 2:] + conditions[:number % 2]:
                batches.append({"cell":cell,"condition":condition,"n":16,
                    "task":cfg["competing_task"] if cell=="competing" else cfg["task"],
                    "goal":cfg["competing_goal"] if cell=="competing" else cfg["goal"],
                    "workspace":str(adaptation/"edit-fit/workspace") if condition=="weightsmith" else None,
                    "layouts":str(layout),"layout_ids":layout_ids,
                    "slot_start":protocol["heldout"]["slot_start"][case],
                    "input_prompt":prompt,"effective_prompt":prompt})
        plans[case] = {"cohort_id": f"rss-weightsmith-goals-20261118-{case}-heldout-v1",
            "case": case, "split": "heldout", "inference_batching": "fixed16", "batches": batches,
            "preservation_competence_qualified": read(qualification_path)["preservation_competence_qualified"],
            "protocol_sha256": sha(HERE / "protocol.json")}
    output.mkdir(parents=True)
    manifest = {"cohort_id": "rss-weightsmith-goals-20261118", "cases": expected,
        "qualified_cases": selected, "construction_outcomes": construction,
        "plan_sha256": {case: payload_sha(plan) for case, plan in plans.items()},
        "artifacts": identities, "planned_episodes": 160 * len(plans),
        "construction": "All selected candidates frozen together before any new-goal test evaluation"}
    path = output / "freeze.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for case, plan in plans.items():
        plan["global_freeze"] = {"path": str(path), "sha256": sha(path)}
        (output / f"{case}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.root, args.output)
    print(json.dumps({"cases": manifest["cases"], "planned_episodes": manifest["planned_episodes"], "output": str(args.output)}))
