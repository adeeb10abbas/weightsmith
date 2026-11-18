"""Audit complete new-goal cohorts; qualification failures remain part of the result."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from worker import digest, verify_global_freeze
from qualify import build_plan, gate, read_outcomes


def read(path):
    return json.loads(Path(path).read_text())


def verify_qualification(root, protocol, evidence):
    """Reconstruct every required attempt, including goals that fail the gate."""
    root = Path(root).resolve()
    state = read(root / "qualification-status.json")
    if state.get("status") != "complete" or state.get("protocol_sha256") != digest(HERE / "protocol.json"):
        raise ValueError("Qualification must be complete under the frozen protocol")
    frozen = root / "qualification-source-freeze.json"
    for path, checksum in read(frozen).items():
        if digest(path) != checksum:
            raise ValueError("A qualification source changed after freeze")
    evidence[str(frozen)] = digest(frozen)
    evidence[str(root / "qualification-status.json")] = digest(root / "qualification-status.json")
    required_owners, required_receipts, development = set(), set(), []
    initial, repeat, competing = {}, {}, {}
    def check(case, stage):
        owner = root / "qualification" / case / stage
        expected = build_plan(protocol, case, stage)
        if read(owner / "plan.json") != expected:
            raise ValueError("A development owner differs from its predeclared stage plan")
        rows = read_outcomes(owner)
        required_owners.add(owner)
        for path in (owner / "plan.json", owner / "status.json", owner / "identity.json"):
            evidence[str(path)] = digest(path)
        for number, spec in enumerate(expected["batches"]):
            batch = owner / f"batch-{number:02d}-{spec['cell']}-base"
            receipt_path = batch / "completed.json"
            receipt = read(receipt_path)
            required_receipts.add(receipt_path)
            episodes = read(batch / "episodes.json")
            if {r["slot"] for r in episodes} != set(range(spec["slot_start"], spec["slot_start"] + 16)):
                raise ValueError("Qualification noise slots differ from the protocol")
            for row in episodes:
                identity = {"cohort_id": expected["cohort_id"], "split": "development", "case": case,
                    "cell": spec["cell"], "condition": "base", "inference_batching": "fixed16", "ok": True,
                    "task_name": spec["task"], "effective_prompt": spec["effective_prompt"], "input_prompt": spec["input_prompt"],
                    "layout_id": spec["layout_ids"][row["slot"] % 4]}
                if any(row.get(k) != v for k, v in identity.items()):
                    raise ValueError("Qualification episode identity differs from its frozen plan")
            development.append({"case": case, "stage": stage,
                **{k: receipt.get(k) for k in ("cell", "n", "successes", "wall_seconds", "queue_seconds", "execution_seconds")}})
            for path in (receipt_path, batch / "episodes.json"):
                evidence[str(path)] = digest(path)
        return rows
    for case in protocol["case_order"]:
        initial[case] = check(case, "initial")
        initial[case]["passes_gate"] = gate(initial[case]["original"]["successes"], initial[case]["teacher"]["successes"])
        if initial[case]["passes_gate"]:
            repeat[case] = check(case, "repeat")
            repeat[case]["passes_gate"] = gate(repeat[case]["original"]["successes"], repeat[case]["teacher"]["successes"])
    selected = [case for case in protocol["case_order"] if repeat.get(case, {}).get("passes_gate")][:2]
    for case in selected:
        competing[case] = check(case, "competing")
        competent = competing[case]["competing"]["successes"] >= 8
        competing[case]["competence_qualified"] = competent
        qualified_path = root / "qualification" / case / "qualified.json"
        qualified = read(qualified_path)
        expected_rounds = [{cell: r[case][cell]["batch"] for cell in ("original", "teacher")} for r in (initial, repeat)]
        if (qualified.get("case") != case or qualified.get("split") != "development"
                or qualified.get("rounds") != expected_rounds
                or qualified.get("teacher_batch") != expected_rounds[0]["teacher"]
                or qualified.get("preservation_batch") != competing[case]["competing"]["batch"]
                or qualified.get("preservation_competence_qualified") != competent
                or qualified.get("protocol_sha256") != state["protocol_sha256"]
                or qualified.get("qualification_source_freeze_sha256") != digest(frozen)):
            raise ValueError("Qualified-goal receipt differs from its actual development evidence")
        evidence[str(qualified_path)] = digest(qualified_path)
    if (state["initial"] != initial or state["repeat"] != repeat or state["competing"] != competing
            or state["selected"] != selected):
        raise ValueError("Qualification decisions or selected order differ from the actual receipts")
    found_owners = {p.parent for p in (root / "qualification").glob("*/*/status.json")}
    found_receipts = set((root / "qualification").glob("*/*/batch-*/completed.json"))
    if found_owners != required_owners or found_receipts != required_receipts:
        raise ValueError("Missing or extra qualification attempts; failed goals cannot disappear")
    return state, development


def paired(reference, method):
    a = {(r["layout_id"], r["slot"]): r for r in reference}
    b = {(r["layout_id"], r["slot"]): r for r in method}
    if len(a) != len(reference) or len(b) != len(method) or set(a) != set(b):
        raise ValueError("Paired method records must contain exactly the same unique planned trials")
    keys = sorted(a)
    if any(a[k]["policy_seed"] != b[k]["policy_seed"] for k in keys):
        raise ValueError("Paired policy-noise streams differ")
    before = np.array([a[k]["success"] for k in keys], dtype=bool)
    after = np.array([b[k]["success"] for k in keys], dtype=bool)
    delta = after.astype(float) - before.astype(float)
    layouts = sorted({k[0] for k in keys})
    means = np.array([delta[[k[0] == layout for k in keys]].mean() for layout in layouts])
    rng = np.random.default_rng(17)
    draws = means[rng.integers(0, len(layouts), (20000, len(layouts)))].mean(axis=1)
    fixed, broken = int((~before & after).sum()), int((before & ~after).sum())
    discordant = fixed + broken
    probability = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(fixed, broken) + 1)) / 2**discordant)
    return {"n": len(keys), "layouts": len(layouts), "reference_successes": int(before.sum()),
        "method_successes": int(after.sum()), "fixed": fixed, "broken": broken,
        "delta_percentage_points": float(delta.mean() * 100),
        "layout_cluster_ci95_percentage_points": (np.quantile(draws, [.025, .975]) * 100).tolist(),
        "mcnemar_exact_two_sided": probability,
        "layout_delta_percentage_points": dict(zip(layouts, (means * 100).tolist())),
        "first_camera_identical_pairs": sum(a[k]["first_camera_sha256"] == b[k]["first_camera_sha256"] for k in keys)}


def verify_case(owner, protocol, evidence):
    plan, status, identity = (read(owner / name) for name in ("plan.json", "status.json", "identity.json"))
    case = plan["case"]
    if (plan["split"] != "heldout" or status["status"] != "complete" or status["completed_episodes"] != 144
            or status["completed_batches"] != 9 or len(plan["batches"]) != 9
            or plan.get("inference_batching") != "fixed16"):
        raise ValueError("A new-goal test must contain all144 trials before reporting")
    verify_global_freeze(plan)
    for path, checksum in identity.items():
        if digest(path) != checksum:
            raise ValueError("A source or candidate changed after test execution")
    groups, costs, metrics_cache = {}, [], {}
    for number, spec in enumerate(plan["batches"]):
        batch = owner / f"batch-{number:02d}-{spec['cell']}-{spec['condition']}"
        receipt, rows = read(batch / "completed.json"), read(batch / "episodes.json")
        if len(rows) != 16 or receipt["n"] != 16 or digest(batch / "episodes.json") != receipt["episodes_sha256"]:
            raise ValueError("Incomplete or changed test batch")
        if {r["slot"] for r in rows} != set(range(spec["slot_start"], spec["slot_start"] + 16)):
            raise ValueError("Test slots differ from the frozen plan")
        for row in rows:
            expected = {"case": case, "split": "heldout", "condition": spec["condition"], "cell": spec["cell"],
                "cohort_id": plan["cohort_id"], "task_name": spec["task"], "input_prompt": spec["input_prompt"],
                "effective_prompt": spec["effective_prompt"], "inference_batching": "fixed16", "ok": True,
                "selected_goal": spec["goal"], "layout_id": spec["layout_ids"][row["slot"] % 8]}
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError("Test record identity differs from the frozen case plan")
            metrics_path = Path(row["metrics"]).resolve()
            obs_path = Path(row["obs"]).resolve()
            if not metrics_path.is_relative_to(batch) or not obs_path.is_relative_to(batch):
                raise ValueError("Test inputs or metrics belong to another batch")
            if str(metrics_path) not in metrics_cache:
                metrics_cache[str(metrics_path)] = (digest(metrics_path), read(metrics_path))
            checksum, metrics = metrics_cache[str(metrics_path)]
            if checksum != row["metrics_sha256"] or digest(obs_path) != row["obs_sha256"]:
                raise ValueError("Raw observations or task metrics changed after evaluation")
            with np.load(obs_path, allow_pickle=False) as recorded:
                camera_hashes = {key: hashlib.sha256(recorded[key][0].tobytes()).hexdigest()
                                 for key in ("base_img", "wrist_img")}
                if int(recorded["seed"][0]) != row["policy_seed"] or camera_hashes != row["first_camera_sha256"]:
                    raise ValueError("Actual recorded policy seed or initial images differ from episode metadata")
            if metrics["layout"]["layout_ids"][row["env"]] != row["layout_id"]:
                raise ValueError("Actual simulator layout differs from the planned pairing")
            if metrics["predicate_version"] != protocol["scoring"]["version"] or metrics["selected_goal"] != spec["goal"]:
                raise ValueError("The scored task differs from the declared task")
            for name in ("strict_final", "strict_ever", "stock_final", "stock_ever"):
                actual = {goal: bool(values[row["env"]]) for goal, values in metrics[name].items()}
                if row[name] != actual:
                    raise ValueError("An episode score differs from its physical-metric record")
            if bool(row["success"]) != row["strict_final"][spec["goal"]]:
                raise ValueError("Primary success differs from strict final goal score")
        if receipt["successes"] != sum(r["success"] for r in rows):
            raise ValueError("Receipt count differs from its individual episodes")
        key = (spec["cell"], spec["condition"])
        if key in groups:
            raise ValueError("A test condition was repeated")
        groups[key] = rows
        costs.append({**{k: receipt.get(k) for k in ("cell", "condition", "n", "wall_seconds", "queue_seconds", "execution_seconds")},
                      "simulator_attempt_logs": [str(p) for p in sorted(batch.glob("sim-*.log"))]})
        for path in (batch / "completed.json", batch / "episodes.json"):
            evidence[str(path)] = digest(path)
    expected_groups = {(cell, condition) for cell in protocol["heldout"]["cells"] for condition in protocol["heldout"]["conditions"]}
    if set(groups) != expected_groups:
        raise ValueError("Missing or unexpected comparison cell")
    comparisons, totals = [], []
    for cell in protocol["heldout"]["cells"]:
        for condition in protocol["heldout"]["conditions"]:
            rows = groups[cell, condition]
            totals.append({"case": case, "cell": cell, "condition": condition, "n": len(rows),
                "successes": sum(r["success"] for r in rows),
                "cross_goal_final_successes": {g: sum(r["strict_final"][g] for r in rows) for g in rows[0]["strict_final"]}})
        for reference, method in (("base", "closed_form"), ("base", "saved_rewrite"), ("saved_rewrite", "closed_form")):
            comparisons.append({"case": case, "cell": cell, "reference": reference, "method": method,
                                **paired(groups[cell, reference], groups[cell, method])})
    for path in (owner / "plan.json", owner / "status.json", owner / "identity.json"):
        evidence[str(path)] = digest(path)
    return {"case": case, "primary": totals, "comparisons": comparisons, "batch_costs": costs,
        "preservation_competence_qualified": plan["preservation_competence_qualified"], "verified_policy_episodes": 144}


def report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = read(HERE / "protocol.json")
    evidence = {str(HERE / "protocol.json"): digest(HERE / "protocol.json"), str(Path(__file__)): digest(__file__)}
    qualification, development = verify_qualification(root, protocol, evidence)
    cases = [verify_case(root / f"{case}-heldout-v1", protocol, evidence) for case in qualification["selected"]]
    adaptation = {case: read(root / f"{case}-adaptation/status.json") for case in qualification["selected"]}
    result = {"status": "complete", "cohort_id": protocol["cohort"], "protocol": protocol,
        "qualification": qualification, "development": development, "adaptation_costs": adaptation, "cases": cases,
        "counts": {"candidate_goals": len(protocol["case_order"]), "qualified_goals": len(cases),
            "new_development_policy_episodes": sum(x["n"] for x in development), "new_heldout_policy_episodes": 144 * len(cases)},
        "verification": {"raw_test_observation_hashes_checked": True, "physical_metric_scores_checked": True,
                         "source_and_candidate_freezes_checked": bool(cases)},
        "simulator_attempt_logs": [str(p) for p in sorted(root.rglob("sim-l*a*.log"))],
        "limits": ["Historically informed task and prompt choice; all new qualifications and failures are reported.",
            "Eight layout clusters per goal; paired camera inputs need not be identical; exploratory intervals are not preservation equivalence tests.",
            "Saved routing is hand-engineered with the registered vocabulary; only weight adaptation excludes the exact unseen phrase.",
            "LoRA is compared in the separate original two-goal cohort, not in this goal-expansion cohort.",
            "Historical discovery and inherited runtime work are separate costs; human effort is not automatically metered.",
            "Episode totals count sealed completed trials. Preserved simulator retries require a separate attempted-episode audit if any nonzero attempt is present."],
        "evidence_sha256": evidence}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(json.dumps(result["counts"]))
