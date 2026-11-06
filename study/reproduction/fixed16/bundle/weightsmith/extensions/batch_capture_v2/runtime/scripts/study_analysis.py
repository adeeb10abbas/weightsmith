#!/usr/bin/env python3
"""Operator-only, dependency-free analysis for the approved RQ1-RQ3 study.

This script never launches evaluation or chooses policies from held-out results.
It rejects duplicate/mispaired slots, preserves missingness, and writes immutable
JSON/Markdown reports. Candidate selection is available as a pure development-
only function. Inputs can be pool rows.json, a row list, or owsp index.jsonl.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/study_rq123/protocol.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slot(value):
    if isinstance(value, str) and re.fullmatch(r"s?\d+", value):
        value = int(value.lstrip("s"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid sampling slot: {value!r}")
    return value


def read_rows(path, label=None, fingerprint=None):
    """Load one explicitly identified cohort; never average historical repeats."""
    path = Path(path)
    raw = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        rows = data.get("rows") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("Expected a row list, pool {'rows': [...]} file, or JSONL cohort")
    if label is not None:
        rows = [r for r in rows if r.get("cand") == label]
    if fingerprint is not None:
        # A fingerprint is an assertion, not a filter that could conceal stale rows.
        if any(r.get("ok", True) and r.get("fingerprint") != fingerprint for r in rows):
            raise ValueError("A completed row does not match the pinned fingerprint")
    return rows


def normalize_rows(rows, expected_slots, *, task=None, instruction=None, fingerprint=None):
    expected = set(expected_slots)
    result = {}
    for raw in rows:
        slot = _slot(raw.get("slot"))
        if slot not in expected:
            raise ValueError(f"Unexpected slot {slot}; select one exact cohort")
        if slot in result:
            raise ValueError(f"Duplicate slot {slot}; repeated bases must not be pooled")
        ok = raw.get("ok")
        if ok is None and "eid" in raw:
            ok = True  # owsp index contains completed episodes only.
        if not isinstance(ok, bool):
            raise ValueError(f"Slot {slot} lacks a boolean completion field")
        row = dict(raw, slot=slot, ok=ok)
        if ok:
            if not isinstance(raw.get("success"), bool):
                raise ValueError(f"Slot {slot} lacks a boolean success field")
            row_prompt = raw.get("prompt")
            for field in ("task", "task_name", "sim_task"):
                if task is not None and raw.get(field) is not None and raw[field] != task:
                    raise ValueError(f"Task mismatch at slot {slot}: {raw[field]!r}")
            if instruction is not None and row_prompt is not None and row_prompt != instruction:
                raise ValueError(f"Instruction mismatch at slot {slot}")
            if fingerprint is not None and raw.get("fingerprint") != fingerprint:
                raise ValueError(f"Fingerprint mismatch at slot {slot}")
            if raw.get("steps") is not None:
                steps = raw["steps"]
                if isinstance(steps, bool) or not isinstance(steps, (int, float)) or not math.isfinite(steps) or steps < 0:
                    raise ValueError(f"Invalid duration at slot {slot}")
        result[slot] = row
    fingerprints = {r["fingerprint"] for r in result.values() if r["ok"] and r.get("fingerprint")}
    if len(fingerprints) > 1:
        raise ValueError("One arm contains multiple policy fingerprints; select one exact cohort")
    return result


def percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def mcnemar_exact(fixed, broken):
    n = fixed + broken
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2 ** n)


def paired_bootstrap(differences, *, resamples=10000, seed=20261104):
    if not differences:
        return None
    if not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    rng = random.Random(seed)
    n = len(differences)
    estimates = [sum(rng.choices(differences, k=n)) / n for _ in range(resamples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def _durations(rows):
    successes = [r for r in rows if r["ok"] and r["success"]]
    steps = [r["steps"] for r in successes if r.get("steps") is not None]
    return {"successful_episodes": len(successes), "durations_available": len(steps),
            "mean_steps": statistics.mean(steps) if steps else None,
            "median_steps": statistics.median(steps) if steps else None,
            "minimum_steps": min(steps) if steps else None,
            "maximum_steps": max(steps) if steps else None,
            "interpretation": "conditional on successful episodes, not all-episode latency"}


def analyze_pairs(base_rows, candidate_rows, expected_slots, *, task=None, instruction=None,
                  base_fingerprint=None, candidate_fingerprint=None, resamples=10000, seed=20261104):
    expected_slots = sorted(_slot(s) for s in expected_slots)
    if not expected_slots or len(set(expected_slots)) != len(expected_slots):
        raise ValueError("Expected slots must be a nonempty unique set")
    base = normalize_rows(base_rows, expected_slots, task=task, instruction=instruction, fingerprint=base_fingerprint)
    cand = normalize_rows(candidate_rows, expected_slots, task=task, instruction=instruction, fingerprint=candidate_fingerprint)
    paired = [s for s in expected_slots if base.get(s, {}).get("ok") and cand.get(s, {}).get("ok")]
    for slot in paired:
        for key in ("task", "task_name", "prompt"):
            left, right = base[slot].get(key), cand[slot].get(key)
            if left is not None and right is not None and left != right:
                raise ValueError(f"Paired {key} differs at slot {slot}")
    diffs = [int(cand[s]["success"]) - int(base[s]["success"]) for s in paired]
    fixed = [s for s in paired if not base[s]["success"] and cand[s]["success"]]
    broken = [s for s in paired if base[s]["success"] and not cand[s]["success"]]
    kb, kc = sum(base[s]["success"] for s in paired), sum(cand[s]["success"] for s in paired)
    transitions = Counter((base[s].get("reason") or "unrecorded", cand[s].get("reason") or "unrecorded") for s in paired)
    return {
        "status": "complete" if len(paired) == len(expected_slots) else "incomplete",
        "expected_n": len(expected_slots), "paired_n": len(paired), "paired_slots": paired,
        "unpaired_slots": sorted(set(expected_slots) - set(paired)),
        "base_missing_or_incomplete_slots": [s for s in expected_slots if not base.get(s, {}).get("ok")],
        "candidate_missing_or_incomplete_slots": [s for s in expected_slots if not cand.get(s, {}).get("ok")],
        "base_success": kb, "candidate_success": kc,
        "base_rate_paired": kb / len(paired) if paired else None,
        "candidate_rate_paired": kc / len(paired) if paired else None,
        "success_difference": sum(diffs) / len(diffs) if diffs else None,
        "paired_bootstrap_95": paired_bootstrap(diffs, resamples=resamples, seed=seed),
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "fixed": len(fixed), "broken": len(broken), "fixed_slots": fixed, "broken_slots": broken,
        "both_success": sum(base[s]["success"] and cand[s]["success"] for s in paired),
        "both_failure": sum(not base[s]["success"] and not cand[s]["success"] for s in paired),
        "mcnemar_exact_p_exploratory": mcnemar_exact(len(fixed), len(broken)) if paired else None,
        "success_duration": {"base": _durations([base[s] for s in paired]), "candidate": _durations([cand[s] for s in paired])},
        "terminal_reason_transitions": [{"base": a, "candidate": b, "episodes": n} for (a, b), n in sorted(transitions.items())],
        "base_fingerprints": sorted({r["fingerprint"] for r in base.values() if r.get("fingerprint")}),
        "candidate_fingerprints": sorted({r["fingerprint"] for r in cand.values() if r.get("fingerprint")}),
        "limitations": ["Differences use complete paired slots only; missingness is not silently imputed.",
                        "Bootstrap unit is an episode slot; batched simulator dependence may reduce effective sample size.",
                        "One comparison conditions on its discovered policy and does not estimate variability across editor restarts.",
                        "McNemar is exploratory; no isolated confirmatory significance claim or non-inferiority claim is warranted."]
    }


def select_development_candidate(base_rows, candidates, expected_slots, *, incumbent=None):
    """Rank only complete development candidates against the pinned original base.

    candidates: [{name, rows, deployment_delta_l2, changed_elements, ordinal, ...}]. An incumbent
    is optional but must be one of these candidates. The original base is always
    available. Norms must refer to actual cumulative changes after bf16 rounding.
    """
    expected = sorted(_slot(s) for s in expected_slots)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Selection requires a nonempty set of unique expected slots")
    base = normalize_rows(base_rows, expected)
    if len(base) != len(expected) or any(not r["ok"] for r in base.values()):
        raise ValueError("Selection requires one complete pinned development base")
    kb = sum(r["success"] for r in base.values())
    names, ordinals = set(), set()
    evaluated = []
    for candidate in candidates:
        name, ordinal, norm = candidate["name"], candidate["ordinal"], candidate["deployment_delta_l2"]
        changed = candidate.get("changed_elements")
        if type(changed) is not int or changed < 0:
            raise ValueError("Actual post-bfloat16 changed_elements count is required")
        if name in names or ordinal in ordinals or name == "base":
            raise ValueError("Candidate names and ordinals must be unique; 'base' is reserved")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError("Candidate ordinal must be a positive integer")
        if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
            raise ValueError("Deployment delta norm must be finite and nonnegative")
        names.add(name); ordinals.add(ordinal)
        rows = normalize_rows(candidate["rows"], expected)
        complete = len(rows) == len(expected) and all(r["ok"] for r in rows.values())
        successes = sum(r["success"] for r in rows.values() if r["ok"])
        broken = sum(base[s]["success"] and not r["success"] for s, r in rows.items() if r["ok"])
        eligible = complete and changed > 0 and successes > kb and broken <= 2
        evaluated.append({"name": name, "ordinal": ordinal, "deployment_delta_l2": norm,
                          "changed_elements": changed, "identity_candidate": changed == 0,
                          "complete": complete, "success": successes, "broken_against_original": broken,
                          "eligible": eligible})
    if incumbent is not None and incumbent != "base" and incumbent not in names:
        raise ValueError("Incumbent must be included with its pinned rows and deployment norm")
    eligible = [c for c in evaluated if c["eligible"]]
    best = min(eligible, key=lambda c: (-c["success"], c["deployment_delta_l2"], c["ordinal"])) if eligible else None
    selected = best["name"] if best else "base"
    return {"selected": selected, "base_success": kb, "candidate_reports": evaluated,
            "incumbent": incumbent, "replace_incumbent": selected != (incumbent or "base"),
            "reference": "original_pinned_development_base"}


def markdown_report(result):
    def display(value):
        return "not available" if value is None else f"{value:.3f}"
    ci = result["paired_bootstrap_95"]
    interval = "not available" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    lines = ["# Paired episode comparison", "", f"Status: **{result['status']}**. All rates below use complete pairs.", "",
             "| Quantity | Result |", "|---|---:|",
             f"| Expected / complete pairs | {result['expected_n']} / {result['paired_n']} |",
             f"| Base / candidate successes | {result['base_success']} / {result['candidate_success']} |",
             f"| Candidate minus base success fraction | {display(result['success_difference'])} |",
             f"| Paired bootstrap 95% interval | {interval} |",
             f"| Fixed / broken | {result['fixed']} / {result['broken']} |",
             f"| Exact McNemar p (exploratory) | {display(result['mcnemar_exact_p_exploratory'])} |", "",
             f"Unpaired slots: {result['unpaired_slots'] or 'none'}.", "",
             "Recorded terminal-reason transitions:", "", "| Base reason | Candidate reason | Episodes |", "|---|---|---:|"]
    for row in result["terminal_reason_transitions"]:
        a, b = (str(row[k]).replace("|", "\\|").replace("\n", " ") for k in ["base", "candidate"])
        lines.append(f"| {a} | {b} | {row['episodes']} |")
    lines += ["", "Interpretive limits:", ""] + [f"- {x}" for x in result["limitations"]]
    return "\n".join(lines) + "\n"


def verify_design(path):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if sha256(path) != expected:
        raise ValueError("Frozen protocol hash differs; record a versioned amendment")
    return json.loads(path.read_text()), expected


def freeze_runtime(protocol_path, binding, output):
    protocol, digest = verify_design(protocol_path)
    for key in protocol["runtime_binding_required_fields"]:
        value = binding.get(key)
        if key.endswith("_sha256"):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Runtime binding requires a SHA256 for {key}")
        elif key.endswith("_enabled") or key.endswith("_validated"):
            if not isinstance(value, bool):
                raise ValueError(f"Runtime binding requires a boolean for {key}")
        elif not isinstance(value, str) or not value.strip() or "pending" in value.lower():
            raise ValueError(f"Runtime binding requires an exact value for {key}")
    if not binding["runtime_control_validated"]:
        raise ValueError("Runtime control is not validated; a versioned pre-search amendment is required")
    if binding["protected_task_enabled"] and not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("protected_task_validation_sha256", ""))):
        raise ValueError("An enabled protected task needs its validation artifact SHA256")
    record = {"schema_version": 1, "study_id": protocol["study_id"], "protocol_sha256": digest,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "binding": binding,
              "note": "Binds operator-supplied evidence hashes; this command does not itself perform GPU or sandbox validation."}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-design")
    verify.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze = sub.add_parser("freeze-runtime")
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--binding", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--base-label")
    compare.add_argument("--candidate-label")
    compare.add_argument("--base-fingerprint")
    compare.add_argument("--candidate-fingerprint")
    compare.add_argument("--split", choices=["development", "heldout", "optional_protected"], required=True)
    compare.add_argument("--task")
    compare.add_argument("--instruction")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--allow-incomplete", action="store_true", help="Write diagnostic incomplete results, never mark them complete")
    args = parser.parse_args(argv)
    if args.command == "verify-design":
        protocol, digest = verify_design(args.protocol)
        print(json.dumps({"study_id": protocol["study_id"], "sha256": digest, "status": protocol["design_status"]}, indent=2))
        return 0
    if args.command == "freeze-runtime":
        freeze_runtime(args.protocol, json.loads(args.binding.read_text()), args.output)
        print(f"Frozen runtime binding: {args.output}")
        return 0
    protocol, digest = verify_design(args.protocol)
    if args.split == "optional_protected" and (not args.task or not args.instruction):
        raise ValueError("Protected analysis requires its frozen task and instruction explicitly")
    split = protocol["slots"][args.split]
    slots = range(split["start"], split["start"] + split["n"])
    task = args.task or protocol["environment"]["task"]
    instruction = args.instruction or protocol["environment"]["instruction"]
    base = read_rows(args.base, args.base_label, args.base_fingerprint)
    cand = read_rows(args.candidate, args.candidate_label, args.candidate_fingerprint)
    settings = protocol["analysis"]["bootstrap"]
    result = analyze_pairs(base, cand, slots, task=task, instruction=instruction,
                           base_fingerprint=args.base_fingerprint, candidate_fingerprint=args.candidate_fingerprint,
                           resamples=settings["resamples"], seed=settings["seed"])
    result.update(study_id=protocol["study_id"], protocol_sha256=digest, split=args.split, task=task, instruction=instruction,
                  analysis_python_version=sys.version, input_sha256={"base": sha256(args.base), "candidate": sha256(args.candidate)})
    if args.output.exists():
        raise ValueError("Output already exists; analysis reports are never overwritten")
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(markdown_report(result))
    print(json.dumps({k: result[k] for k in ["status", "paired_n", "base_success", "candidate_success", "fixed", "broken", "success_difference"]}, indent=2))
    return 0 if result["status"] == "complete" or args.allow_incomplete else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"study_analysis: {error}")
