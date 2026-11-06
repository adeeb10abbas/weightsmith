#!/usr/bin/env python3
"""Operator-only frozen RQ1-RQ3 scoring; never exposed by the editor broker.

Run under run_isolated.py with a fresh owned project and a complete frozen
manifest. Nine fresh paired comparisons plus one 64-episode base repeat total
1,216 episode capacity. No retry, resume, optional task, or second editor exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weightsmith import config as C
from weightsmith.integrity import workspace_fingerprint
from weightsmith.lab.openpi_vlm_full import validate_workspace
from weightsmith.pool import run_pool
from scripts.study_analysis import analyze_pairs, markdown_report, sha256, verify_design, DEFAULT_PROTOCOL
from scripts.study_development import checked_identity, now, write_json

IDS = tuple(f"adaptive-r{restart}-round{round_}" for restart in (1, 2) for round_ in (1, 2, 3)) + (
    "nonadaptive-r1-final", "nonadaptive-r2-final", "gripper-control")
SLOTS = tuple(range(90000, 90064))
DEVELOPMENT = tuple(range(20032, 20064))
TASK = "StackYellowOnRedTask"
INSTRUCTION = "Stack the yellow block on the red block"
HASH = re.compile(r"^[0-9a-f]{64}$")


class IncompleteBlockError(RuntimeError):
    pass


def owned(path, isolation_root, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or isolation_root not in path.resolve().parents:
        raise ValueError(f"Path must be owned and beneath the isolated deployment: {path}")
    if directory and not path.is_dir():
        raise ValueError(f"Missing owned directory: {path}")
    if not directory and not path.is_file():
        raise ValueError(f"Missing owned file: {path}")
    return path.resolve()


def artifact(reference, isolation_root):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An artifact reference needs exactly path and sha256")
    path = owned(reference["path"], isolation_root)
    if not HASH.fullmatch(str(reference["sha256"])) or sha256(path) != reference["sha256"]:
        raise ValueError(f"Frozen artifact checksum mismatch: {path}")
    return path, json.loads(path.read_text())


def checked_source(entry, isolation_root):
    source = entry["source_workspace"]
    if source is None:
        if entry["fingerprint"] != "base" or entry["kind"] not in ("no_edit", "runtime_control"):
            raise ValueError("A base/no-edit state needs null workspace and fingerprint 'base'")
        if entry["metadata"]["incumbent"] != "base":
            raise ValueError("A null workspace must identify the base incumbent")
        return None, {"changed_elements": 0, "changed_rows": 0, "changed_layers": [], "l2_delta": 0.0}
    if entry["kind"] != "weight_edit" or not HASH.fullmatch(str(entry["fingerprint"])):
        raise ValueError("An edited source needs kind weight_edit and an exact fingerprint")
    source = owned(source, isolation_root, directory=True)
    if (source / "workspace.json").is_symlink() or (source / "edits").is_symlink():
        raise ValueError("Candidate manifest and edit directory cannot be symlinks")
    manifest = json.loads((source / "workspace.json").read_text())
    if len(manifest.get("tensors", {})) != 18 or not manifest.get("edit_constraints"):
        raise ValueError("Candidate requires all 18 restricted action-down-projection slices")
    if (source / "base").resolve() != (C.BASE_WS / "base").resolve():
        raise ValueError("Candidate does not reference the verified original export")
    if any(p.is_symlink() or not p.is_file() or p.suffix != ".npy" for p in (source / "edits").iterdir()):
        raise ValueError("Candidate edits must contain only regular .npy files")
    report = validate_workspace(source)
    if report.get("changed_elements", 0) <= 0:
        raise ValueError("An identity candidate is base/no-edit, not an eligible repair incumbent")
    if workspace_fingerprint(source) != entry["fingerprint"]:
        raise ValueError("Candidate fingerprint differs from frozen manifest")
    return source, report


def check_manifest(path, isolation_root, protocol_path=DEFAULT_PROTOCOL):
    path = owned(path, isolation_root)
    manifest = json.loads(path.read_text())
    protocol, protocol_hash = verify_design(protocol_path)
    if manifest.get("schema_version") != 1 or manifest.get("study_id") != protocol["study_id"]:
        raise ValueError("Unknown final-scoring manifest schema or study")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("Final manifest does not bind the current frozen protocol")
    if (manifest.get("task") != TASK or manifest.get("instruction") != INSTRUCTION
            or manifest.get("envs") != 16 or manifest.get("development_slots") != list(DEVELOPMENT)
            or manifest.get("heldout_slots") != list(SLOTS)):
        raise ValueError("Task, instruction, environment layout or fixed slot sets differ from the protocol")
    if set(manifest["development_slots"]) & set(manifest["heldout_slots"]):
        raise ValueError("Development and held-out slots overlap")
    runtime_path, runtime = artifact(manifest["runtime_binding"], isolation_root)
    if runtime.get("study_id") != protocol["study_id"] or runtime.get("protocol_sha256") != protocol_hash:
        raise ValueError("Runtime binding belongs to a different study/protocol")
    binding = runtime.get("binding", {})
    for key in protocol["runtime_binding_required_fields"]:
        value = binding.get(key)
        if key.endswith("_sha256"):
            if not HASH.fullmatch(str(value)):
                raise ValueError(f"Missing runtime evidence hash: {key}")
        elif key.endswith(("_enabled", "_validated")):
            if type(value) is not bool:
                raise ValueError(f"Missing runtime boolean: {key}")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing exact runtime model/setting: {key}")
    if not binding["runtime_control_validated"]:
        raise ValueError("The calibrated runtime control was not validated")
    identity_path, _ = artifact(manifest["identity_report"], isolation_root)
    if sha256(identity_path) != binding["zero_edit_parity_report_sha256"]:
        raise ValueError("Identity report differs from runtime binding")
    checked_identity(identity_path)
    calibration_path, calibration = artifact(manifest["calibration_selection"], isolation_root)
    threshold = calibration.get("selected_gripper_threshold")
    if (type(threshold) not in (int, float) or threshold not in (.4, .5, .6)
            or calibration.get("selection_data") != "development_only"
            or calibration.get("policy_fingerprint") != "base"):
        raise ValueError("Invalid development-only calibration selection")
    entries = manifest.get("comparisons")
    if not isinstance(entries, list) or [e.get("id") for e in entries] != list(IDS):
        raise ValueError("Manifest must contain exactly the canonical nine comparison IDs, in order")
    sources = {}
    for entry in entries:
        name = entry["id"]
        meta = entry.get("metadata", {})
        if set(meta) != {"source_run", "mode", "restart", "round", "incumbent"}:
            raise ValueError(f"Missing curve/source metadata for {name}")
        if not isinstance(meta["source_run"], str) or not meta["source_run"] or meta["incumbent"] not in ("base", "p1", "p2", "p3"):
            raise ValueError(f"Invalid source run/incumbent for {name}")
        if name.startswith("adaptive-"):
            expected = ("adaptive", int(name[10]), int(name[-1]))
        elif name.startswith("nonadaptive-"):
            expected = ("nonadaptive", int(name[13]), None)
        else:
            expected = ("control", None, None)
        if (meta["mode"], meta["restart"], meta["round"]) != expected:
            raise ValueError(f"Curve metadata disagrees with comparison ID {name}")
        expected_threshold = float(threshold) if name == "gripper-control" else .5
        if type(entry.get("gripper_threshold")) not in (int, float) or entry["gripper_threshold"] != expected_threshold:
            raise ValueError(f"Runtime threshold mismatch for {name}")
        if name == "gripper-control" and (entry.get("kind") != "runtime_control" or entry.get("source_workspace") is not None):
            raise ValueError("Gripper calibration must use original weights")
        if name != "gripper-control" and entry.get("kind") not in ("weight_edit", "no_edit"):
            raise ValueError("Only sparse edits or retained base states belong in policy comparisons")
        sources[name] = checked_source(entry, isolation_root)
    return manifest, sources, {"runtime_binding": runtime_path, "identity_report": identity_path,
                               "calibration_selection": calibration_path, "protocol": Path(protocol_path).resolve()}


def snapshot_all(root, manifest, sources, inputs):
    directory = root / "frozen"; directory.mkdir()
    for name, path in inputs.items():
        shutil.copy2(path, directory / f"{name}.json")
    frozen = {}
    for entry in manifest["comparisons"]:
        name = entry["id"]; source, report = sources[name]
        dest = None
        if source is not None:
            dest = directory / "workspaces" / name
            dest.mkdir(parents=True)
            shutil.copy2(source / "workspace.json", dest / "workspace.json")
            shutil.copytree(source / "edits", dest / "edits")
            # The shared, checksum-verified original export is read-only. Copy
            # sparse edits and manifests, never duplicate the entire base export.
            (dest / "base").symlink_to((C.BASE_WS / "base").resolve())
            (dest / "scratch").mkdir()
            if workspace_fingerprint(dest) != entry["fingerprint"] or workspace_fingerprint(source) != entry["fingerprint"]:
                raise ValueError("Candidate changed while snapshotting all policies")
            validate_workspace(dest)
        frozen[name] = {**entry, "snapshot_workspace": str(dest) if dest else None, "validation": report}
    write_json(directory / "manifest.json", manifest, exclusive=True)
    write_json(directory / "snapshots.json", frozen, exclusive=True)
    files = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(directory.rglob("*")) if p.is_file() and not p.is_symlink()}
    write_json(directory / "files.json", files, exclusive=True)
    for p in directory.rglob("*"):
        if p.is_file() and not p.is_symlink(): p.chmod(0o444)
    return frozen, sha256(directory / "files.json")


def verify_snapshots(root, frozen, files_hash):
    file = root / "frozen/files.json"
    if sha256(file) != files_hash:
        raise ValueError("Frozen snapshot inventory changed")
    for name, record in json.loads(file.read_text()).items():
        path = root / name
        if path.is_symlink() or root not in path.resolve().parents or sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen snapshot file changed: {name}")
    identity = validate_workspace(C.BASE_WS)
    if identity.get("changed_elements") != 0:
        raise ValueError("Verified original export acquired an edit")
    for entry in frozen.values():
        workspace = entry["snapshot_workspace"]
        if workspace_fingerprint(workspace) != entry["fingerprint"]:
            raise ValueError("Frozen candidate identity changed")
        if workspace:
            validate_workspace(Path(workspace))  # includes original export checksums before each load.


def validate_block(rows, server, slots, entry, *, arm, record_root=None, isolation_root=None):
    expected_fp = "base" if arm == "base" else entry["fingerprint"]
    workspace = None if arm == "base" else entry["snapshot_workspace"]
    threshold = .5 if arm == "base" else entry["gripper_threshold"]
    server = json.loads(server) if isinstance(server, str) else server
    if server.get("fingerprint") != expected_fp or server.get("workspace") != workspace:
        raise ValueError("Server did not load the exact frozen policy requested")
    if len(rows) != len(slots) or {r.get("slot") for r in rows} != set(slots):
        raise IncompleteBlockError("Block must return one row for each scheduled slot")
    for row in rows:
        if type(row.get("ok")) is not bool:
            raise ValueError("Completion field must be boolean")
        if not row["ok"]: continue
        if (row.get("fingerprint") != expected_fp or row.get("gripper_threshold") != threshold
                or row.get("prompt") != INSTRUCTION or row.get("task_name") not in (None, TASK)
                or type(row.get("success")) is not bool):
            raise ValueError(f"Policy/runtime/task attribution differs at slot {row['slot']}")
        path = Path(row["obs"])
        if record_root is not None and (path.is_symlink() or record_root not in path.resolve().parents):
            raise ValueError("Held-out observations must stay inside the operator-only final project")
        if row.get("out_dir") and isolation_root is not None and isolation_root not in Path(row["out_dir"]).resolve().parents:
            raise ValueError("Simulator output reference escapes the owned deployment")
        with np.load(path, allow_pickle=False) as data:
            required = {"base_img", "wrist_img", "joint", "gripper", "actions", "seed", "step", "prompt"}
            if not required <= set(data.files): raise ValueError("Missing exact recorded policy inputs")
            n = len(data["step"])
            if not n or any(len(data[k]) != n for k in required): raise ValueError("Recorded request counts differ")
            shapes = {"base_img": (n, 224, 224, 3), "wrist_img": (n, 224, 224, 3),
                      "joint": (n, 7), "gripper": (n, 1), "actions": (n, 15, 8)}
            if any(data[k].shape != shape for k, shape in shapes.items()): raise ValueError("Invalid recorded shape")
            if set(map(str, data["prompt"])) != {INSTRUCTION}: raise ValueError("Actual recorded prompt changed")
            origin = ((row["slot"] // 16) * 64 + row["slot"] % 16) * 100
            if not np.issubdtype(data["seed"].dtype, np.integer) or not np.array_equal(data["seed"], origin + np.arange(n)):
                raise ValueError("Recorded sampling stream differs from the paired slot")
            if not all(np.isfinite(data[k]).all() for k in required - {"prompt"}): raise ValueError("Nonfinite recording")
        row["obs_sha256"] = sha256(path)
    if any(row["ok"] is not True for row in rows):
        raise IncompleteBlockError("Incomplete block: stop without retry or advancing to the next comparison")


def paired_output(directory, rows, entry, protocol, *, status):
    settings = protocol["analysis"]["bootstrap"]
    analysis = analyze_pairs(rows["base"], rows["candidate"], SLOTS, task=TASK, instruction=INSTRUCTION,
                             base_fingerprint="base", candidate_fingerprint=entry["fingerprint"],
                             resamples=settings["resamples"], seed=settings["seed"])
    analysis.update(comparison_id=entry["id"], comparison_status=status, metadata=entry["metadata"],
                    policy_fingerprint=entry["fingerprint"], gripper_threshold=entry["gripper_threshold"],
                    intervention_kind=entry["kind"], changed_elements=entry["validation"]["changed_elements"])
    write_json(directory / "paired_rows.json", rows)
    write_json(directory / "analysis.json", analysis)
    (directory / "report.md").write_text(markdown_report(analysis))
    return analysis


def archival_manifest(root, directory, rows, *, complete):
    refs = sorted({r["out_dir"] for arm in rows.values() for r in arm if r.get("out_dir")})
    inventory = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "archivable_manifest.json"}
    value = {"schema_version": 1, "operator_only": True, "complete": complete,
             "safe_to_prune_raw_observations_only_after_external_archive_verification": complete,
             "files": inventory, "raw_simulator_output_references": refs,
             "note": "The runner never archives or deletes data. Retain paired_rows.json/analysis/status; later analysis and the base repeat do not reread old observations."}
    write_json(directory / "archivable_manifest.json", value, exclusive=True)
    (directory / "archivable_manifest.json").chmod(0o444)
    return sha256(directory / "archivable_manifest.json")


def publish_index(root, state):
    write_json(root / "study/status.json", state)
    write_json(root / "study/results_index.json", {"operator_only": True, "status": state["status"],
                "reserved_episode_capacity": state["reserved_episode_capacity"],
                "validated_completed_episodes": state["validated_completed_episodes"],
                "comparisons": state["comparisons"]})
    lines = ["# Operator-only RQ1-RQ3 final scoring", "", f"Status: **{state['status']}**.", "",
             "| Comparison | Status | Pairs | Base / candidate successes | Fixed / broken |", "|---|---|---:|---:|---:|"]
    for name in (*IDS, "base-repeat"):
        row = state["comparisons"].get(name, {})
        stats = row.get("statistics", {})
        lines.append(f"| {name} | {row.get('status', 'pending')} | {stats.get('paired_n', '-')} | {stats.get('base_success', '-')} / {stats.get('candidate_success', '-')} | {stats.get('fixed', '-')} / {stats.get('broken', '-')} |")
    lines += ["", "Only complete frozen comparisons support outcome statements. Missing rows remain explicit; no retry or post-test policy selection is performed.", ""]
    (root / "study/summary.md").write_text("\n".join(lines))


def run(root, frozen_path, *, protocol_path=DEFAULT_PROTOCOL):
    raw_root = Path(root)
    isolation = os.environ.get("WSP_ISOLATED_ROOT")
    if not isolation or os.environ.get("WSP_MANAGED_PROCESS_GROUP") != "1":
        raise ValueError("Final scoring requires run_isolated.py context")
    isolation = Path(isolation).resolve()
    root = raw_root.resolve()
    if not raw_root.is_absolute() or raw_root.is_symlink() or isolation not in root.parents:
        raise ValueError("Final project must be a fresh absolute path beneath the owned deployment")
    if root.exists(): raise FileExistsError("Existing final roots cannot be overwritten or automatically resumed")
    manifest, sources, inputs = check_manifest(frozen_path, isolation, protocol_path)
    protocol, protocol_hash = verify_design(protocol_path)
    root.mkdir(parents=True, mode=0o700)
    (root / "study").mkdir(); (root / "state").mkdir()
    write_json(root / "config.json", {"task": TASK, "instruction": INSTRUCTION, "envs": 16,
        "rendering_type": "performance", "video_mode": "none", "max_attempts": 1,
        "prompt_overrides": False, "practice_seed_base": 20032, "practice_slots": 0,
        "limits": {"practice_episodes": 0, "acts_items": 0, "acts_items_per_call": 0},
        "operator_only": True, "heldout_episode_capacity": 1216}, exclusive=True)
    state = {"status": "freezing", "operator_only": True, "started_at_utc": now(), "comparisons": {},
             "blocks": [], "reserved_episode_capacity": 0, "reported_completed_episodes": 0,
             "validated_completed_episodes": 0, "planned_episode_capacity": 1216,
             "input_manifest_sha256": sha256(frozen_path), "protocol_sha256": protocol_hash,
             "script_sha256": sha256(__file__), "config_sha256": sha256(root / "config.json"),
             "max_attempts_per_block": 1}
    publish_index(root, state)
    active = None
    try:
        frozen, files_hash = snapshot_all(root, manifest, sources, inputs)
        state["snapshot_inventory_sha256"] = files_hash
        verify_snapshots(root, frozen, files_hash)
        state["status"] = "running"; publish_index(root, state)
        base_reference = None
        base_reference_hash = None
        for name in (*IDS, "base-repeat"):
            repeat = name == "base-repeat"
            entry = frozen[name] if not repeat else {"id": name, "fingerprint": "base", "snapshot_workspace": None,
                "gripper_threshold": .5, "kind": "base_repeat", "validation": {"changed_elements": 0},
                "metadata": {"source_run": IDS[0], "mode": "base_repeat", "restart": None, "round": None, "incumbent": "base"}}
            directory = root / "comparisons" / name; directory.mkdir(parents=True)
            rows = {"base": [], "candidate": []}
            if repeat:
                if sha256(base_reference) != base_reference_hash:
                    raise ValueError("Pinned first-comparison base rows changed before the repeat")
                rows["base"] = json.loads(base_reference.read_text())["base"]
            comparison = {"status": "running", "metadata": entry["metadata"], "fingerprint": entry["fingerprint"],
                          "gripper_threshold": entry["gripper_threshold"], "kind": entry["kind"], "started_at_utc": now()}
            state["comparisons"][name] = comparison
            schedule = [("candidate", SLOTS)] if repeat else [("base", SLOTS[:32]), ("candidate", SLOTS[:32]), ("candidate", SLOTS[32:]), ("base", SLOTS[32:])]
            for index, (arm, slots) in enumerate(schedule):
                verify_snapshots(root, frozen, files_hash)
                blockdir = directory / "blocks" / f"{index + 1:02d}-{arm}"; blockdir.mkdir(parents=True)
                block = {"comparison": name, "arm": arm, "status": "started", "slots": list(slots),
                         "reserved_episode_capacity": len(slots), "attempt": 1, "started_at_utc": now()}
                active = (block, blockdir, comparison, directory, rows, entry)
                state["blocks"].append(block); state["reserved_episode_capacity"] += len(slots)
                write_json(blockdir / "status.json", block); publish_index(root, state)
                write_json(root / "state/ledger.json", {"reserved_episode_capacity": state["reserved_episode_capacity"], "blocks": state["blocks"], "operator_only": True})
                print(json.dumps({"event": "block_started", **block}), flush=True)
                start = time.monotonic()
                workspace = None if arm == "base" else entry["snapshot_workspace"]
                threshold = .5 if arm == "base" else entry["gripper_threshold"]
                returned, server = run_pool(workspace=workspace, task=TASK, tag=f"rq123-ho-{name}-{index + 1}",
                    slots=slots, envs=16, env_seed=0, prompt=INSTRUCTION, rendering_type="performance", video_mode="none",
                    gripper_threshold=threshold, max_attempts=1, log_root=blockdir / "pool")
                write_json(blockdir / "raw_pool_rows.json", {"rows": returned, "server": server}, exclusive=True)
                block["reported_completed"] = sum(r.get("ok") is True for r in returned)
                state["reported_completed_episodes"] += block["reported_completed"]
                try:
                    validate_block(returned, server, slots, entry, arm=arm, record_root=root, isolation_root=isolation)
                except IncompleteBlockError:
                    rows[arm].extend(returned)
                    paired_output(directory, rows, entry, protocol, status="incomplete")
                    raise
                rows[arm].extend(returned)
                for row in returned:
                    log = Path(row.get("log", ""))
                    if log.is_file() and isolation in log.resolve().parents:
                        target = blockdir / "terminal_logs"; target.mkdir(exist_ok=True)
                        shutil.copy2(log, target / f"slot-{row['slot']}.json")
                write_json(blockdir / "rows.json", returned, exclusive=True)
                block.update(status="completed", validated_completed=len(slots), elapsed_seconds=time.monotonic() - start, finished_at_utc=now())
                state["validated_completed_episodes"] += len(slots)
                write_json(blockdir / "status.json", block)
                write_json(directory / "partial_rows.json", rows)
                publish_index(root, state)
            verify_snapshots(root, frozen, files_hash)
            result = paired_output(directory, rows, entry, protocol, status="complete")
            if result["status"] != "complete": raise IncompleteBlockError("Comparison does not contain all 64 valid pairs")
            comparison.update(status="complete", finished_at_utc=now(),
                statistics={k: result[k] for k in ("paired_n", "base_success", "candidate_success", "fixed", "broken", "success_difference", "paired_bootstrap_95")},
                analysis=str((directory / "analysis.json").relative_to(root)))
            comparison["paired_rows_sha256"] = sha256(directory / "paired_rows.json")
            write_json(directory / "status.json", comparison)
            comparison["archivable_manifest_sha256"] = archival_manifest(root, directory, rows, complete=True)
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink(): path.chmod(0o444)
            publish_index(root, state)
            print(json.dumps({"event": "comparison_complete", "comparison": name, **comparison}), flush=True)
            if name == IDS[0]:
                base_reference = directory / "paired_rows.json"
                base_reference_hash = comparison["paired_rows_sha256"]
            active = None
        if state["reserved_episode_capacity"] != 1216 or state["validated_completed_episodes"] != 1216:
            raise RuntimeError("Final episode accounting differs from the frozen core schedule")
        state.update(status="complete", finished_at_utc=now())
    except BaseException as error:
        state.update(status="incomplete" if isinstance(error, IncompleteBlockError) else "failed",
                     error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        if active:
            block, blockdir, comparison, directory, rows, entry = active
            block.update(status=state["status"], error=state["error"], finished_at_utc=now())
            comparison.update(status=state["status"], error=state["error"])
            write_json(blockdir / "status.json", block); write_json(directory / "status.json", comparison)
            write_json(directory / "partial_rows.json", rows)
        raise
    finally:
        write_json(root / "state/ledger.json", {"reserved_episode_capacity": state["reserved_episode_capacity"], "blocks": state["blocks"], "operator_only": True})
        publish_index(root, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.frozen), indent=2))


if __name__ == "__main__":
    main()
