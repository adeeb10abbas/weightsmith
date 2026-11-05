#!/usr/bin/env python3
"""Operator-only, bounded development baseline and gripper-control collection.

Run through scripts/run_isolated.py. This launches one new-runtime
32-slot baseline block, with one simulator attempt per block. It does not launch editors
or inspect held-out slots. A partial block stops the run without selecting a
control. Only the original-threshold base is exported to editor evidence.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weightsmith import config as C
from weightsmith.integrity import workspace_fingerprint
from weightsmith.lab.openpi_vlm_full import _file_sha256, validate_workspace
from weightsmith.pool import run_pool
from weightsmith.lab.batch_context_v2 import read_packet
from verify_runtime import verify_binding

SLOTS = tuple(range(30016, 30048))
BLOCKS = (("base", 0.5),)
CONFIG_SOURCE = C.REPO / "configs/v2_operator_development.json"


class IncompleteBlockError(RuntimeError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    else:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(path)


def checked_identity(path):
    report = json.loads(Path(path).read_text())
    if report.get("status") != "passed":
        raise ValueError("a passed identity probe is required before development collection")
    for key, tolerance in [("base_vs_historical", .02), ("identity_vs_historical", .02),
                           ("identity_vs_same_process_base", 0.0)]:
        check = report.get("comparisons", {}).get(key, {})
        if not (check.get("checked_count") == 3 and check.get("skipped_count") == 0
                and check.get("invalid_count") == 0 and check.get("passed") is True
                and check.get("tolerance") == tolerance):
            raise ValueError(f"identity prerequisite incomplete: {key}")
    identity = report.get("identity_load", {})
    if not (identity.get("identity_candidate") is True and identity.get("verified_base_slices") == 18
            and identity.get("changed_elements") == 0):
        raise ValueError("identity load did not verify all 18 unedited slices")
    ws = Path(report["workspace"])
    if ws.resolve() != C.BASE_WS.resolve():
        raise ValueError("configured BASE_WORKSPACE differs from the verified identity workspace")
    if Path(report["checkpoint"]).resolve() != C.CHECKPOINT.resolve():
        raise ValueError("configured checkpoint differs from the identity probe")
    if workspace_fingerprint(ws) != report["workspace_fingerprint"]:
        raise ValueError("identity workspace changed after its probe")
    if validate_workspace(ws)["changed_elements"] != 0:
        raise ValueError("identity workspace now contains an edit")
    return report


def checked_config(path=CONFIG_SOURCE):
    config = json.loads(Path(path).read_text())
    expected = {"task": "StackYellowOnRedTask", "instruction": "Stack the yellow block on the red block",
                "envs": 16, "practice_seed_base": 30016, "practice_slots": 32,
                "prompt_overrides": False, "video_mode": "none"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("operator development config differs from the approved fixed protocol")
    if config.get("study", {}).get("max_attempts") != 1 or config.get("max_attempts", 1) != 1:
        raise ValueError("development collection requires max_attempts=1")
    if config.get("limits", {}).get("practice_episodes") != 32:
        raise ValueError("operator development budget must be exactly32 episodes")
    # pool/owsp consumes the top-level key; keep the nested protocol declaration.
    config["max_attempts"] = 1
    return config


def validate_block(rows, load_report, threshold, config):
    server = json.loads(load_report) if isinstance(load_report, str) else load_report
    if server.get("fingerprint") != "base" or server.get("workspace") is not None:
        raise ValueError("runtime-control collection must use the original policy weights")
    if len(rows) != 32 or {r.get("slot") for r in rows} != set(SLOTS):
        raise IncompleteBlockError("block lacks exactly one row for each of the 32 development slots")
    if any(r.get("ok") is not True for r in rows):
        raise IncompleteBlockError("block did not complete all 32 episodes; no automatic retry")
    packet_cache = {}
    runtime_hash = _file_sha256(C.PROJECT_ROOT / 'v2_runtime_binding.json')
    for row in rows:
        if row.get("fingerprint") != "base" or row.get("gripper_threshold") != threshold:
            raise ValueError(f"weight/runtime attribution mismatch at slot {row['slot']}")
        if row.get("task_name") not in (None, config["task"]):
            raise ValueError(f"task mismatch at slot {row['slot']}")
        if row.get("prompt") != config["instruction"] or not isinstance(row.get("success"), bool):
            raise ValueError(f"prompt or result type mismatch at slot {row['slot']}")
        obs = Path(row["obs"])
        if not obs.is_file():
            raise ValueError(f"missing recording at slot {row['slot']}")
        with np.load(obs, allow_pickle=False) as a:
            raw_fields = {"base_img", "wrist_img", "joint", "gripper", "seed", "step", "prompt", "actions"}
            packet_fields = {'batch_packet_id', 'batch_packet_sha256', 'batch_position',
                             'batch_real_count', 'batch_padded_count'}
            required = raw_fields | packet_fields
            if not required <= set(a.files):
                raise ValueError(f"recorded fields missing at slot {row['slot']}")
            n = len(a["step"])
            if not n or any(len(a[k]) != n for k in required):
                raise ValueError(f"recorded request counts invalid at slot {row['slot']}")
            steps = row.get('steps')
            if (type(steps) is not int or not 1 <= steps <= 900 or n != (steps + 14) // 15
                    or not np.array_equal(a['step'], np.arange(n) * 15)):
                raise ValueError('recording does not cover the full reported episode at the fixed action-chunk cadence')
            if set(map(str, a["prompt"])) != {config["instruction"]}:
                raise ValueError(f"actual recorded prompts mismatch at slot {row['slot']}")
            if np.any(a["seed"] < 0) or not np.issubdtype(a["seed"].dtype, np.integer):
                raise ValueError(f"deterministic request seeds missing at slot {row['slot']}")
            shapes = {"base_img": (n, 224, 224, 3), "wrist_img": (n, 224, 224, 3),
                      "joint": (n, 7), "gripper": (n, 1), "actions": (n, 15, 8)}
            if any(a[k].shape != shape for k, shape in shapes.items()):
                raise ValueError(f"recorded shapes invalid at slot {row['slot']}")
            if not all(np.isfinite(a[k]).all() for k in raw_fields - {"prompt"}):
                raise ValueError(f"nonfinite record at slot {row['slot']}")
            origin = ((row['slot'] // 16) * 64 + row['slot'] % 16) * 100
            if not np.array_equal(a['seed'], origin + np.arange(n)):
                raise ValueError('recorded development sampling stream differs')
            for k in range(n):
                packet_id, packet_hash = str(a['batch_packet_id'][k]), str(a['batch_packet_sha256'][k])
                if not re.fullmatch('[0-9a-f]{32}', packet_id):
                    raise ValueError('invalid batch packet identifier')
                key = (packet_id, packet_hash)
                if key not in packet_cache:
                    packet_cache[key] = read_packet(C.PROJECT_ROOT/'data/batch_packets'/packet_id, packet_hash)
                packet, requests, native_actions = packet_cache[key]
                pos = int(a['batch_position'][k])
                if (not 0 <= pos < len(requests) or int(a['batch_real_count'][k]) != len(requests)
                        or int(a['batch_padded_count'][k]) != 16
                        or packet['layout']['runtime_variant'] != 'fixed16_new_runtime'
                        or packet['identity']['runtime_manifest_sha256'] != runtime_hash
                        or packet['identity']['policy_fingerprint'] != 'base'):
                    raise ValueError('v2 packet runtime/policy/position differs')
                request = requests[pos]
                fields = {'base_img':'observation/exterior_image_1_left', 'wrist_img':'observation/wrist_image_left',
                          'joint':'observation/joint_position', 'gripper':'observation/gripper_position'}
                if (any(not np.array_equal(a[short][k], request[full]) for short, full in fields.items())
                        or int(a['seed'][k]) != request['sampling_seed'] or str(a['prompt'][k]) != request['prompt']
                        or not np.array_equal(a['actions'][k], native_actions[pos].astype(a['actions'].dtype))):
                    raise ValueError('episode request/actions disagree with the actual live batch packet')
    return server


def events(row):
    path = Path(row.get("log") or Path(row["out_dir"]) / f"log_{row['run']}_env{row['env']}.json")
    if not path.is_file():
        return []
    keep = {"OBJECT_GRABBED_SUCCESS", "OBJECT_IN_CONTAINER_SUCCESS", "TARGET_OBJECT_DROPPED",
            "WRONG_OBJECT_GRABBED_FAILURE", "GRIPPER_FULLY_CLOSED", "OBJECT_BUMPED", "MULTIPLE_OBJECTS_GRABBED"}
    raw = json.loads(path.read_text()).get("events", [])
    return [{"step": int(e["step"]), "name": e["name"], "info": e.get("info", "")[:160]}
            for e in raw if e.get("name") in keep or "subtask" in e.get("info", "").lower()][:80]


def package_block(root, rows, name, threshold, config):
    is_base = name == "base"
    directory = root / "data/episodes" if is_base else root / "runtime_controls" / name / "episodes"
    directory.mkdir(parents=True, exist_ok=True)
    label = "base" if is_base else "runtime-" + name
    records = []
    for row in sorted(rows, key=lambda r: r["slot"]):
        eid = f"{label}.base.s{row['slot']}"
        dest = directory / eid
        dest.mkdir(exist_ok=False)
        source = Path(row["obs"])
        digest = _file_sha256(source)
        shutil.move(str(source), dest / "obs.npz")
        record = {"eid": eid, "cand": label, "fingerprint": "base", "slot": f"s{row['slot']}",
                  "success": row["success"], "prompt": config["instruction"], "prompt_recorded": True,
                  "checkpoint": str(C.CHECKPOINT.resolve()), "task": config["task"],
                  "sim_task": row.get("task_name"), "sim_instruction": row.get("instruction"),
                  "steps": row.get("steps"), "score": row.get("score"), "reason": row.get("reason"),
                  "events": events(row), "video": [], "obs_sha256": digest,
                  "gripper_threshold": threshold,
                  "intervention_kind": "original_policy" if is_base else "runtime_control"}
        write_json(dest / "summary.json", record, exclusive=True)
        records.append(record)
    with (directory / "index.jsonl").open("x") as stream:
        for rec in records:
            stream.write(json.dumps({k: v for k, v in rec.items() if k not in ("events", "video")}) + "\n")
    return records


def freeze_reference(root, rows):
    write_json(root / "study/reference_base_rows.json", rows, exclusive=True)
    paths = sorted((root / "data/episodes").rglob("*")) + sorted((root / "data/batch_packets").rglob("*")) + [root / "study/reference_base_rows.json", root / "v2_runtime_binding.json"]
    manifest = {"protocol": "wsp-manipulation-batch-aware-v2", "slots": list(SLOTS), "threshold": .5,
                "files": {str(p.relative_to(root)): _file_sha256(p) for p in paths if p.is_file()}}
    write_json(root / "study/reference_base_manifest.json", manifest, exclusive=True)
    for p in paths:
        if p.is_file(): p.chmod(0o444)
    (root / "study/reference_base_manifest.json").chmod(0o444)
    return _file_sha256(root / "study/reference_base_manifest.json")


def verify_reference(root):
    root = Path(root)
    path = root / "study/reference_base_manifest.json"
    status_path = root / "study/status.json"
    if status_path.is_file():
        pinned = json.loads(status_path.read_text()).get("reference_base_manifest_sha256")
        if pinned is not None and _file_sha256(path) != pinned:
            raise ValueError("reference manifest identity changed")
    manifest = json.loads(path.read_text())
    if manifest.get("slots") != list(SLOTS) or manifest.get("threshold") != .5:
        raise ValueError("reference manifest does not describe the original development base")
    for name, digest in manifest["files"].items():
        p = root / name
        if root.resolve() not in p.resolve().parents or _file_sha256(p) != digest:
            raise ValueError(f"pinned base artifact changed: {name}")
    return manifest


def copy_reference_to_editor(source, destination):
    """Operator helper for an already configured, unused editor project.

    Copies only pinned base evidence. Runtime controls and operator budgets are
    never copied. This helper is not exposed by the editor's tool interface.
    """
    source, destination = Path(source), Path(destination)
    verify_reference(source)
    config = json.loads((destination / "config.json").read_text())
    original = json.loads((source / "config.json").read_text())
    for key in ("task", "instruction", "envs", "practice_seed_base", "practice_slots"):
        if config.get(key) != original[key]:
            raise ValueError(f"editor project {key} differs from the pinned base")
    episodes = destination / "data/episodes"
    budget = destination / "state/budget.json"
    if budget.exists() or (episodes.exists() and any(episodes.iterdir())):
        raise ValueError("refusing to reset a used editor project or overwrite evidence")
    episodes.parent.mkdir(parents=True, exist_ok=True)
    if episodes.exists(): episodes.rmdir()
    shutil.copytree(source / "data/episodes", episodes)
    shutil.copytree(source / "data/batch_packets", destination / "data/batch_packets")
    binding = destination / "v2_runtime_binding.json"
    if binding.exists() and binding.read_bytes() != (source / "v2_runtime_binding.json").read_bytes():
        raise ValueError("editor runtime binding differs from base")
    if not binding.exists(): shutil.copy2(source / "v2_runtime_binding.json", binding)
    for name, digest in json.loads((source / "study/reference_base_manifest.json").read_text())["files"].items():
        if name.startswith(("data/episodes/", "data/batch_packets/")) and _file_sha256(destination / name) != digest:
            raise ValueError(f"copied base artifact checksum mismatch: {name}")
    # New candidate records append to the index; original obs/summary files stay read-only.
    (episodes / "index.jsonl").chmod(0o644)
    budget.parent.mkdir(parents=True, exist_ok=True)
    write_json(budget, {"episodes": 0, "acts_calls": 0, "acts_items": 0, "evals": 0}, exclusive=True)
    write_json(destination / "state/operator_base_import.json",
               {"source_root": str(source.resolve()), "reference_manifest_sha256":
                _file_sha256(source / "study/reference_base_manifest.json"), "imported_at_utc": now(),
                "operator_only": True, "preloaded_episodes": 32, "editor_budget_charged": 0}, exclusive=True)


def select_control(results):
    if set(results) != {name for name, _ in BLOCKS} or any(r["completed"] != 32 for r in results.values()):
        raise ValueError("all three complete blocks are required for threshold selection")
    # Declared order resolves ties: default 0.5 first, then 0.4, then 0.6.
    name = max((n for n, _ in BLOCKS), key=lambda n: results[n]["successes"])
    return {"selected_block": name, "selected_gripper_threshold": results[name]["gripper_threshold"],
            "selection_data": "development_only", "criterion": "highest success count; ties prefer 0.5, then 0.4, then 0.6",
            "intervention_kind": "runtime_control", "policy_fingerprint": "base"}


def run(root, identity_path, *, runtime_binding, capture_validation, config_path=CONFIG_SOURCE):
    root, identity_path = Path(root).resolve(), Path(identity_path).resolve()
    isolated = os.environ.get("WSP_ISOLATED_ROOT")
    if not isolated or Path(isolated).resolve() not in root.parents:
        raise ValueError("run through run_isolated.py and place the fresh project beneath its owned root")
    if C.PROJECT_ROOT.resolve() != root:
        raise ValueError('WSP_ROOT must bind the new baseline project before importing the fork')
    if root.exists():
        raise FileExistsError("refusing to overwrite or resume an existing development root")
    config = checked_config(config_path)
    identity = checked_identity(identity_path)
    runtime_binding, capture_validation = Path(runtime_binding), Path(capture_validation)
    binding = json.loads(runtime_binding.read_text())
    validation = json.loads(capture_validation.read_text())
    if (binding.get('capture_protocol_id') != 'wsp-batch-capture-v2-fixed16'
            or binding.get('runtime_variant') != 'fixed16_new_runtime'
            or binding.get('native_server_sha256') != _file_sha256(C.PKG/'batched_server.py')
            or Path(binding.get('checkpoint_path', '')).resolve() != C.CHECKPOINT.resolve()):
        raise ValueError('new fixed16 runtime binding is required')
    if (validation.get('status') != 'passed'
            or validation.get('runtime_binding_sha256') != _file_sha256(runtime_binding)):
        raise ValueError('a passed binding-matched v2 GPU capture validation is required')
    root.mkdir(parents=True, exist_ok=False)
    for part in ["state", "study", "data/episodes", "data/acts", "data/frames", "cands", "notes"]:
        (root / part).mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config, exclusive=True)
    shutil.copy2(identity_path, root / "study/identity_report.json")
    shutil.copy2(runtime_binding, root / 'v2_runtime_binding.json')
    shutil.copy2(capture_validation, root / 'study/capture_validation.json')
    (root/'v2_runtime_binding.json').chmod(0o444)
    state = {"status": "running", "started_at_utc": now(), "operator_only": True,
             "identity_report_sha256": _file_sha256(identity_path),
             "runtime_binding_sha256": _file_sha256(runtime_binding),
             "capture_validation_sha256": _file_sha256(capture_validation),
             "config_source_sha256": _file_sha256(Path(config_path)),
             "config_sha256": _file_sha256(root / "config.json"),
             "script_sha256": _file_sha256(Path(__file__)),
             "checkpoint": identity["checkpoint"], "blocks": {}, "attempted_episodes": 0,
             "completed_episodes": 0, "reported_completed_episodes": 0,
             "max_attempts_per_block": 1}
    results = {}
    try:
        state['source_checkpoint_verification'] = verify_binding(root/'v2_runtime_binding.json', verify_checkpoint=True)
        write_json(root/'study/status.json', state)
        for name, threshold in BLOCKS:
            blockdir = root / "study/blocks" / name
            blockdir.mkdir(parents=True)
            block = {"status": "started", "started_at_utc": now(), "slots": list(SLOTS),
                     "attempt": 1, "reserved_episode_capacity": 32, "gripper_threshold": threshold,
                     "policy_fingerprint": "base", "intervention_kind": "original_policy" if name == "base" else "runtime_control"}
            state["blocks"][name] = block
            state["attempted_episodes"] += 32
            write_json(root / "state/budget.json", {"episodes": state["attempted_episodes"],
                       "attempted_episodes": state["attempted_episodes"], "evals": len(state["blocks"]),
                       "acts_calls": 0, "acts_items": 0, "operator_only": True})
            write_json(blockdir / "status.json", block)
            write_json(root / "study/status.json", state)
            print(json.dumps({"block": name, **block}), flush=True)
            start = time.monotonic()
            rows, server = run_pool(workspace=None, task=config["task"], tag=f"rq123-v2-dev-{name}",
                                    slots=SLOTS, envs=16, prompt=config["instruction"],
                                    rendering_type=config["rendering_type"], video_mode="none",
                                    gripper_threshold=threshold, max_attempts=1, log_root=blockdir / "pool")
            write_json(blockdir / "raw_pool_rows.json", {"rows": rows, "server": server}, exclusive=True)
            block["reported_completed"] = sum(row.get("ok") is True for row in rows)
            state["reported_completed_episodes"] += block["reported_completed"]
            validate_block(rows, server, threshold, config)
            packaged = package_block(root, rows, name, threshold, config)
            write_json(blockdir / "rows.json", packaged, exclusive=True)
            block.update(status="completed", completed=32, successes=sum(r["success"] for r in packaged),
                         elapsed_seconds=time.monotonic() - start, finished_at_utc=now(),
                         rows_sha256=_file_sha256(blockdir / "rows.json"))
            if name == "base":
                state["reference_base_manifest_sha256"] = freeze_reference(root, packaged)
            else:
                verify_reference(root)
                base = {r["slot"]: r["success"] for r in json.loads((root / "study/reference_base_rows.json").read_text())}
                block.update(fixed=sum(not base[r["slot"]] and r["success"] for r in packaged),
                             broken=sum(base[r["slot"]] and not r["success"] for r in packaged))
            results[name] = dict(block)
            state["completed_episodes"] += 32
            write_json(blockdir / "status.json", block)
            write_json(root / "study/status.json", state)
            print(json.dumps({"block": name, **block}), flush=True)
        state.update(status="completed", no_threshold_tuning=True, finished_at_utc=now())
    except BaseException as error:
        state.update(status="incomplete" if isinstance(error, IncompleteBlockError) else "failed",
                     error=f"{type(error).__name__}: {error}", finished_at_utc=now())
        if state["blocks"]:
            block.update(status=state["status"], error=state["error"], finished_at_utc=now())
            write_json(blockdir / "status.json", block)
        raise
    finally:
        write_json(root / "study/status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--capture-validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.identity_report, runtime_binding=args.runtime_binding, capture_validation=args.capture_validation), indent=2))


if __name__ == "__main__":
    main()
