#!/usr/bin/env python3
"""Operator readiness gate: historical practice replay before/after identity load.

Loads the policy once, uses three exact recorded prompts/noise seeds, and calls
the same batch twice. Same-process identity agreement must be exact; comparison
with historical recorded actions keeps the predeclared 0.02 tolerance. No
simulation, held-out evidence, weight proposal, or server warmup is performed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weightsmith import config as C
from weightsmith.integrity import workspace_fingerprint
from weightsmith.lab.openpi_vlm_full import _file_sha256, validate_workspace
from weightsmith.locking import GpuLock
from weightsmith.replay import ParityCheck, parity_skip_reason

HISTORICAL_TOLERANCE = 0.02


def recorded_requests(obs_path: Path, summary: dict, indices: list[int], checkpoint: Path):
    """Fail closed on provenance or malformed recordings; never skip a request."""
    slot = re.fullmatch(r"s?(\d+)", str(summary.get("slot", "")))
    if not slot or not 20000 <= int(slot[1]) < 20256:
        raise ValueError("identity probe accepts only the original practice slot range")
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("exactly three distinct request indices are required")
    with np.load(obs_path, allow_pickle=False) as archive:
        required = {"base_img", "wrist_img", "joint", "gripper", "seed", "step", "prompt", "actions"}
        if not required <= set(archive.files):
            raise ValueError(f"recording fields missing: {sorted(required - set(archive.files))}")
        arrays = {name: archive[name] for name in required}
    n = len(arrays["step"])
    if any(len(a) != n for a in arrays.values()) or any(i < 0 or i >= n for i in indices):
        raise ValueError("invalid request index or mismatched recording lengths")
    requests, rows = [], []
    for k in indices:
        prompt, seed = str(arrays["prompt"][k]), int(arrays["seed"][k])
        reason = parity_skip_reason(workspace=None, summary=summary, recorded_prompt=prompt,
                                    prompt=prompt, seed=seed, checkpoint=checkpoint)
        if reason:
            raise ValueError(f"request {k} provenance invalid: {reason}")
        if not prompt:
            raise ValueError("empty recorded prompt")
        if (arrays["base_img"][k].shape != (224, 224, 3)
                or arrays["wrist_img"][k].shape != (224, 224, 3)
                or arrays["joint"][k].shape != (7,)
                or arrays["gripper"][k].shape != (1,)
                or arrays["actions"][k].shape != (15, 8)):
            raise ValueError(f"request {k} has unexpected policy input/output shapes")
        if not all(np.isfinite(arrays[name][k]).all() for name in required - {"prompt"}):
            raise ValueError(f"request {k} has nonfinite inputs/actions")
        requests.append({"observation/exterior_image_1_left": arrays["base_img"][k],
                         "observation/wrist_image_left": arrays["wrist_img"][k],
                         "observation/joint_position": arrays["joint"][k],
                         "observation/gripper_position": arrays["gripper"][k],
                         "sampling_seed": seed, "prompt": prompt})
        rows.append({"request_index": k, "control_step": int(arrays["step"][k]),
                     "sampling_seed": seed, "prompt": prompt})
    selected = {name: arrays[name][indices] for name in sorted(required)}
    return requests, rows, selected


def compare_actions(base, identity, recorded):
    comparisons = {"base_vs_historical": ParityCheck(HISTORICAL_TOLERANCE),
                   "identity_vs_historical": ParityCheck(HISTORICAL_TOLERANCE),
                   "identity_vs_same_process_base": ParityCheck(0.0)}
    if len(base) != 3 or len(identity) != 3 or len(recorded) != 3:
        raise ValueError("all comparisons require exactly three requests")
    rows = []
    for b, i, r in zip(base, identity, recorded):
        rows.append({"base_vs_historical": comparisons["base_vs_historical"].add(b, r),
                     "identity_vs_historical": comparisons["identity_vs_historical"].add(i, r),
                     "identity_vs_same_process_base": comparisons["identity_vs_same_process_base"].add(i, b)})
    return {name: value.report() for name, value in comparisons.items()}, rows, all(
        value.require_all(3) for value in comparisons.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs=3, default=[0, 30, 59])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to reuse an output directory")
    summary_path = args.summary or args.obs.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    requests, items, selected = recorded_requests(args.obs, summary, args.indices, C.CHECKPOINT)
    before = validate_workspace(args.workspace)
    cfg = json.loads((args.workspace / "workspace.json").read_text())
    if (len(cfg["tensors"]) != 18 or before["changed_elements"] != 0
            or not cfg.get("zero_edit_allowed") or not before.get("edit_constraints")):
        parser.error("probe requires an empty restricted workspace with all 18 action down projections")
    fingerprint = workspace_fingerprint(args.workspace)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "recorded_inputs.npz", **selected)
    report = {"status": "running", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "checkpoint": str(C.CHECKPOINT.resolve()), "source_obs": str(args.obs.resolve()),
              "source_obs_sha256": _file_sha256(args.obs),
              "source_summary_sha256": _file_sha256(summary_path),
              "workspace": str(args.workspace.resolve()), "workspace_fingerprint": fingerprint,
              "items": items, "historical_tolerance": HISTORICAL_TOLERANCE, "identity_tolerance": 0.0,
              "script_sha256": _file_sha256(Path(__file__)),
              "server_source_sha256": _file_sha256(C.PKG / "batched_server.py"),
              "deployment_gpu": C.SERVER_GPU}
    start = time.monotonic()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(C.SERVER_GPU)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = C.SERVER_MEM_FRACTION
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(C.JAX_CACHE))
        from weightsmith.batched_server import Server
        with GpuLock(C.SERVER_GPU):
            server = Server()
            base = np.stack([item["actions"] for item in server.infer_batch(requests)])
            identity_report = server.load(args.workspace)
            report["identity_load"] = identity_report
            if not (identity_report.get("identity_candidate") is True
                    and identity_report.get("verified_base_slices") == 18
                    and identity_report.get("changed_elements") == 0
                    and identity_report.get("fingerprint") == fingerprint):
                raise ValueError("identity load report is incomplete or describes a weight change")
            identity = np.stack([item["actions"] for item in server.infer_batch(requests)])
        np.savez_compressed(args.output / "actions.npz", base_actions=base,
                            identity_actions=identity, recorded_actions=selected["actions"])
        checks, details, passed = compare_actions(base, identity, selected["actions"])
        report.update(comparisons=checks, per_request_comparisons=details)
        validate_workspace(args.workspace)  # rechecks immutable base file hashes
        if workspace_fingerprint(args.workspace) != fingerprint:
            raise ValueError("workspace changed during identity probe")
        if not passed:
            raise ValueError("strict identity or historical parity failed; no skipped checks are accepted")
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
