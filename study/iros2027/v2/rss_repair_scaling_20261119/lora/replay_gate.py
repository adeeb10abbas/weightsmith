"""Certify the first fresh source block before spending more collection budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import fresh_lora as F
from run import PYTHON, completed


def gate(args):
    source = Path(args.source_manifest).resolve()
    project = source.parents[2]
    collection = json.loads((project / "collection-status.json").read_text())
    if collection.get("status") != "complete":
        raise ValueError("Source collection is not complete; do not compete for its GPU")
    manifest = json.loads(source.read_text())
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"source_manifest": str(source), "source_manifest_sha256": F.sha(source),
                "cohort_id": manifest["cohort_id"], "eligible_pairs": manifest["eligible_pairs"],
                "attempted_pairs": manifest["attempted_pairs"], "gpu": args.gpu}
    if 1 not in manifest["correction_budgets_available"]:
        F.dump(root / "status.json", {**identity, "status": "no_eligible_pairs"})
        raise RuntimeError("No eligible source pair; no target data or replacement repair was manufactured")
    gpu_uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(args.gpu)], text=True).strip()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    if any(line.split(",")[0].strip() == gpu_uuid for line in active.splitlines()):
        raise RuntimeError("Chosen GPU still has a compute process; replay has not started")
    if (root / "passed.json").exists():
        raise FileExistsError("Replay gate already has a completion receipt")
    start = time.monotonic()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(F.REPO),
           "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90", "PYTHONUNBUFFERED": "1"}
    reports = {}
    artifacts = {str(path.resolve()): F.sha(path) for path in (source, Path(__file__), F.HERE / "fresh_lora.py", F.HERE / "run.py")}
    try:
        for key, arm, name in (("full_trajectory", "full_trajectory", "full-trajectory-n1"),
                               ("correction", "matched_decision", "correction-n1")):
            F.dump(root / "status.json", {**identity, "status": "running", "stage": arm})
            spec = Path(manifest["specs"]["1"][key])
            target = root / (name + ("-" + args.attempt_tag if args.attempt_tag else ""))
            command = [str(PYTHON), str(F.HERE / "fresh_lora.py"), "prepare", "--spec", str(spec),
                       "--arm", arm, "--output", str(target), "--gpu", str(args.gpu)]
            completed(target, "dataset.json", command, env)
            _, meta = F.load_dataset(target)
            stored = json.loads((target / "input-spec.json").read_text())
            if stored["source_manifest_sha256"] != identity["source_manifest_sha256"]:
                raise ValueError("Prepared labels belong to another source manifest")
            reports[arm] = {"dataset": str(target / "dataset.json"), "dataset_sha256": F.sha(target / "dataset.json"),
                            "items": meta["train_inputs"], "live_replay_gate": meta["live_replay_gate"],
                            "normalization_gate": meta["normalization_gate"]}
            artifacts.update(meta["source_artifacts"])
            artifacts.update(meta["live_replay_context_hashes"])
            for path in (target / "dataset.json", target / "input-spec.json", spec):
                artifacts[str(path.resolve())] = F.sha(path)
        if F.sha(source) != identity["source_manifest_sha256"]:
            raise ValueError("Source manifest changed during replay")
        for path, expected in artifacts.items():
            if F.sha(path) != expected:
                raise ValueError(f"Replay gate artifact changed: {path}")
        result = {**identity, "status": "passed", "datasets": reports, "artifacts": artifacts, "wall_seconds": time.monotonic() - start,
                  "interpretation": "Native replay and label preparation only; no fitting or edited-policy behavior evaluated"}
        F.dump(root / "passed.json", result)
        F.dump(root / "status.json", {**identity, "status": "complete", "stage": "passed", "wall_seconds": result["wall_seconds"]})
        print(json.dumps(result), flush=True)
    except Exception as error:
        F.dump(root / "status.json", {**identity, "status": "failed", "error": str(error), "wall_seconds": time.monotonic() - start})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    p.add_argument("--attempt-tag", default="")
    gate(p.parse_args())


if __name__ == "__main__":
    main()
