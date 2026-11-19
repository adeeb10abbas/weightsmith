"""Run the frozen two-rate full-self-demonstration LoRA comparison, serially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import fresh_lora as F

PYTHON = Path("/home/ali/openpi-robolab/.venv/bin/python")


def verify(path, expected):
    if F.sha(path) != expected:
        raise ValueError(f"Changed source or candidate artifact: {path}")


def completed(path, receipt, command, env):
    """Only resume an already successful stage; partial failures stay untouched."""
    path = Path(path)
    if path.exists():
        if not (path / receipt).is_file():
            raise RuntimeError(f"Incomplete prior stage preserved at {path}; refusing an automatic retry")
        return
    with (path.parent / (path.name + ".log")).open("x") as log:
        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not (path / receipt).is_file():
        raise RuntimeError(f"Stage exited without its completion receipt: {path}")


def run(args):
    manifest_path = Path(args.source_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "development":
        raise ValueError("LoRA may consume only a development source manifest")
    available = manifest["correction_budgets_available"]
    budgets = available if args.budget is None else [args.budget]
    if not budgets or any(n not in F.COUNTS or n not in available for n in budgets):
        raise ValueError("Requested source budget is unavailable")
    for name, expected in manifest.get("artifacts", {}).items():
        verify(name, expected)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {"cohort_id": manifest["cohort_id"], "source_manifest": str(manifest_path),
                "source_manifest_sha256": F.sha(manifest_path), "available_budgets": available,
                "trainer_sha256": F.sha(F.HERE / "fresh_lora.py"), "runner_sha256": F.sha(__file__)}
    identity_path = root / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Adaptation output belongs to a different source or code identity")
    else:
        F.dump(identity_path, identity)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(F.REPO),
           "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90", "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()
    base = [str(PYTHON), str(F.HERE / "fresh_lora.py")]
    try:
        for n in budgets:
            prefix = root / f"n{n}"
            prefix.mkdir(exist_ok=True)
            spec = Path(manifest.get("specs", {}).get(str(n), {}).get("full_trajectory", manifest_path.parent / f"n{n}/full_trajectory.json"))
            if json.loads(spec.read_text()).get("source_manifest_sha256") != identity["source_manifest_sha256"]:
                raise ValueError("Full-trajectory spec is not owned by this source manifest")
            dataset = prefix / "dataset"
            F.dump(root / "status.json", {**identity, "status": "running", "stage": "preparing", "budget": n})
            completed(dataset, "dataset.json", base + ["prepare", "--spec", str(spec), "--output", str(dataset), "--gpu", str(args.gpu)], env)
            F.load_dataset(dataset)
            if n == budgets[0]:
                smoke = prefix / "smoke"
                completed(smoke, "smoke.json", base + ["smoke", "--dataset", str(dataset), "--output", str(smoke), "--gpu", str(args.gpu)], env)
            for rate in F.LEARNING_RATES:
                fit = prefix / f"lr-{rate:g}"
                F.dump(root / "status.json", {**identity, "status": "running", "stage": "training", "budget": n, "learning_rate": rate})
                completed(fit, "seal.json", base + ["train", "--dataset", str(dataset), "--output", str(fit), "--gpu", str(args.gpu), "--lr", str(rate)], env)
            publish(root, identity)
        F.dump(root / "status.json", {**identity, "status": "complete", "stage": "complete", "requested_budgets": budgets,
                                     "wall_seconds": time.monotonic() - start})
    except Exception as error:
        F.dump(root / "status.json", {**identity, "status": "failed", "stage": "failed", "error": str(error),
                                     "wall_seconds": time.monotonic() - start})
        raise


def publish(root, identity):
    candidates, finished = [], []
    for n in identity["available_budgets"]:
        seals = [root / f"n{n}/lr-{rate:g}/seal.json" for rate in F.LEARNING_RATES]
        if not all(p.is_file() for p in seals):
            continue
        for path, rate in zip(seals, F.LEARNING_RATES, strict=True):
            seal = json.loads(path.read_text())
            if seal["correction_count"] != n or seal["learning_rate"] != rate or seal["arm"] != "full_trajectory":
                raise ValueError("Candidate identity differs from the frozen rate/budget grid")
            for file, expected in seal["artifacts"].items():
                verify(file, expected)
            candidates.append({"candidate_id": seal["candidate_id"], "budget": n, "learning_rate": rate,
                               "workspace": seal["workspace"], "training_path": str(path.parent / "training.json"),
                               "artifacts": {**seal["artifacts"], str(path): F.sha(path)},
                               "source_manifest": identity["source_manifest"],
                               "source_manifest_sha256": identity["source_manifest_sha256"]})
        finished.append(n)
    F.dump(root / "candidates.json", {**identity, "completed_budgets": finished, "candidates": candidates})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    p.add_argument("--budget", type=int, choices=F.COUNTS)
    run(p.parse_args())


if __name__ == "__main__":
    main()
