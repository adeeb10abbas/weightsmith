"""Launch a new, sealed target-ablation case through the unchanged fixed16 worker.

This entry point refuses any existing result directory, including incomplete
attempts. Diagnosis and an explicitly new attempt are required after failure.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from pipeline import HERE, PRIOR, read, require, sha, module
from plan import check_plan


def run(plan_path, root, gpu):
    root = Path(root).resolve()
    require(not root.exists(), "Refusing to rerun an existing case directory; preserve and diagnose the prior attempt")
    require(root.is_relative_to(HERE.parents[3] / "projects/rss-residual-targets-20261117"),
            "Ablation outputs must stay in their separately named project")
    protocol = read(HERE / "protocol.json")
    plan = read(plan_path)
    check_plan(plan, protocol)
    require(plan["protocol_sha256"] == sha(HERE / "protocol.json"), "Protocol changed after plan creation")
    worker = module("residual_target_sealed_worker", PRIOR / "worker.py")
    worker.verify_global_freeze(plan)
    return subprocess.run([sys.executable, str(PRIOR / "worker.py"), "--plan", str(Path(plan_path).resolve()),
                           "--root", str(root), "--gpu", str(gpu)], check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", choices=(0, 1), type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.plan, args.root, args.gpu))
