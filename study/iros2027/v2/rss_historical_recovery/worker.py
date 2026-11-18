"""Execute only the sealed 96-episode GPU0 historical recovery diagnostic."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PRIOR = HERE.parent / "rss_repair_comparison"
POOL = HERE.parent / "rss_goal_expansion_v2/runtime/single_attempt_pool.py"
sys.path.insert(0, str(REPO))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0], required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    if (plan["cohort_id"] != protocol["cohort_id"] or plan["split"] != "development"
            or plan["inference_batching"] != "fixed16"
            or [b["condition"] for b in plan["batches"]] != protocol["conditions"]
            or len(plan["batches"]) != 6
            or any(b["n"] != 16 or b["slot_start"] != protocol["slot_start"] for b in plan["batches"])
            or args.root.resolve() != REPO / "projects" / protocol["cohort_id"] / "evaluation"):
        raise ValueError("Plan is outside the frozen recovery diagnostic")
    # Set environment before importing config/pool: config captures its port once.
    os.environ.update(WSP_POOL_PORT="8700", WSP_SERVER_GPU="0", WSP_SIM_GPU="0",
                      WSP_EVAL_EXTRA_GPUS="", WSP_TASK_DIRS=str(PRIOR / "runtime"))
    old = load("rss_recovery_prior_worker", PRIOR / "worker.py")
    old.verify_global_freeze(plan)
    import weightsmith
    pool = load("weightsmith.pool", POOL)
    weightsmith.pool = pool
    old.main()


if __name__ == "__main__":
    main()
