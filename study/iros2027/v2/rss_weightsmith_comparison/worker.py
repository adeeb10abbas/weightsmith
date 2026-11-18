"""Run only the declared selection/test plan using the frozen single-attempt pool."""
import argparse
import os
from pathlib import Path

from common import POOL, PRIOR, ROOT, load, read, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    plan = read(args.plan); validate_plan(plan)
    expected = ROOT / ("selection" if plan["split"] == "development" else "evaluation")
    if args.root.resolve() != expected:
        raise ValueError("Output path differs from the declared comparison")
    os.environ.update(WSP_POOL_PORT=str(8700 + 10 * args.gpu), WSP_SERVER_GPU=str(args.gpu),
                      WSP_SIM_GPU=str(args.gpu), WSP_EVAL_EXTRA_GPUS="", WSP_TASK_DIRS=str(PRIOR / "runtime"))
    old = load("weightsmith_matched_prior_worker", PRIOR / "worker.py")
    old.verify_global_freeze(plan)
    import weightsmith
    pool = load("weightsmith.pool", POOL)
    weightsmith.pool = pool
    old.main()


if __name__ == "__main__":
    main()
