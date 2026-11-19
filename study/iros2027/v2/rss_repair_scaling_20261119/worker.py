"""A separately sealed collection entry point using the validated temporal runtime."""
import argparse
import os
import sys
from pathlib import Path
from common import HERE, PRIOR, POOL, ROOT, load, read, validate_collection_plan

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = ap.parse_args(); plan = read(args.plan); validate_collection_plan(plan)
    if not args.root.resolve().is_relative_to(ROOT / 'collection'):
        raise ValueError('Source output must be in this cohort collection directory')
    # Config snapshots ports at import time. Set these before importing pool or
    # any policy configuration, so two workers cannot share the default port.
    os.environ.update(WSP_POOL_PORT=str(8700+10*args.gpu),WSP_SERVER_GPU=str(args.gpu),
                      WSP_SIM_GPU=str(args.gpu),WSP_EVAL_EXTRA_GPUS='',WSP_TASK_DIRS=str(PRIOR/'runtime'))
    prior = load('scaling_temporal_worker', PRIOR / 'worker.py')
    prior.verify_global_freeze(plan)
    import weightsmith
    pool = load('weightsmith.pool', POOL); weightsmith.pool = pool
    prior.main()

if __name__ == '__main__': main()
