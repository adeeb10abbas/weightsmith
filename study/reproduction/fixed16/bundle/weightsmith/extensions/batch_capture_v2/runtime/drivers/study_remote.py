#!/usr/bin/env python3
"""Fixed remote entry point for study tools, private caches and bounded workers."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ISOLATION_ROOT = REPO.parents[2]
sys.path.insert(0, str(ISOLATION_ROOT / 'scripts'))
from run_isolated import run_guarded

READ_COMMANDS = {'status', 'episodes', 'show', 'tensors', 'diff', 'new'}
GPU_COMMANDS = {'eval', 'acts'}


def command_for(root, tool, arguments, seconds):
    if not root.is_absolute() or root.is_symlink() or ISOLATION_ROOT / 'projects' not in root.resolve().parents:
        raise ValueError('Study project must be an owned, nonsymlinked project under the deployment')
    if root.name not in ('rq123-v2-adaptive-r1', 'rq123-v2-readiness'):
        raise ValueError('Only the declared v2 editor projects are exposed')
    if not (root / 'config.json').is_file():
        raise ValueError('Study project config is missing')
    if not 0 < seconds <= 7200:
        raise ValueError('Remaining time must be in (0,7200] seconds')
    if tool == 'tensor':
        if arguments:
            raise ValueError('Tensor requests use JSON stdin only')
        return [sys.executable, str(REPO / 'scripts/verify_runtime.py'), '--project', str(root),
                '--tool', tool, '--'], False
    if not arguments or arguments[0] not in READ_COMMANDS | GPU_COMMANDS:
        raise ValueError('Only bounded study commands are available; no Python or held-out tools')
    return [sys.executable, str(REPO / 'scripts/verify_runtime.py'), '--project', str(root),
            '--tool', tool, '--', *arguments], arguments[0] in GPU_COMMANDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--tool', choices=['owsp', 'tensor'], required=True)
    ap.add_argument('--seconds', type=float, required=True)
    ap.add_argument('arguments', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    command, gpu = command_for(a.root, a.tool, args, a.seconds)
    deadline = time.monotonic() + a.seconds
    os.environ['WSP_ROOT'] = str(a.root.resolve())
    lock = None
    try:
        if gpu:
            lockdir = ISOLATION_ROOT / 'runtime/locks'
            lockdir.mkdir(parents=True, exist_ok=True)
            lock = (lockdir / 'study-dispatch.lock').open('a')
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Study GPU queue exhausted the remaining wall-time budget')
                    time.sleep(.25)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('No wall-time budget remains')
        # --foreground keeps all descendants in run_guarded's owned process group.
        # Cleanup kills only that group, including orphaned policy/simulator workers.
        bounded = ['/usr/bin/timeout', '--foreground', '--signal=TERM', '--kill-after=5',
                   f'{remaining:.3f}s', *command]
        return run_guarded(bounded, ISOLATION_ROOT, min_free_gib=8, gpu=[0, 1] if gpu else None)
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        raise SystemExit(str(exc))
