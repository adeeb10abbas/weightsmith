"""GPU flock helpers shared by owsp (editor) and heldout (operator)."""
from __future__ import annotations

import fcntl

from weightsmith import config as C

LOCKS = C.LOCK_DIR


class GpuLock:
    """Exclusive flock on one or more GPUs (lock files gpu<i>.lock, shared by all projects).
    `optional` GPUs are taken only if free right now (non-blocking); see .acquired."""

    def __init__(self, *gpus, optional=()):
        self.names = sorted({f"gpu{g}" for g in gpus})  # fixed order -> no lock-order deadlock
        self.optional = [g for g in optional if f"gpu{g}" not in self.names]

    def __enter__(self):
        LOCKS.mkdir(parents=True, exist_ok=True)
        self.fs, self.acquired = [], []
        for name in self.names:
            f = open(LOCKS / f"{name}.lock", "w")
            fcntl.flock(f, fcntl.LOCK_EX)
            self.fs.append(f)
        for g in self.optional:
            f = open(LOCKS / f"gpu{g}.lock", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            self.fs.append(f)
            self.acquired.append(g)
        return self

    def __exit__(self, *a):
        for f in reversed(self.fs):
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


