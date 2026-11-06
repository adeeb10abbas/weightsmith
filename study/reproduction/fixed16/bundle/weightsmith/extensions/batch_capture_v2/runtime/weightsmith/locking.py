"""Process locks shared by operator and editor entry points (stdlib only)."""
from __future__ import annotations

import fcntl
from weightsmith import config as C


class ResourceLock:
    def __init__(self, *names):
        self.names = sorted(set(names))
        self.files = []

    def __enter__(self):
        C.LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for name in self.names:
                stream = open(C.LOCK_DIR / f"{name}.lock", "a")
                self.files.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for stream in reversed(self.files):
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
        self.files = []


class GpuLock(ResourceLock):
    """Acquire GPUs in a fixed order, globally across all projects."""
    def __init__(self, *gpus):
        super().__init__(*(f"gpu{gpu}" for gpu in gpus))
