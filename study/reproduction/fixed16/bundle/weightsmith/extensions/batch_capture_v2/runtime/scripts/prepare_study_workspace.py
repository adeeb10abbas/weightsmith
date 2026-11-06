#!/usr/bin/env python3
"""Operator-only, CPU export for the bounded RQ1--RQ3 editing surface.

Uses the configured OpenPI interpreter but exposes no GPU. The export writes
18 float32 arrays containing effective-bf16 values (~288 MiB), not a duplicate
checkpoint or the full ~11 GB editable workspace. Full checkpoint restoration
still requires host RAM. Run under the deployment's normal resource guard.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weightsmith import config as C

EXPORT_BYTES = 18 * 4096 * 1024 * 4 + (1 << 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=C.CHECKPOINT)
    parser.add_argument("--workspace", type=Path, default=C.BASE_WS)
    parser.add_argument("--min-free-gib", type=float, default=8)
    args = parser.parse_args()
    if args.min_free_gib < 8:
        parser.error("the study export retains at least the normal 8 GiB reserve")
    if not (args.checkpoint / "params" / "_METADATA").is_file():
        parser.error("checkpoint params/_METADATA is missing")
    if args.workspace.exists() and any(args.workspace.iterdir()):
        parser.error("refusing to reuse a non-empty workspace")
    existing = args.workspace.resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    needed = EXPORT_BYTES + int(args.min_free_gib * (1 << 30))
    if free < needed:
        parser.error(f"need {needed / (1 << 30):.2f} GiB free including reserve; have {free / (1 << 30):.2f}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    cmd = [C.OPENPI_PY, "-m", "weightsmith.lab.openpi_vlm_full", "prepare",
           "--checkpoint", str(args.checkpoint.resolve()), "--workspace", str(args.workspace.resolve()),
           "--action-down-only", "--max-changed-rows", "128", "--max-changed-layers", "2"]
    raise SystemExit(subprocess.call(cmd, cwd=C.REPO, env=env))


if __name__ == "__main__":
    main()
