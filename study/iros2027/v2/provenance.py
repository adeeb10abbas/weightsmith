"""Read-only host provenance for a new experiment; never inspect episode or held-out data.

Run ``python study/iros2027/v2/provenance.py --verify-checkpoint > ledger.json``.
Source digests identify the installed code even when a dependency lacks Git metadata.
This is an inventory, not certification of simulator determinism or benchmark readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".sh", ".patch"}
EXCLUDED_DIRS = {
    "__pycache__", "assets", "output", "data", "logs", "projects", "state", "cands",
    "heldout", "distill-runs", "site", "study", "paper", "checkpoints", "workspace", "workspaces",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args: list[str], *, cwd: Path | None = None) -> dict:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def source_inventory(root: Path) -> dict:
    """Hash code, excluding data/results and symlinks; no claim about asset integrity."""
    root = root.resolve()
    files = {}
    if root.is_dir():
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in EXCLUDED_DIRS)
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix in SOURCE_SUFFIXES and not path.is_symlink():
                    files[path.relative_to(root).as_posix()] = sha256_file(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    # Do not let Git silently report an ancestor repository for an unpacked dependency.
    git = command(["git", "rev-parse", "HEAD"], cwd=root) if (root / ".git").exists() else None
    return {
        "path": str(root), "exists": root.is_dir(),
        "git_sha": git["stdout"] if git and git["returncode"] == 0 else None,
        "source_sha256": hashlib.sha256(encoded).hexdigest() if files else None,
        "source_files": files, "scope": "code only; excludes assets, data, results, hidden directories and symlinks",
    }


def verify_checkpoint(root: Path, manifest: Path, *, verify: bool) -> dict:
    """Verify each published file; reject manifest paths escaping the checkpoint root."""
    root = root.resolve()
    records = []
    seen = set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*")
        path = (root / relative).resolve()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_relative_to(root) or path == root:
            raise ValueError(f"invalid checkpoint manifest entry: {relative!r}")
        if path in seen:
            raise ValueError(f"duplicate checkpoint manifest entry: {relative!r}")
        seen.add(path)
        exists = path.is_file()
        observed = sha256_file(path) if verify and exists else None
        records.append({"file": relative, "exists": exists, "expected_sha256": expected,
                        "observed_sha256": observed, "matches": observed == expected if verify else None})
    if not records:
        raise ValueError("checkpoint manifest is empty")
    return {
        "path": str(root), "manifest_sha256": sha256_file(manifest), "files": records,
        "verification": "verified" if verify and all(r["matches"] for r in records)
        else "mismatch_or_missing" if verify else "not_requested",
        "identity_definition": "sha256 of the published per-file manifest; not a hash of concatenated model weights",
    }


def environment(python: str) -> dict:
    # Read distribution metadata only. Importing Torch/Isaac/JAX could initialize a GPU.
    code = """
import importlib.metadata as m, json, sys
names = ['jax', 'jaxlib', 'torch', 'isaacsim', 'isaaclab', 'robolab', 'numpy',
         'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12']
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'packages': versions}))
"""
    result = command([python, "-c", code])
    return {"executable": python, **(json.loads(result["stdout"]) if result["returncode"] == 0 else result)}


def collect(*, verify: bool = False) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from weightsmith import config as c

    ws = c.BASE_WS / "workspace.json"
    workspace = json.loads(ws.read_text()) if ws.is_file() else {}
    free = shutil.disk_usage(c.REPO).free
    return {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(), "free_bytes": free,
        "experiment_tools": source_inventory(Path(__file__).resolve().parent),
        "repository": source_inventory(c.REPO), "openpi": source_inventory(c.OPENPI_DIR),
        "robolab": source_inventory(c.ROBOLAB),
        "environments": {"policy": environment(c.OPENPI_PY), "simulator": environment(str(c.SIM_PY))},
        "gpus": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,"
                         "memory.used,utilization.gpu,power.limit,temperature.gpu", "--format=csv"]),
        "checkpoint": verify_checkpoint(c.CHECKPOINT, c.REPO / "third_party/pi05_droid_jointpos.sha256", verify=verify),
        "workspace": {"path": str(c.BASE_WS), "metadata_sha256": sha256_file(ws) if ws.is_file() else None,
                      "declared_base_sha256": workspace.get("base_editable_sha256"),
                      "tensor_count": workspace.get("tensor_count"), "tensor_bytes_reverified": False},
        "limits": ["No simulator run, asset verification, scene-pose check or held-out access is performed.",
                   "Source digests identify observed code, without proving equivalence to a published revision.",
                   "Workspace identity is declared metadata, not fresh verification of exported tensor bytes."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-checkpoint", action="store_true", help="read and hash every checkpoint file (~12 GB)")
    args = parser.parse_args(argv)
    report = collect(verify=args.verify_checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["checkpoint"]["verification"] == "mismatch_or_missing")


if __name__ == "__main__":
    raise SystemExit(main())
