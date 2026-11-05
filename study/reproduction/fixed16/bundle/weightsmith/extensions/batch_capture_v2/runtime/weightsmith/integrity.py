"""Stable artifact identities shared by live evaluation, replay and held-out runs."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _add_file(digest, path: Path, label: str) -> None:
    # Length delimiters keep different names/content partitions unambiguous.
    name = label.encode("utf-8")
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def edits_fingerprint(edits: Path | str) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(edits).glob("*.npy")):
        _add_file(digest, path, path.name)
    return digest.hexdigest()


def workspace_fingerprint(workspace: Path | str | None) -> str:
    """Identify exact edits and the manifest that maps them into the checkpoint.

    The checkpoint's identity remains the validated base hash in workspace.json.
    No path or timestamp enters the digest, so a frozen copy has the same ID.
    """
    if workspace is None:
        return "base"
    workspace = Path(workspace)
    digest = hashlib.sha256()
    _add_file(digest, workspace / "workspace.json", "workspace.json")
    for path in sorted((workspace / "edits").glob("*.npy")):
        _add_file(digest, path, "edits/" + path.name)
    return digest.hexdigest()
