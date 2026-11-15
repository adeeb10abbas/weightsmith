"""Provenance must detect drift without reading sealed data or escaping a manifest root."""
import hashlib

import pytest
from provenance import sha256_file, source_inventory, verify_checkpoint


def test_source_inventory_ignores_results_and_external_links(tmp_path):
    (tmp_path / "runtime.py").write_text("x = 1\n")
    for name in ("heldout", "study", "projects", ".venv"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "secret.py").write_text("must not be inventoried")
    (tmp_path / "link.py").symlink_to(tmp_path / "heldout/secret.py")
    first = source_inventory(tmp_path)
    assert set(first["source_files"]) == {"runtime.py"}
    assert first["git_sha"] is None
    assert source_inventory(tmp_path)["source_sha256"] == first["source_sha256"]
    (tmp_path / "runtime.py").write_text("x = 2\n")
    assert source_inventory(tmp_path)["source_sha256"] != first["source_sha256"]


def test_checkpoint_verification_detects_changed_and_missing_files(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"model")
    manifest = tmp_path / "expected.sha256"
    manifest.write_text(f"{sha256_file(weights)}  ./weights.bin\n")
    assert verify_checkpoint(tmp_path, manifest, verify=False)["verification"] == "not_requested"
    assert verify_checkpoint(tmp_path, manifest, verify=True)["verification"] == "verified"
    weights.write_bytes(b"edited")
    assert verify_checkpoint(tmp_path, manifest, verify=True)["verification"] == "mismatch_or_missing"
    weights.unlink()
    assert verify_checkpoint(tmp_path, manifest, verify=True)["verification"] == "mismatch_or_missing"


@pytest.mark.parametrize("entry", ["../outside", "/etc/passwd", "link.bin"])
def test_checkpoint_manifest_cannot_escape_root(tmp_path, entry):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "link.bin").symlink_to(tmp_path / "outside")
    manifest = tmp_path / "manifest"
    manifest.write_text(f"{'0' * 64}  {entry}\n")
    with pytest.raises(ValueError, match="invalid checkpoint"):
        verify_checkpoint(root, manifest, verify=True)


@pytest.mark.parametrize("text", ["", "bad-hash  weights.bin\n", f"{'0' * 64}  x\n" * 2])
def test_checkpoint_manifest_rejects_incomplete_identity(tmp_path, text):
    manifest = tmp_path / "manifest"
    manifest.write_text(text)
    with pytest.raises(ValueError):
        verify_checkpoint(tmp_path, manifest, verify=True)


def test_source_inventory_does_not_follow_directory_links(tmp_path):
    outside = tmp_path / "heldout"
    outside.mkdir()
    (outside / "private.py").write_text("sealed")
    (tmp_path / "linked-directory").symlink_to(outside, target_is_directory=True)
    assert source_inventory(tmp_path)["source_files"] == {}
    assert sha256_file(outside / "private.py") == hashlib.sha256(b"sealed").hexdigest()
