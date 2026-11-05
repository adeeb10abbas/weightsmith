#!/usr/bin/env python3
"""Verify the separately sealed v2 source before a typed operation or GPU phase."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

RUNTIME = Path(__file__).resolve().parents[1]
ISOLATION = RUNTIME.parents[2]
HASH = re.compile(r'^[0-9a-f]{64}$')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def regular_beneath(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Invalid sealed source path')
    path = root.joinpath(*rel.parts)
    if not path.is_file() or root not in path.resolve().parents:
        raise ValueError('Sealed source must be a regular file beneath the isolated deployment')
    for node in (path, *path.parents):
        if node == root: break
        if node.is_symlink(): raise ValueError('Symlink in sealed source path')
    return path


def verify_binding(binding_path, *, verify_checkpoint=False, isolation=None):
    root = Path(isolation or ISOLATION).resolve()
    binding_path = Path(binding_path)
    if binding_path.is_symlink() or root not in binding_path.resolve().parents:
        raise ValueError('Runtime binding must be an owned regular artifact')
    binding = json.loads(binding_path.read_text())
    if (binding.get('capture_protocol_id') != 'wsp-batch-capture-v2-fixed16'
            or binding.get('runtime_variant') != 'fixed16_new_runtime'):
        raise ValueError('Expected the separately bound fixed16 runtime')
    manifest_path = regular_beneath(root, 'operator/rq123-v2/source_manifest.json')
    source_hash = digest(manifest_path)
    if source_hash != binding.get('source_manifest_sha256'):
        raise ValueError('Full v2 source manifest differs from native runtime binding')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('path_base') != 'original_isolated_deployment' or not manifest.get('files'):
        raise ValueError('Source manifest lacks an explicit nonempty deployment mapping')
    for name, wanted in manifest['files'].items():
        if (not name.startswith(('extensions/batch_capture_v2/', 'operator/rq123-v2/design/'))
                or not HASH.fullmatch(str(wanted)) or digest(regular_beneath(root, name)) != wanted):
            raise ValueError('Sealed v2 source changed: ' + str(name))
    runtime = root/'extensions/batch_capture_v2/runtime'
    actual_python = {p.relative_to(root).as_posix() for p in runtime.rglob('*.py')}
    declared_python = {name for name in manifest['files']
                       if name.startswith('extensions/batch_capture_v2/runtime/') and name.endswith('.py')}
    if actual_python != declared_python:
        raise ValueError('Runtime Python file inventory differs from the sealed source mapping')
    server = 'extensions/batch_capture_v2/runtime/weightsmith/batched_server.py'
    if manifest['files'].get(server) != binding.get('native_server_sha256'):
        raise ValueError('Native server identity differs from complete source manifest')
    if digest(regular_beneath(root, 'wsp.env')) != binding.get('environment_file_sha256'):
        raise ValueError('Private runtime environment configuration changed')
    workspace = Path(binding.get('base_workspace_path', ''))
    if not workspace.is_absolute() or workspace.is_symlink() or workspace.resolve().parent != root/'workspaces':
        raise ValueError('Original editable export must be bound inside the owned workspace directory')
    workspace = workspace.resolve()
    workspace_manifest = regular_beneath(root, (workspace/'workspace.json').relative_to(root).as_posix())
    if digest(workspace_manifest) != binding.get('editable_export_manifest_sha256'):
        raise ValueError('Original editable export manifest changed')
    checkpoint_manifest = regular_beneath(root, 'third_party/pi05_droid_jointpos.sha256')
    checkpoint_manifest_hash = digest(checkpoint_manifest)
    if checkpoint_manifest_hash != binding.get('checkpoint_manifest_sha256'):
        raise ValueError('Checkpoint checksum manifest changed')
    checked_files = 0
    if verify_checkpoint:
        checkpoint = Path(binding['checkpoint_path']).resolve()
        # The explicit verified RAM staging location may be outside the source
        # checkout; its contents must still match all 26 pinned checkpoint files.
        expected = {}
        for line in checkpoint_manifest.read_text().splitlines():
            if not line.strip(): continue
            checksum, name = line.split(None, 1)
            name = name.lstrip('*').removeprefix('./')
            if name in expected or not HASH.fullmatch(checksum): raise ValueError('Invalid checkpoint inventory')
            expected[name] = checksum
        if len(expected) != 26: raise ValueError('All 26 checkpoint files are required')
        actual = {p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob('*') if p.is_file() or p.is_symlink()}
        if actual != set(expected): raise ValueError('Actual checkpoint file set differs')
        for name, wanted in expected.items():
            if digest(regular_beneath(checkpoint, name)) != wanted:
                raise ValueError('Checkpoint bytes changed: ' + name)
        checked_files = len(expected)
    return {'source_manifest_sha256': source_hash, 'verified_source_files': len(manifest['files']),
            'runtime_binding_sha256': digest(binding_path), 'checkpoint_manifest_sha256': checkpoint_manifest_hash,
            'verified_checkpoint_files': checked_files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--tool', choices=('owsp', 'tensor'), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    project = args.project.resolve()
    if (args.project.is_symlink() or project.parent != ISOLATION/'projects'
            or project.name not in ('rq123-v2-adaptive-r1', 'rq123-v2-readiness')
            or os.environ.get('WSP_MANAGED_PROCESS_GROUP') != '1'
            or Path(os.environ.get('WSP_ROOT', '')).resolve() != project):
        raise ValueError('Typed v2 dispatch requires its declared isolated project')
    verify_binding(project/'v2_runtime_binding.json')
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.tool == 'tensor':
        if arguments: raise ValueError('Tensor operations use structured stdin only')
        command = RUNTIME/'drivers/study_tensor_ops.py'
    else:
        if not arguments or arguments[0] not in {'status','episodes','show','tensors','diff','new','eval','acts'}:
            raise ValueError('Unsupported typed v2 command')
        command = RUNTIME/'weightsmith/owsp.py'
    os.execv(sys.executable, [sys.executable, str(command), *arguments])


if __name__ == '__main__': main()
