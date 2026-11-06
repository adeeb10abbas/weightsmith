#!/usr/bin/env python3
"""Verify the exact package and streamed archive without extraction or imports."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def checked_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('Unsafe package member')
    return name


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    expected = set(manifest['files']) | {'MANIFEST.json'}
    if actual != expected: raise ValueError('Package inventory changed')
    for name, row in manifest['files'].items():
        path = root/checked_name(name)
        if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
            raise ValueError('Symlink in package')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Changed package file: ' + name)
    sealed = json.loads((root/'reference/source_manifest.json').read_text())
    if len(sealed['files']) != 77: raise ValueError('Expected all 77 sealed sources')
    for name, expected_hash in sealed['files'].items():
        if hashlib.sha256((root/'bundle/weightsmith'/checked_name(name)).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Sealed source changed: ' + name)
    return {'verified_files': len(manifest['files']), 'sealed_files': 77,
            'native_gpu_validation_included': False, 'experimental_outcomes_included': False}


def verify_archive(archive):
    seen = {}; manifest = None
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = checked_name(member.name)
            if name in seen or not member.isfile(): raise ValueError('Duplicate or non-regular archive member')
            stream = tar.extractfile(member)
            digest = hashlib.sha256(); size = 0; chunks = [] if name == 'MANIFEST.json' else None
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block); size += len(block)
                if chunks is not None: chunks.append(block)
            if size != member.size: raise ValueError('Truncated archive member')
            seen[name] = {'bytes': size, 'sha256': digest.hexdigest()}
            if chunks is not None: manifest = json.loads(b''.join(chunks))
    if manifest is None or set(seen) != set(manifest['files']) | {'MANIFEST.json'}:
        raise ValueError('Missing or additional archive member')
    if any(seen[name] != expected for name, expected in manifest['files'].items()):
        raise ValueError('Archive checksum mismatch')
    return {'verified_archive_members': len(seen), 'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            'bytes': Path(archive).stat().st_size, 'manifest_sha256': seen['MANIFEST.json']['sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive) if args.archive else verify(Path(__file__).resolve().parents[1]), indent=2))
