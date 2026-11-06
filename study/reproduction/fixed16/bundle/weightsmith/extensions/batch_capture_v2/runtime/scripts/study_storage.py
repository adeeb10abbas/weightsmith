#!/usr/bin/env python3
"""Operator-only, checksum-verified RAM staging for this study's checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory, manifest):
    expected = {}
    for line in Path(manifest).read_text().splitlines():
        checksum, name = line.split(None, 1)
        rel = Path(name.removeprefix('./'))
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe manifest path')
        expected[str(rel)] = checksum
    files = {str(p.relative_to(directory)): p for p in Path(directory).rglob('*') if p.is_file()}
    if set(files) != set(expected):
        raise ValueError('Checkpoint file inventory differs from pinned manifest')
    for name, p in files.items():
        if p.is_symlink() or digest(p) != expected[name]:
            raise ValueError(f'Checkpoint verification failed: {name}')
    return {'files': len(files), 'bytes': sum(p.stat().st_size for p in files.values()),
            'manifest_sha256': digest(manifest)}


def save(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def memory_headroom():
    cg = Path('/sys/fs/cgroup')
    maximum = (cg / 'memory.max').read_text().strip()
    if maximum == 'max':
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(fields['MemAvailable'].split()[0]) * 1024
    return int(maximum) - int((cg / 'memory.current').read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=['stage', 'restore'])
    ap.add_argument('--root', required=True, type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    source = root / 'runtime/cache/openpi/openpi-assets-simeval/pi05_droid_jointpos'
    manifest = root / 'third_party/pi05_droid_jointpos.sha256'
    state = root / 'runtime/study-checkpoint-staging.json'
    if args.operation == 'stage':
        if state.exists() or source.is_symlink() or not source.is_dir():
            raise ValueError('Refusing existing staging state or non-original checkpoint directory')
        checked = verify(source, manifest)
        required = checked['bytes']
        if memory_headroom() < required + 24 * 1024**3:
            raise RuntimeError('Insufficient cgroup headroom for staging plus 24 GiB reserve')
        if shutil.disk_usage('/dev/shm').free < required + 16 * 1024**3:
            raise RuntimeError('Insufficient shared-memory reserve')
        token = uuid.uuid4().hex
        owner = Path('/dev/shm') / f'{root.name}-study-{token[:8]}'
        owner.mkdir(mode=0o700)
        (owner / 'owner.json').write_text(json.dumps({'root': str(root), 'token': token}))
        ram = owner / 'checkpoint'
        backup = source.with_name(source.name + '.stage-' + token[:8])
        record = {'root': str(root), 'source': str(source), 'ram': str(ram), 'token': token,
                  'backup': str(backup), 'verification': checked, 'started_utc_epoch': time.time(),
                  'phase': 'copying_to_ram'}
        save(state, record)
        shutil.copytree(source, ram)
        assert verify(ram, manifest) == checked
        record['phase'] = 'ram_verified'; save(state, record)
        source.rename(backup)
        try:
            source.symlink_to(ram, target_is_directory=True)
            assert verify(source, manifest) == checked
        except BaseException:
            if source.is_symlink(): source.unlink()
            backup.rename(source)
            raise
        record['phase'] = 'source_redirected'; save(state, record)
        # This is the verified redundant copy in this deployment, never shared storage.
        shutil.rmtree(backup)
        record.update(phase='staged', free_disk_bytes=shutil.disk_usage(root).free,
                      memory_headroom_bytes=memory_headroom())
        save(state, record)
    else:
        record = json.loads(state.read_text())
        ram = Path(record['ram'])
        owner = json.loads((ram.parent / 'owner.json').read_text())
        if owner != {'root': str(root), 'token': record['token']}:
            raise ValueError('Staging ownership mismatch')
        if record['phase'] != 'staged' or not source.is_symlink() or source.resolve() != ram:
            raise ValueError('Staging state requires operator inspection')
        checked = verify(ram, manifest)
        if checked != record['verification']:
            raise ValueError('Staged checkpoint changed')
        if shutil.disk_usage(root).free < checked['bytes'] + 3 * 1024**3:
            raise RuntimeError('Archive owned experiment output before restoring; retain 3 GiB reserve')
        destination = source.with_name(source.name + '.restore-' + record['token'][:8])
        shutil.copytree(ram, destination)
        assert verify(destination, manifest) == checked
        source.unlink()
        destination.rename(source)
        assert verify(source, manifest) == checked
        shutil.rmtree(ram)
        # Keep old recorded RAM paths resolvable without retaining their payload.
        ram.symlink_to(source, target_is_directory=True)
        record.update(phase='restored', restored_utc_epoch=time.time(),
                      free_disk_bytes=shutil.disk_usage(root).free)
        save(state, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
