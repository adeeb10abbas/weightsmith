#!/usr/bin/env python3
"""Local packaging only; no deployment, model call, input data, or GPU action."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from verify_package import verify, verify_archive


def build(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if root in archive.parents: raise ValueError('Archive must stay outside its own package')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('No symlink may be packaged')
        if not path.is_file() or path.name == 'MANIFEST.json': continue
        name = path.relative_to(root).as_posix()
        if (set(path.parts) & {'__pycache__', '.git', '.codex', '.aws', 'data', 'agents'}
                or path.suffix in {'.npz', '.npy', '.pyc', '.jsonl', '.key', '.pem', '.mp4'}
                or path.name in {'wsp.env','auth.json','broker_state.json','driver_result.json','NOTEBOOK.md','FINAL.md'}):
            raise ValueError('Runtime data, auth, caches, and editor traces are excluded: ' + name)
        data = path.read_bytes(); files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (root/'MANIFEST.json').write_text(json.dumps({'package_id':'weightsmith-fixed16-teammate-v1',
        'status':'source_and_CPU_packaging_only; native_GPU_and_live_provider_readiness_pending',
        'sealed_source_files':77, 'files':files},indent=2)+'\n')
    checked = verify(root)
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for name in sorted([*files, 'MANIFEST.json']):
                data = (root/name).read_bytes(); item = tarfile.TarInfo(name)
                item.size = len(data); item.mode = 0o644; item.mtime = 0
                tar.addfile(item,io.BytesIO(data))
    return {**checked, **verify_archive(archive)}


if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(build(root,root.parent/(root.name+'.tar.gz')),indent=2))
