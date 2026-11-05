#!/usr/bin/env python3
"""Bind an already installed teammate deployment; CPU hashing only, no setup/GPU.

Preserves the original 77-file source seal and writes a NEW native binding for
this deployment. It does not certify GPU parity or authorize editor search.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    if not args.root.is_absolute() or args.root.is_symlink(): raise ValueError('Owned absolute installation root required')
    if args.workspace.resolve().parent != root/'workspaces': raise ValueError('Export must be beneath owned root/workspaces')
    package=Path(__file__).resolve().parents[1]
    reference=json.loads((package/'reference/v2_runtime_binding.json').read_text())
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    output=root/'operator/rq123-v2/v2_runtime_binding.json'
    if output.exists() or output.is_symlink(): raise ValueError('Native binding is immutable; use a fresh deployment for another binding')
    reference.update(bound_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_path=str(args.checkpoint.resolve(strict=True)),
        base_workspace_path=str(args.workspace.resolve(strict=True)),
        editable_export_manifest_sha256=digest(args.workspace/'workspace.json'),
        environment_file_sha256=digest(root/'wsp.env'),
        handoff_package_id='weightsmith-fixed16-teammate-v1',
        reference_native_runtime_binding_sha256=digest(package/'reference/v2_runtime_binding.json'))
    # Reference provenance is not an assertion that this host ran the original study.
    reference.pop('original_runtime_binding_sha256',None)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(reference,stream,indent=2);stream.write('\n')
    output.chmod(0o444)
    module_path=root/'extensions/batch_capture_v2/runtime/scripts/verify_runtime.py'
    spec=importlib.util.spec_from_file_location('teammate_runtime_verifier',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        result=module.verify_binding(output,verify_checkpoint=True,isolation=root)
    except BaseException:
        print('Binding attempt retained; verification failed. Do not launch GPU or editor.',file=sys.stderr)
        raise
    print(json.dumps({'CPU_binding_verification':result,'GPU_parity_verified':False,'binding_path':str(output)},indent=2))


if __name__=='__main__':main()
