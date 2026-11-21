"""Prepare nine native replay targets using the unchanged, tested numerical path.

Only the old schema reader is replaced: it requires nested 1/3/6/12 prefixes.
The new validator explicitly certifies nine sources in three disjoint triplets.
No LoRA training routine is called. The legacy preparation routine itself is
reused verbatim, including fixed16 warmup/replay and singleton teacher actions.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SCALING, load, read, save, sha, verify
from adaptation.sources import qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    _, _, spec_path, spec, cache = qualified(args.goal)
    from weightsmith import config as C
    if os.environ.get('CUDA_VISIBLE_DEVICES', str(args.gpu)) != str(args.gpu):
        raise ValueError('The sole CUDA device must match --gpu')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='1', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                      XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION)
    legacy = load('fewpair_native_prepare_unchanged', SCALING/'lora/fresh_lora.py')

    def validated_nine(path, arm):
        if Path(path).resolve() != spec_path or arm != 'matched_decision':
            raise ValueError('Only the qualified nine-source matched-decision dataset is permitted')
        verify(cache['hashes'])
        return spec, spec['items'], cache, 9

    legacy.read_spec = validated_nine
    from weightsmith.locks import GpuLock as SharedGpuLock
    with SharedGpuLock(args.gpu):
        legacy.prepare(SimpleNamespace(spec=str(spec_path), arm='matched_decision', output=str(args.output)))
    verify(cache['hashes'])
    meta_path = args.output/'dataset.json'
    meta = read(meta_path)
    meta['adaptation_driver_artifacts'] = {str(p): sha(p) for p in (Path(__file__), HERE/'sources.py')}
    meta['source_information'] = 'Nine original-request failure states from nine fresh qualified episode pairs; no human action labels.'
    meta['schema_adapter'] = 'Explicit nine-source validator; unchanged historical matched-decision native replay/teacher preparation.'
    save(meta_path, meta)


if __name__ == '__main__':
    main()
