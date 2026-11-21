"""CPU gate: reproduce all 17 archived N=3 deployed tensors exactly."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, REPO, read, save, sha, verify
from adaptation.recipe import ARRAYS, LAYERS, capture_alignment, pooled_key_values, fit, merge
from weightsmith import distill

ARCHIVE = REPO/'projects/rss-repair-scaling-20261119/edits'
PINNED = {
    str(ARCHIVE/'n3/construction.json'): '68520cb055690bbd4f42fc6cfb2cbab8b100ec54ddcab0bdfcc234c2d4c93fc2',
    str(ARCHIVE/'captures/original/index.json'): '26fa28eeaf17236a5226c229229a5d2e98fdbada12760fffa56ce05cd15e3e65',
    str(ARCHIVE/'captures/teacher/index.json'): '3f940545a40d61952b7e2924cbd151c28b71bf66a250ce26322d03d7405b6e54',
    str(ARCHIVE/'captures/original/acts.npz'): 'c1c86e151a9c40aa6c43b27044ab3ced313ee4bc168322ea88be1d6912f9d74a',
    str(ARCHIVE/'captures/teacher/acts.npz'): 'c9acbd7638ca8531cdb46f2a1f794909f9874e185d3e86b9709f8ee4f8d400a9',
}


def ensure(path=None):
    path = Path(path or ROOT/'adaptation-compatibility/archived-n3.json').resolve()
    own = {str(p): sha(p) for p in (Path(__file__), HERE/'recipe.py', REPO/'weightsmith/distill.py')}
    if path.exists():
        old = read(path)
        if old.get('status') != 'pass' or old.get('construction_code') != own:
            raise ValueError('Existing historical gate is failed or belongs to different construction code')
        verify(old['artifacts'])
        return path
    start = time.time()
    verify(PINNED)
    old = read(ARCHIVE/'n3/construction.json')
    verify(old['artifacts'])
    arrays, indices = {}, {}
    for which in ('original', 'teacher'):
        indices[which] = read(ARCHIVE/f'captures/{which}/index.json')
        with np.load(ARCHIVE/f'captures/{which}/acts.npz', allow_pickle=False) as z:
            arrays[which] = {key: z[key][:3].copy() for key in ARRAYS}
    all_slots, alignment = capture_alignment(indices['original'], indices['teacher'], expected_count=12)
    expected_slots = list(zip([0, 1, 2]+list(range(13, 50)), [0, 1, 2]+list(range(17, 54)), strict=True))
    if all_slots != expected_slots:
        raise ValueError('Dynamic State-anchor alignment does not reproduce the archived window')
    kv = pooled_key_values(arrays['original'], arrays['teacher'], [0, 1, 2], all_slots)
    workspace = Path(old['workspace'])
    config = read(workspace/'workspace.json')
    layers, artifacts = [], {**PINNED, **old['artifacts'], **own}
    for layer, record in zip(LAYERS, old['layers'], strict=True):
        name = distill.MLP_DOWN.format(layer)
        entry = config['tensors'][name]
        base_path, expected_path = workspace/'base'/entry['file'], workspace/'edits'/entry['file']
        if sha(base_path) != record['base_sha256'] or sha(expected_path) != record['edited_sha256']:
            raise ValueError('Archived deployed tensor or base changed')
        base, expected = np.load(base_path, allow_pickle=False), np.load(expected_path, allow_pickle=False)
        delta, stats = fit(*kv[layer])
        rebuilt = merge(base, delta)
        mismatch = int(np.count_nonzero(rebuilt != expected))
        layers.append({'layer': layer, 'bit_exact': mismatch == 0, 'different_values': mismatch,
                       'max_abs_error': float(np.max(np.abs(rebuilt-expected))),
                       'ridge_absolute': stats['ridge_absolute'], 'expected_sha256': sha(expected_path)})
        artifacts[str(base_path)] = record['base_sha256']
        print({'historical_n3_layer': layer, 'bit_exact': mismatch == 0}, flush=True)
        if mismatch:
            save(path, {'status': 'failed', 'layers': layers, 'construction_code': own, 'artifacts': artifacts})
            raise ValueError(f'Archived N=3 compatibility failed at layer {layer}; no fresh fitting permitted')
    verify(artifacts)
    save(path, {'status': 'pass', 'gate': 'CPU reconstruction of all17 archived N=3 BF16 deployed weights',
                'source_indices': [0, 1, 2], 'selected_decisions': [31, 45, 6],
                'fixed_prompt_positions': 40, 'image_cells': 32, 'key_rows': 72,
                'archived_all12_alignment': alignment, 'layers': layers,
                'construction_code': own, 'artifacts': artifacts, 'wall_seconds': time.time()-start,
                'scope': 'Numerical recipe compatibility only; not a new policy evaluation.'}, exclusive=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(ensure(args.output))
