"""CPU-only construction, integrity checks, and token/cell alignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'weightsmith-few-pair-baseline-v1'
METHODS = ('embedding_mean', 'residual_cells')
WIDTH = 2048
LAYERS = 18


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def instruction_mask(tokens, valid, prefix_ids, delimiter_ids):
    """Locate only the instruction in the native Task: ..., State: ... format."""
    tokens = np.asarray(tokens)
    valid = np.asarray(valid, dtype=bool)
    if tokens.ndim != 1 or valid.shape != tokens.shape:
        raise ValueError('Expected one token sequence and its exact valid mask')
    prefix_ids, delimiter_ids = list(prefix_ids), list(delimiter_ids)
    if not prefix_ids or not delimiter_ids or list(tokens[:len(prefix_ids)]) != prefix_ids:
        raise ValueError('Unsupported native task prefix')
    starts = [i for i in range(len(prefix_ids), len(tokens) - len(delimiter_ids) + 1)
              if valid[i:i+len(delimiter_ids)].all()
              and list(tokens[i:i+len(delimiter_ids)]) == delimiter_ids]
    if len(starts) != 1 or starts[0] <= len(prefix_ids):
        raise ValueError('Expected one unambiguous state delimiter after a nonempty instruction')
    mask = np.zeros(len(tokens), dtype=bool)
    mask[len(prefix_ids):starts[0]] = True
    if not valid[mask].all():
        raise ValueError('Instruction contains padded token positions')
    return mask


def expand_image_cells(direction):
    """[camera,4x4 cells,width] to the first 512 native image-token positions."""
    a = np.asarray(direction)
    if a.ndim != 3 or a.shape[:2] != (2, 16):
        raise ValueError('Expected two cameras and sixteen row-major cells')
    return a.reshape(2, 4, 4, a.shape[-1]).repeat(4, axis=1).repeat(4, axis=2).reshape(512, a.shape[-1])


def residual_direction(original, clarified, indices, layer_index):
    a, b = np.asarray(original), np.asarray(clarified)
    indices = list(indices)
    if a.shape != b.shape or a.ndim != 5 or a.shape[2:] != (2, 17, WIDTH):
        raise ValueError('Expected matched [sources,layers,2,17,2048] residual captures')
    if len(indices) != 3 or len(set(indices)) != 3 or any(i < 0 or i >= len(a) for i in indices):
        raise ValueError('Exactly three distinct source observations are required')
    if not 0 <= layer_index < a.shape[1]:
        raise ValueError('Requested residual layer was not captured')
    # Cell zero is the whole-camera mean; cells1..16 are the 4x4 source support.
    gap = b[indices, layer_index, :, 1:, :].astype(np.float64) - a[indices, layer_index, :, 1:, :].astype(np.float64)
    if not np.isfinite(gap).all():
        raise ValueError('Nonfinite paired residual targets')
    return gap.mean(axis=0).astype(np.float32)


def write_artifact(output, method, direction, construction, *, strength=1., layer=None, tokenization=None):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError('Refusing to overwrite a baseline construction: ' + str(root))
    if method not in METHODS or not np.isfinite(strength) or strength < 0:
        raise ValueError('Invalid baseline method or nonnegative strength')
    direction = np.asarray(direction, dtype=np.float32)
    expected = (WIDTH,) if method == 'embedding_mean' else (2, 16, WIDTH)
    if direction.shape != expected or not np.isfinite(direction).all():
        raise ValueError('Invalid finite direction shape')
    if method == 'residual_cells' and not 0 <= int(layer) < LAYERS - 1:
        raise ValueError('Residual intervention must precede the final prefix block')
    if method == 'embedding_mean' and not tokenization:
        raise ValueError('Embedding baseline requires native instruction-span metadata')
    root.mkdir(parents=True)
    np.save(root/'direction.npy', direction, allow_pickle=False)
    spec = {'schema': SCHEMA, 'method': method, 'strength': float(strength),
            'direction_shape': list(direction.shape), 'direction_sha256': sha(root/'direction.npy'),
            'direction_l2': float(np.linalg.norm(direction.astype(np.float64))),
            'layer': int(layer) if layer is not None else None,
            'tokenization': tokenization, 'construction': construction,
            'checkpoint_weights_modified': False, 'prompt_rewrite_at_execution': False,
            'runtime_code_sha256': sha(Path(__file__).with_name('native.py')),
            'artifact_code_sha256': sha(__file__)}
    save(root/'baseline.json', spec)
    return spec


def load_artifact(path, *, verify_sources=False):
    root = Path(path).resolve()
    spec = read(root/'baseline.json')
    if spec.get('schema') != SCHEMA or spec.get('method') not in METHODS:
        raise ValueError('Not a registered few-pair baseline artifact')
    if sha(root/'direction.npy') != spec['direction_sha256']:
        raise ValueError('Baseline direction hash changed')
    if spec.get('runtime_code_sha256') != sha(Path(__file__).with_name('native.py')):
        raise ValueError('Baseline runtime changed after construction')
    direction = np.load(root/'direction.npy', allow_pickle=False)
    expected = (WIDTH,) if spec['method'] == 'embedding_mean' else (2, 16, WIDTH)
    if direction.dtype != np.float32 or direction.shape != expected or list(direction.shape) != spec['direction_shape']:
        raise ValueError('Baseline direction dtype or dimensions changed')
    if not np.isfinite(direction).all() or not np.isfinite(spec['strength']) or spec['strength'] < 0:
        raise ValueError('Nonfinite/negative baseline intervention')
    if spec['method'] == 'residual_cells' and not 0 <= spec['layer'] < LAYERS - 1:
        raise ValueError('Invalid residual layer')
    if verify_sources:
        for source, digest in spec['construction'].get('artifacts', {}).items():
            if sha(source) != digest:
                raise ValueError('Baseline source changed: ' + source)
    return spec, direction
