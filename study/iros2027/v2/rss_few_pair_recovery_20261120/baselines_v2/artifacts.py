"""CPU-only construction, integrity checks, and token/cell alignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'weightsmith-few-pair-baseline-v2'
LEGACY_SCHEMA = 'weightsmith-few-pair-baseline-v1'
LEGACY_RUNTIME_SHA256 = '4f2e073062fc6f725dc8a352026e322fd590d44d91c3fd8d10886db1e7d65da5'
LEGACY_ARTIFACT_SHA256 = 'c00300fddadd5878f643687ae9f5b0bbfd37a628254240fbd84946b27dd1f56c'
REPO = Path(__file__).resolve().parents[5]
LEGACY_ROOT = REPO/'projects/rss-few-pair-validation-20261120/temporal/adaptation/baselines'
COMPATIBILITY_GATE = REPO/'projects/rss-few-pair-recovery-20261120/baseline-v2-qualification/passed.json'
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


def instruction_mask(tokens, valid, prefix_ids, delimiter_ids, separator_token_ids=None):
    """Exclude the complete separator token, including a fused period/comma.

    Native SentencePiece emits either ',' or '.,' immediately before State.
    The State/colon anchor is stable. The complete preceding separator token
    is excluded, including its period when fused; no state or padding is edited.
    """
    tokens = np.asarray(tokens)
    valid = np.asarray(valid, dtype=bool)
    if tokens.ndim != 1 or valid.shape != tokens.shape:
        raise ValueError('Expected one token sequence and its exact valid mask')
    prefix_ids, delimiter_ids = list(prefix_ids), list(delimiter_ids)
    if not prefix_ids or len(delimiter_ids)!=3 or list(tokens[:len(prefix_ids)]) != prefix_ids:
        raise ValueError('Unsupported native task prefix')
    if not valid[:len(prefix_ids)].all() or np.any(np.diff(valid.astype(int))>0):
        raise ValueError('Valid tokens must form a prefix without holes')
    separators=list(separator_token_ids or [delimiter_ids[0]])
    anchors=[i for i in range(len(tokens)-1)
             if valid[i:i+2].all() and list(tokens[i:i+2])==delimiter_ids[1:]]
    if len(anchors)!=1 or anchors[0]-1<=len(prefix_ids):
        raise ValueError('Expected one unambiguous state delimiter after a nonempty instruction')
    end=anchors[0]-1
    if not valid[end] or int(tokens[end]) not in separators:
        raise ValueError('Unsupported punctuation at the instruction/state boundary')
    mask = np.zeros(len(tokens), dtype=bool)
    mask[len(prefix_ids):end] = True
    if not valid[mask].all():
        raise ValueError('Instruction contains padded token positions')
    return mask


def upgrade_tokenization(descriptor):
    """Extend the pinned native tokenizer descriptor without changing directions."""
    if (descriptor.get('model_sha256')!='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
            or descriptor.get('prefix_ids')!=[2,7071,235292]
            or descriptor.get('delimiter_ids')!=[235269,3040,235292]
            or descriptor.get('max_token_len')!=200):
        raise ValueError('Unsupported legacy tokenizer descriptor')
    return {**descriptor,'separator_token_ids':[235269,1173],
            'boundary_rule':'Unique State/colon anchor; exclude its preceding comma or fused-period/comma token.'}


def certify_legacy(root,spec):
    """Permit only the unchanged temporal artifacts proven equivalent by the gate."""
    if (not root.is_relative_to(LEGACY_ROOT)
            or spec.get('runtime_code_sha256')!=LEGACY_RUNTIME_SHA256
            or spec.get('artifact_code_sha256')!=LEGACY_ARTIFACT_SHA256):
        raise ValueError('Uncertified legacy artifact or runtime')
    gate=read(COMPATIBILITY_GATE)
    if (gate.get('status')!='pass' or gate.get('kernel_batch')!=16
            or gate.get('new_policy_episodes')!=0
            or gate.get('legacy_exact_parity') is not True):
        raise ValueError('Legacy temporal compatibility gate has not passed')
    for path in (Path(__file__),Path(__file__).with_name('native.py'),Path(__file__).with_name('server.py')):
        if gate['artifacts'].get(str(path))!=sha(path):
            raise ValueError('V2 compatibility runtime changed')
    for path in (root/'baseline.json',root/'direction.npy'):
        if gate['legacy_artifacts'].get(str(path))!=sha(path):
            raise ValueError('Legacy artifact was not certified, or changed')
    result=dict(spec)
    if result['method']=='embedding_mean':result['tokenization']=upgrade_tokenization(result['tokenization'])
    return result


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
    if spec.get('schema') not in (SCHEMA,LEGACY_SCHEMA) or spec.get('method') not in METHODS:
        raise ValueError('Not a registered few-pair baseline artifact')
    legacy=spec['schema']==LEGACY_SCHEMA
    if legacy:spec=certify_legacy(root,spec)
    if sha(root/'direction.npy') != spec['direction_sha256']:
        raise ValueError('Baseline direction hash changed')
    if not legacy and spec.get('runtime_code_sha256') != sha(Path(__file__).with_name('native.py')):
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
