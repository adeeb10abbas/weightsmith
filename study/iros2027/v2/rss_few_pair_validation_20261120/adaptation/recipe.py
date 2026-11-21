"""CPU-only historical 72-row Weightsmith recipe, with arbitrary source triplets."""
from __future__ import annotations

import numpy as np
from weightsmith import distill

ARRAYS = ('vlm.mlp_hidden', 'vlm.resid_out', 'vlm.img_mlp_hidden', 'vlm.img_resid_out')
LAYERS = tuple(range(17))


def fixed_window_slots(original, teacher):
    """Three opening tokens and exactly 37 State-anchored positions.

    The archived construction used a fixed window, not a variable valid-token
    suffix. Depending on tokenized state length, its end can contain padding or
    truncate the tail. Paired token IDs AND masks must still match everywhere.
    """
    a, b = original['prompt_token_ids_at_slots'], teacher['prompt_token_ids_at_slots']
    av, bv = original['prompt_token_valid_at_slots'], teacher['prompt_token_valid_at_slots']
    if len(a) != len(av) or len(b) != len(bv):
        raise ValueError('Token IDs and validity masks have different lengths')
    valid = distill.pi05_slot_map(a, b, av, bv)
    if valid[:3] != [(0, 0), (1, 1), (2, 2)] or len(valid) <= 3:
        raise ValueError('Missing the three-token pi0.5 template opening')
    left, right = valid[3]
    if a[left] != distill.PI05_STATE_TOKEN or b[right] != distill.PI05_STATE_TOKEN:
        raise ValueError('The fixed window must begin at the common State anchor')
    pairs = [(0, 0), (1, 1), (2, 2)] + [(left+i, right+i) for i in range(37)]
    for x, y in pairs:
        if x >= len(a) or y >= len(b):
            raise ValueError('Capture does not cover the complete 40-position fixed window')
        if a[x] != b[y] or bool(av[x]) != bool(bv[y]):
            raise ValueError(f'Paired token ID or validity mismatch at {x}->{y}')
    return pairs


def capture_alignment(original_index, teacher_index, *, expected_count):
    if original_index.get('vlm_layers') != list(LAYERS) or teacher_index.get('vlm_layers') != list(LAYERS):
        raise ValueError('Captures must cover VLM layers 0 through 16 in order')
    for index in (original_index, teacher_index):
        positions = index.get('prompt_token_slots', [])
        if not positions or positions != list(range(len(positions))):
            raise ValueError('Prompt capture positions must be a complete zero-based prefix')
        if len(index['items']) != expected_count:
            raise ValueError('Wrong source count in paired capture')
    slots, records = None, []
    for i, (a, b) in enumerate(zip(original_index['items'], teacher_index['items'], strict=True)):
        for field in ('episode', 'k', 'seed', 'sim_step'):
            if a[field] != b[field]:
                raise ValueError('Capture pairing identity differs: '+field)
        current = fixed_window_slots(a, b)
        if slots is not None and current != slots:
            raise ValueError('Template offsets differ across source states')
        slots = current
        records.append({'index': i, 'valid_positions': sum(bool(a['prompt_token_valid_at_slots'][x]) for x, _ in current),
                        'padding_positions': sum(not bool(a['prompt_token_valid_at_slots'][x]) for x, _ in current)})
    return slots, records


def pooled_key_values(original, teacher, indices, slots):
    n = original[ARRAYS[0]].shape[0]
    if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or not 0 <= i < n for i in indices):
        raise ValueError('Require exactly three distinct available source indices')
    if len(slots) != 40 or len(set(map(tuple, slots))) != 40:
        raise ValueError('Require the 40 distinct historical prompt positions')
    for arrays in (original, teacher):
        for key in ARRAYS:
            if arrays[key].shape[:2] != (n, len(LAYERS)) or not np.isfinite(arrays[key]).all():
                raise ValueError('Wrong source/layer count or nonfinite activation capture')
    return distill.build_keys_values(
        {key: original[key][indices] for key in ARRAYS},
        {key: teacher[key][indices] for key in ARRAYS}, layers=LAYERS,
        mode='both', alpha=1., slot_map=slots, cells=distill.IMAGE_CELLS,
        capture_layers=LAYERS)


def fit(keys, values):
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != 72 or values.shape[0] != 72:
        raise ValueError('Expected the historical 72-key system')
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite ridge inputs')
    ridge = 1e-3 * np.sum(keys*keys) / len(keys)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Degenerate key covariance')
    delta = keys.T @ np.linalg.solve(keys @ keys.T + ridge*np.eye(len(keys)), values)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite weight update')
    return delta, {'keys': 72, 'rank_upper_bound': 72, 'ridge_relative': .001,
                   'ridge_absolute': float(ridge), 'merge_multiplier': 1.,
                   'fit_relative_error': float(np.linalg.norm(keys@delta-values)/max(np.linalg.norm(values), 1e-30))}


def merge(weight, delta):
    if weight.shape != delta.shape or not np.isfinite(weight).all():
        raise ValueError('Invalid base weight or update shape')
    return distill.bf16_round(weight.astype(np.float64)+delta).astype(np.float32)
