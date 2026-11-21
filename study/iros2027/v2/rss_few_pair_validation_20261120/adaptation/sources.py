"""Fail-closed validation of the qualified nine-source manifest before CUDA."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from common import HERE, SCALING, canonical, goal, goal_index, goal_root, load, protocol, read, sha, verify


def validate_groups(groups, pair_ids, grouping_seed):
    if len(pair_ids) != 9 or len(set(pair_ids)) != 9:
        raise ValueError('Require nine distinct qualified source pairs')
    order = np.random.default_rng(grouping_seed).permutation(9).tolist()
    expected = [{'group_id': f'g{i}', 'indices': order[3*i:3*i+3],
                 'pair_ids': [pair_ids[k] for k in order[3*i:3*i+3]]} for i in range(3)]
    if groups != expected:
        raise ValueError('Source groups differ from the frozen deterministic disjoint triplets')
    return groups


def qualified(goal_id):
    p, g, root = protocol(), goal(goal_id), goal_root(goal_id)
    status_path = root/'source-status.json'
    status = read(status_path)
    if status.get('status') != 'qualified' or status.get('qualified') is not True:
        raise ValueError('Goal has not qualified nine fresh source pairs; no GPU preparation is permitted')
    manifest_path = Path(status['manifest']).resolve()
    if not manifest_path.is_relative_to(root/'sources') or sha(manifest_path) != status['manifest_sha256']:
        raise ValueError('Foreign or changed qualified manifest')
    manifest = read(manifest_path)
    expected = {'cohort_id': p['cohort_id'], 'goal_id': goal_id, 'case': g['runtime_case'],
                'split': 'development', 'protocol_sha256': sha(HERE/'protocol.json'),
                'fresh_collection': True, 'qualified': True}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Source cohort, split, protocol or qualification mismatch')
    verify(manifest['artifacts'])
    attempts = manifest['attempts']
    if len(attempts) not in (16, 32, 48) or manifest['attempted_pairs'] != len(attempts):
        raise ValueError('Source attempts do not form bounded complete paired blocks')
    if [r['slot'] for r in attempts] != sorted({r['slot'] for r in attempts}):
        raise ValueError('Source attempts are not unique and ordered by slot')
    eligible = [r for r in attempts if r['eligible'] is True]
    if manifest['eligible_pairs'] != len(eligible) or manifest['pairs'] != eligible[:9] or len(eligible) < 9:
        raise ValueError('Sources are not the first nine eligible pairs')
    if sum(r['eligible'] is True for r in attempts[:-16]) >= 9:
        raise ValueError('Collection continued after the first qualifying complete block')
    pairs = manifest['pairs']
    pair_ids = [r['pair_id'] for r in pairs]
    seed = p['collection']['grouping_seed'] + goal_index(goal_id)
    if manifest['grouping_seed'] != seed:
        raise ValueError('Changed source grouping seed')
    validate_groups(manifest['groups'], pair_ids, seed)
    spec_path = Path(manifest['spec']).resolve()
    if spec_path.parent != manifest_path.parent or spec_path.name != 'correction.json':
        raise ValueError('Correction spec is outside its qualified manifest directory')
    spec = read(spec_path)
    expected_spec = {**{k: expected[k] for k in ('cohort_id', 'goal_id', 'case', 'split', 'protocol_sha256')},
                     'inference_batching': 'fixed16', 'correction_count': 9, 'arm': 'correction',
                     'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
                     'distinct_pair_ids': pair_ids, 'groups': manifest['groups'],
                     'original_prompt': g['original'], 'teacher_prompt': g['teacher']}
    if any(spec.get(k) != v for k, v in expected_spec.items()) or len(spec['items']) != 9:
        raise ValueError('Correction spec identity differs from the qualified manifest')
    if len({r['obs_path'] for r in spec['items']}) != 9 or len({r['item_id'] for r in spec['items']}) != 9:
        raise ValueError('Source episodes or item identities are duplicated')
    legacy = load('fewpair_legacy_native_validation', SCALING/'lora/fresh_lora.py')
    cache = {'hashes': {}}
    for name in manifest['artifacts']:
        if Path(name).name == 'identity.json':
            identity = read(name)
            verify(identity)
            cache['hashes'].update(identity)
    for pair, item in zip(pairs, spec['items'], strict=True):
        a, b = pair['original'], pair['teacher']
        if a['success'] is not False or b['success'] is not True or pair['rejection_reasons']:
            raise ValueError('Selected pair is not a qualified failure/success pair')
        for key in ('slot', 'layout_id', 'policy_seed'):
            if a[key] != b[key]:
                raise ValueError('Original/clarified source pairing differs')
        expected_item = {'pair_id': pair['pair_id'], 'obs_path': a['obs'],
                         'decision_index': pair['selected_decision'], 'split': 'train', 'role': 'repair',
                         'source_cell': 'original', 'source_slot': a['slot'], 'layout_id': a['layout_id'],
                         'original_prompt': g['original'], 'teacher_prompt': g['teacher'],
                         'recorded_prompt': g['original']}
        if any(item.get(k) != v for k, v in expected_item.items()):
            raise ValueError('Source observation, decision or prompt differs from its selected pair')
        if type(item['decision_index']) is not int or item['decision_index'] < 0:
            raise ValueError('Invalid selected decision')
        for cell, row in (('original', a), ('teacher', b)):
            row_expected = {'cohort_id': p['cohort_id'], 'goal_id': goal_id, 'split': 'development',
                            'condition': 'base', 'inference_batching': 'fixed16', 'ok': True,
                            'task_name': g['task'], 'selected_goal': g['goal'], 'cell': cell,
                            'effective_prompt': g[cell]}
            if (any(row.get(k) != v for k, v in row_expected.items())
                    or row['strict_final'].get(g['goal']) is not row['success']):
                raise ValueError('Source prompt or registered success predicate differs')
            for key, hash_key in (('obs', 'obs_sha256'), ('metrics', 'metrics_sha256')):
                path = Path(row[key]).resolve()
                if not path.is_relative_to(root/'collection') or sha(path) != row[hash_key]:
                    raise ValueError('Foreign or changed source observation/metrics')
                cache['hashes'][str(path)] = row[hash_key]
        legacy.N.validate_live_context(item, spec, cache)
    for path in (manifest_path, spec_path, status_path):
        cache['hashes'][str(path)] = sha(path)
    cache['hashes'].update(manifest['artifacts'])
    return manifest_path, manifest, spec_path, spec, cache
