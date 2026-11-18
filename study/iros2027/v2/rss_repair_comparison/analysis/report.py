"""Verify and summarize the frozen 640-episode RSS comparison.

Partial mode reports completion/integrity only. It never estimates outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


class VerificationError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__('Comparison is incomplete or invalid: ' + '; '.join(status['errors'][:5]))


def expected_keys(protocol):
    n = protocol['heldout']['episodes_per_cell_condition']
    return {(case, cell['id'], condition, slot)
            for case, definition in protocol['cases'].items()
            for cell in definition['cells'] for condition in protocol['conditions']
            for slot in range(protocol['heldout']['slot_start'][case],
                              protocol['heldout']['slot_start'][case] + n)}


def collect(protocol_path, roots):
    """Trust only receipted batches; errors are retained for status-only reports."""
    protocol_path = Path(protocol_path).resolve()
    protocol = read(protocol_path)
    wanted = expected_keys(protocol)
    errors, inventory, records, identities = [], [], [], {}
    if len(wanted) != 640 or protocol['heldout']['planned_total'] != 640:
        errors.append('This registered analysis requires exactly 640 planned policy episodes')
    n = protocol['heldout']['episodes_per_cell_condition']
    if (protocol['heldout']['layouts'], protocol['heldout']['policy_noise_replicates'], n) != (8, 2, 16):
        errors.append('Expected eight layouts, two noise replicates, and 16 episodes per cell/condition')
    checked_metrics = {}
    for case in protocol['cases']:
        root = Path(roots[case]).resolve()
        try:
            plan, state, identity = (read(root/name) for name in ('plan.json', 'status.json', 'identity.json'))
            if plan['case'] != case or plan['split'] != 'heldout' or plan['cohort_id'] != protocol['cohort']:
                raise ValueError('Worker plan is not this case/cohort/heldout split')
            if not any(Path(path).name == 'protocol.json' and digest == sha(protocol_path)
                       for path, digest in identity.items()):
                raise ValueError('Worker did not freeze this exact protocol')
            identities[case] = {'root': str(root), 'identity_sha256': sha(root/'identity.json'),
                                'plan_sha256': sha(root/'plan.json'), 'worker_status': state['status']}
            if state['status'] != 'complete':
                errors.append(f'{case}: worker status is {state["status"]}')
            cells = {cell['id']: cell for cell in protocol['cases'][case]['cells']}
            wanted_batches = {(cell, condition) for cell in cells for condition in protocol['conditions']}
            got_batches = [(batch['cell'], batch['condition']) for batch in plan['batches']]
            if set(got_batches) != wanted_batches or len(got_batches) != len(wanted_batches):
                raise ValueError('Plan does not contain every cell/condition exactly once')
            expected_paths = set()
            for number, batch_spec in enumerate(plan['batches']):
                cell, condition = batch_spec['cell'], batch_spec['condition']
                batch = root/f'batch-{number:02d}-{cell}-{condition}'
                expected_paths.add(batch)
                info = {'case': case, 'cell': cell, 'condition': condition,
                        'batch': str(batch), 'status': 'missing_receipt', 'verified_episodes': 0}
                inventory.append(info)
                try:
                    receipt = read(batch/'completed.json')
                    if receipt['episodes_sha256'] != sha(batch/'episodes.json'):
                        raise ValueError('Episode receipt checksum mismatch')
                    rows = read(batch/'episodes.json')
                    if (receipt['batch'], receipt['cell'], receipt['condition'], receipt['n']) != (number, cell, condition, n):
                        raise ValueError('Receipt identity/count mismatch')
                    if len(rows) != n or batch_spec.get('n', 16) != n:
                        raise ValueError('Completed batch does not contain the registered 16 episodes')
                    workspace = batch_spec.get('workspace')
                    if condition in {'base', 'saved_rewrite'} and workspace:
                        raise ValueError('Unedited comparator unexpectedly loads a weight edit')
                    if condition in {'closed_form', 'lora'}:
                        if not workspace:
                            raise ValueError('Adapted condition has no workspace')
                        workspace = Path(workspace).resolve()
                        if str(workspace/'workspace.json') not in identity or not any(Path(p).parent == workspace/'edits' for p in identity):
                            raise ValueError('Adapted workspace manifest and edited tensors were not frozen')
                    start = protocol['heldout']['slot_start'][case]
                    if batch_spec['slot_start'] != start or sorted(r['slot'] for r in rows) != list(range(start, start+n)):
                        raise ValueError('Completed batch has duplicate, missing, or unexpected slots')
                    for row in rows:
                        if (row['case'], row['cell'], row['condition'], row['cohort_id'], row['split']) != (
                                case, cell, condition, protocol['cohort'], 'heldout'):
                            raise ValueError('Episode identity mismatch')
                        if row['selected_goal'] != cells[cell]['goal']:
                            raise ValueError('Selected goal differs from the frozen cell')
                        if row['input_prompt'] != cells[cell]['prompt'] or row['effective_prompt'] != batch_spec['effective_prompt']:
                            raise ValueError('Prompt differs from the frozen plan/protocol')
                        expected_prompt = cells[cell]['rewrite']['effective_prompt'] if condition == 'saved_rewrite' else cells[cell]['prompt']
                        if row['effective_prompt'] != expected_prompt:
                            raise ValueError('Condition uses an unexpected effective instruction')
                        goals = {c['goal'] for c in cells.values()}
                        if any(type(row[k].get(goal)) is not bool for k in ('strict_final', 'strict_ever', 'stock_final', 'stock_ever') for goal in goals):
                            raise ValueError('Missing or nonboolean task outcomes')
                        if any(row['strict_final'][goal] and not row['strict_ever'][goal] for goal in goals):
                            raise ValueError('A final strict outcome is missing from ever-achieved outcomes')
                        if type(row['success']) is not bool or row['success'] != row['strict_final'][row['selected_goal']]:
                            raise ValueError('Primary success disagrees with strict selected-goal final success')
                        if not row.get('ok') or type(row['policy_seed']) is not int or row['policy_seed'] < 0:
                            raise ValueError('Incomplete episode or invalid sampling seed')
                        if set(row['first_camera_sha256']) != {'base_img', 'wrist_img'}:
                            raise ValueError('Initial camera identity unavailable')
                        metrics = row['metrics']
                        if metrics not in checked_metrics:
                            checked_metrics[metrics] = sha(metrics)
                        if checked_metrics[metrics] != row['metrics_sha256']:
                            raise ValueError('Metrics checksum mismatch')
                    if receipt['successes'] != sum(r['success'] for r in rows):
                        raise ValueError('Receipt success count differs from sealed episode records')
                    counts = Counter(row['layout_id'] for row in rows)
                    if len(counts) != 8 or set(counts.values()) != {2}:
                        raise ValueError('Each batch must contain eight layouts with two replicates each')
                    info.update(status='verified', verified_episodes=len(rows))
                    records.extend(rows)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    info['status'] = 'invalid_or_incomplete'
                    info['error'] = str(exc)
                    errors.append(f'{case}/{cell}/{condition}: {exc}')
            unexpected = [p for p in root.glob('batch-*') if p not in expected_paths and (p/'completed.json').exists()]
            if unexpected:
                errors.append(f'{case}: unexpected receipted batches: {unexpected}')
            if state.get('completed_episodes') != sum(i['verified_episodes'] for i in inventory if i['case'] == case):
                errors.append(f'{case}: worker completion count differs from verified receipts')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f'{case}: {exc}')
    keys = [(r['case'], r['cell'], r['condition'], r['slot']) for r in records]
    if len(keys) != len(set(keys)):
        errors.append('Duplicate case/cell/condition/slot records')
    missing, unexpected = wanted-set(keys), set(keys)-wanted
    if missing:
        errors.append(f'{len(missing)} registered policy episode records are missing')
    if unexpected:
        errors.append(f'{len(unexpected)} unexpected policy episode records')
    indexed = {key: row for key, row in zip(keys, records, strict=True)}
    for (case, cell, condition, slot), row in indexed.items():
        base = indexed.get((case, cell, 'base', slot))
        if base and (row['layout_id'], row['policy_seed']) != (base['layout_id'], base['policy_seed']):
            errors.append(f'Pair layout/seed mismatch: {case}/{cell}/{condition}/{slot}')
    status = {'mode': 'status_only', 'planned_new_policy_episodes': 640,
              'verified_new_policy_episodes': len(records), 'complete_and_verified': not errors and len(records) == 640,
              'protocol_sha256': sha(protocol_path), 'worker_identities': identities,
              'errors': errors, 'inventory': inventory,
              'observation_hash_verification': 'Recorded observation hashes retained; raw observation files are not reread. Episode receipts and referenced metric-file hashes are verified.'}
    return protocol, records, status


def clustered_interval(values, layouts, *, draws=20000, seed=17):
    grouped = defaultdict(list)
    for value, layout in zip(values, layouts, strict=True):
        grouped[layout].append(float(value))
    means = np.asarray([np.mean(grouped[layout]) for layout in sorted(grouped)])
    if len(means) != 8 or len(set(map(len, grouped.values()))) != 1:
        raise ValueError('Registered bootstrap requires eight equally sized layout clusters')
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    degenerate = bool(np.isclose(low, high, rtol=0, atol=1e-12))
    return {'ci_low': float(low), 'ci_high': float(high), 'layout_clusters': len(means),
            'degenerate_ci': degenerate,
            'ci_warning': 'Only 8 layout clusters; exploratory percentile interval, not a population or equivalence claim.' +
                          (' Degenerate interval does not establish certainty.' if degenerate else '')}


def summarize(protocol, records, status, *, draws=20000, seed=17):
    if not status['complete_and_verified']:
        raise VerificationError(status)
    groups = defaultdict(list)
    for row in records:
        groups[row['case'], row['cell'], row['condition']].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row['slot'])
    primary, secondary, paired, collisions, cameras, exact_summary, exact_rows = [], [], [], [], [], [], []

    def rates(case, cell, condition, rows, reuse='new'):
        successes = sum(row['success'] for row in rows)
        return {'case': case, 'cell': cell, 'condition': condition, 'successes': successes,
                'n': len(rows), 'success_rate': successes/len(rows), 'reuse_or_new': reuse,
                **clustered_interval([r['success'] for r in rows], [r['layout_id'] for r in rows], draws=draws, seed=seed)}

    comparisons = [('saved_rewrite', 'base'), ('closed_form', 'base'), ('lora', 'base'),
                   ('closed_form', 'saved_rewrite'), ('closed_form', 'lora')]
    for case, definition in protocol['cases'].items():
        goals = sorted({cell['goal'] for cell in definition['cells']})
        for cell in definition['cells']:
            cid = cell['id']
            for condition in protocol['conditions']:
                rows = groups[case, cid, condition]
                primary.append(rates(case, cid, condition, rows))
                stock = sum(row['stock_final'][cell['goal']] for row in rows)
                secondary.append({'case':case,'cell':cid,'condition':condition,'stock_compatible_successes':stock,
                                  'n':len(rows),'stock_compatible_success_rate':stock/len(rows),
                                  'interpretation':'Secondary stock-compatible predicate; never substituted for strict primary success.'})
                for competing in goals:
                    if competing == cell['goal']:
                        continue
                    collisions.append({'case': case, 'cell': cid, 'condition': condition,
                        'selected_goal': cell['goal'], 'competing_goal': competing, 'n': len(rows),
                        'competing_final': sum(r['strict_final'][competing] for r in rows),
                        'selected_and_competing_final': sum(r['success'] and r['strict_final'][competing] for r in rows),
                        'competing_only_final': sum(not r['success'] and r['strict_final'][competing] for r in rows),
                        'competing_ever': sum(r['strict_ever'][competing] for r in rows),
                        'selected_and_competing_ever': sum(r['strict_ever'][cell['goal']] and r['strict_ever'][competing] for r in rows),
                        'ever_interpretation': 'May occur at different times or as a prerequisite; not a simultaneous collision claim.'})
            for condition, reference in comparisons:
                candidate, baseline = groups[case, cid, condition], groups[case, cid, reference]
                differences = []
                fixed = broken = both_success = both_failure = 0
                for a, b in zip(candidate, baseline, strict=True):
                    if (a['slot'], a['layout_id'], a['policy_seed']) != (b['slot'], b['layout_id'], b['policy_seed']):
                        raise ValueError('Pair identity mismatch after validation')
                    differences.append(int(a['success'])-int(b['success']))
                    fixed += int(a['success'] and not b['success'])
                    broken += int(b['success'] and not a['success'])
                    both_success += int(a['success'] and b['success'])
                    both_failure += int(not a['success'] and not b['success'])
                    cameras.append({'case': case, 'cell': cid, 'condition': condition, 'reference': reference,
                                    'slot': a['slot'], 'layout_id': a['layout_id'], 'policy_seed': a['policy_seed'],
                                    'base_camera_identical': a['first_camera_sha256']['base_img'] == b['first_camera_sha256']['base_img'],
                                    'wrist_camera_identical': a['first_camera_sha256']['wrist_img'] == b['first_camera_sha256']['wrist_img'],
                                    'excluded': False})
                paired.append({'case': case, 'cell': cid, 'condition': condition, 'reference': reference,
                               'n': len(candidate), 'fixed': fixed, 'broken': broken,
                               'both_success': both_success, 'both_failure': both_failure,
                               'success_rate_difference': float(np.mean(differences)),
                               **clustered_interval(differences, [r['layout_id'] for r in candidate], draws=draws, seed=seed)})
            exact_prompt = cell['exact_rewrite']['effective_prompt']
            source = next((condition for condition in ('base', 'saved_rewrite')
                           if all(r['effective_prompt'] == exact_prompt for r in groups[case, cid, condition])), None)
            if source is None:
                raise ValueError(f'No identical effective-prompt evidence for exact cache: {case}/{cid}')
            source_rows = groups[case, cid, source]
            derived = rates(case, cid, 'exact_cache_derived', source_rows, 'reused identical effective-prompt episodes')
            derived.update(source_condition=source, new_independent_policy_episodes=0)
            exact_summary.append(derived)
            exact_rows.extend({'case': case, 'cell': cid, 'slot': row['slot'], 'success': row['success'],
                               'source_condition': source, 'source_obs': row['obs'],
                               'source_obs_sha256': row['obs_sha256'], 'effective_prompt': exact_prompt,
                               'reuse_or_new': 'reused', 'new_independent_policy_episodes': 0} for row in source_rows)
    mismatch = sum(not r['base_camera_identical'] or not r['wrist_camera_identical'] for r in cameras)
    return {'mode': 'complete_comparison', 'verification': status,
            'evidence_counts': {'new_policy_episodes': len(records), 'derived_exact_cache_records': len(exact_rows),
                                'derived_exact_cache_new_independent_episodes': 0},
            'statistical_method': {'bootstrap_unit': 'layout, retaining both policy-noise replicates together',
                                   'draws': draws, 'seed': seed, 'confidence_level': .95,
                                   'interpretation': 'Exploratory paired layout-cluster percentile intervals; no p values, equivalence tests, or population-generalization claim.'},
            'pairing_diagnostics': {'comparisons': len(cameras), 'initial_camera_mismatches': mismatch,
                                    'automatically_excluded_episodes': 0,
                                    'note': 'First camera hashes are a reset/render diagnostic. Mismatches remain in all tables; investigate before attributing paired differences solely to repair.'},
            'primary': primary, 'stock_compatible':secondary, 'paired': paired, 'collisions': collisions,
            'camera_pairs': cameras, 'exact_cache_derived': exact_summary, 'exact_cache_rows': exact_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--temporal-root', type=Path, required=True)
    parser.add_argument('--spatial-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Produce completion/integrity status only, never outcome tables')
    parser.add_argument('--bootstrap-draws', type=int, default=20000)
    parser.add_argument('--bootstrap-seed', type=int, default=17)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        parser.error('At least 1000 bootstrap draws required')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, rows, status = collect(args.protocol, {'temporal': args.temporal_root, 'spatial': args.spatial_root})
    write(args.output/'status.json', status)
    write_csv(args.output/'inventory.csv', status['inventory'])
    if args.allow_partial:
        print(json.dumps({k: status[k] for k in ('mode', 'verified_new_policy_episodes', 'complete_and_verified')}))
        return
    try:
        report = summarize(protocol, rows, status, draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    except VerificationError as exc:
        parser.exit(2, str(exc) + '\nStatus saved in ' + str(args.output/'status.json') + '\n')
    report['analysis_code_sha256'] = sha(__file__)
    write(args.output/'report.json', report)
    for key, filename in [('primary', 'primary.csv'), ('stock_compatible','stock_compatible.csv'),
                          ('paired', 'paired.csv'), ('collisions', 'collisions.csv'),
                          ('camera_pairs', 'pairing_diagnostics.csv'), ('exact_cache_derived', 'exact_cache_derived.csv')]:
        write_csv(args.output/filename, report[key])
    write(args.output/'exact_cache_rows.json', report['exact_cache_rows'])
    print(json.dumps({'mode': report['mode'], **report['evidence_counts'],
                      'initial_camera_mismatches': report['pairing_diagnostics']['initial_camera_mismatches']}))


if __name__ == '__main__':
    main()
