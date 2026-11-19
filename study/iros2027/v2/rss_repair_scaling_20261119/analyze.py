"""Verify all frozen trials and report repair, phrasing transfer, and retention separately."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from common import HERE, PRIOR, ROOT, canonical, load, protocol, read, save, sha


def collect_stage(stage):
    from evaluate import validate_plan
    schedule_path = ROOT / f'{stage}-frozen/schedule.json'
    schedule = read(schedule_path)
    expected_split = 'development' if stage == 'selection' else 'heldout'
    verify = load('scaling_receipt_verification', PRIOR / 'worker.py').verify_global_freeze
    groups, receipts, artifacts = defaultdict(list), [], {str(schedule_path): sha(schedule_path)}
    attempted, checked_metrics = [], {}
    for gpu in (0, 1):
        root = ROOT / stage / f'g{gpu}'
        frozen_plan = ROOT / f'{stage}-frozen/plan-g{gpu}.json'
        if not frozen_plan.exists():
            continue
        plan = read(root / 'plan.json'); validate_plan(plan); verify(plan)
        if plan != read(frozen_plan) or plan['split'] != expected_split or plan['schedule_sha256'] != canonical(schedule):
            raise ValueError('Worker plan differs from the frozen complete schedule')
        if plan['batches'] != [b for b, assigned in zip(schedule['batches'], schedule['gpu_assignment'], strict=True) if assigned == gpu]:
            raise ValueError('Worker violates the frozen device counterbalancing')
        state = read(root / 'status.json')
        if (state['status'] != 'complete' or state['completed_batches'] != len(plan['batches'])
                or state['completed_episodes'] != 16*len(plan['batches'])):
            raise ValueError('Analysis requires every planned trial, without partial-arm selection')
        identity = read(root / 'identity.json')
        if any(sha(path) != digest for path, digest in identity.items()):
            raise ValueError('Recorded worker identity has changed')
        for number, b in enumerate(plan['batches']):
            batch = root / f"batch-{number:02d}-{b['cell']}-{b['condition']}"
            receipt = read(batch / 'completed.json'); rows = read(batch / 'episodes.json')
            if (receipt['episodes_sha256'] != sha(batch / 'episodes.json')
                    or (receipt['batch'], receipt['cell'], receipt['condition'], receipt['n']) != (number, b['cell'], b['condition'], 16)
                    or sorted(r['slot'] for r in rows) != list(range(b['slot_start'], b['slot_start']+16))
                    or len(list(batch.glob('sim-*.log'))) != 1):
                raise ValueError('Batch receipt, slot accounting, or single-attempt contract differs')
            if json.loads(receipt['server']).get('workspace') != b['workspace']:
                raise ValueError('Server loaded another candidate')
            layouts = read(b['layouts'])['layouts']
            for row in rows:
                expected = (protocol()['cohort_id'], 'temporal', expected_split, b['condition'], b['cell'], b['input_prompt'], b['effective_prompt'])
                actual = tuple(row[k] for k in ('cohort_id','case','split','condition','cell','input_prompt','effective_prompt'))
                if (actual != expected or not row['ok'] or row['selected_goal'] != b['goal'] or row['task_name'] != b['task']
                        or row['inference_batching'] != 'fixed16' or row['success'] != row['strict_final'][b['goal']]):
                    raise ValueError('Episode identity or primary outcome differs from its plan')
                if sha(row['obs']) != row['obs_sha256']:
                    raise ValueError('Recorded policy observations changed')
                artifacts[row['obs']] = row['obs_sha256']
                if row['metrics'] not in checked_metrics:
                    checked_metrics[row['metrics']] = sha(row['metrics'])
                if checked_metrics[row['metrics']] != row['metrics_sha256']:
                    raise ValueError('Recorded physical scoring metrics changed')
                metrics = read(row['metrics'])
                if metrics['reset_calls'] != 2 or metrics['layout']['sha256'] != sha(b['layouts']):
                    raise ValueError('Reset or layout receipt mismatch')
                if row['layout_id'] != layouts[row['slot'] % len(layouts)]['id']:
                    raise ValueError('Layout does not match the frozen slot assignment')
                group = 'original' if b['cell'].startswith('original_') else b['cell']
                groups[(b['condition'], group)].append(row)
            if sum(r['success'] for r in rows) != receipt['successes']:
                raise ValueError('Success totals disagree')
            receipts.append({**receipt, 'gpu': gpu, 'split': expected_split})
            attempted.append(b)
            for path in (batch / 'completed.json', batch / 'episodes.json'):
                artifacts[str(path)] = sha(path)
        for path in (root / 'plan.json', root / 'identity.json', root / 'status.json'):
            artifacts[str(path)] = sha(path)
    if sorted(map(canonical, attempted)) != sorted(map(canonical, schedule['batches'])):
        raise ValueError('Executed workers do not cover the complete frozen schedule exactly once')
    for (condition, cell), rows in groups.items():
        rows.sort(key=lambda r: r['slot'])
        n = 32 if cell == 'original' else 16
        counts = Counter(r['layout_id'] for r in rows)
        if len(rows) != n or len({r['slot'] for r in rows}) != n or len(counts) != 16 or set(counts.values()) != {n//16}:
            raise ValueError('Missing trials or unbalanced layout replication')
    artifacts.update(checked_metrics)
    return dict(groups), receipts, artifacts


def interval(values, layout_ids, draws=20000):
    if len(values) != len(layout_ids) or not values:
        raise ValueError('Values and layout identities must be nonempty and aligned')
    groups = defaultdict(list)
    for value, layout in zip(values, layout_ids, strict=True):
        groups[layout].append(float(value))
    if len(set(map(len, groups.values()))) != 1:
        raise ValueError('Cluster bootstrap expects balanced layout replication')
    means = np.array([np.mean(groups[k]) for k in sorted(groups)])
    samples = np.random.default_rng(17).integers(0, len(means), size=(draws, len(means)))
    low, high = np.quantile(means[samples].mean(axis=1), [.025, .975])
    return {'ci_low': float(low), 'ci_high': float(high), 'layout_clusters': len(means),
            'degenerate_ci': bool(np.isclose(low, high))}


def contrast(a, b, condition, reference, cell):
    if len(a) != len(b):
        raise ValueError('Paired arms have different denominators')
    for x, y in zip(a, b, strict=True):
        if (x['slot'], x['layout_id'], x['policy_seed']) != (y['slot'], y['layout_id'], y['policy_seed']):
            raise ValueError('Paired trial identities differ')
    differences = [int(x['success']) - int(y['success']) for x, y in zip(a, b)]
    return {'condition': condition, 'reference': reference, 'cell': cell, 'n': len(a),
            'difference': float(np.mean(differences)),
            'fixed': sum(x['success'] and not y['success'] for x, y in zip(a, b)),
            'broken': sum(y['success'] and not x['success'] for x, y in zip(a, b)),
            'both_success': sum(x['success'] and y['success'] for x, y in zip(a, b)),
            'both_failure': sum(not x['success'] and not y['success'] for x, y in zip(a, b)),
            'first_camera_mismatches': sum(x['first_camera_sha256'] != y['first_camera_sha256'] for x, y in zip(a, b)),
            **interval(differences, [r['layout_id'] for r in a])}


def main():
    output = ROOT / 'analysis-v1'
    if output.exists():
        raise FileExistsError(output)
    groups, receipts, artifacts = collect_stage('heldout')
    selection_groups, selection_receipts, selection_artifacts = collect_stage('selection')
    schedule = read(ROOT / 'heldout-frozen/schedule.json')
    budgets = schedule['available_budgets']
    candidates = {c['candidate_id']: c for c in schedule['candidates']}
    primary, paired, per_layout, scaling = [], [], [], []
    for (condition, cell), rows in sorted(groups.items()):
        values = [r['success'] for r in rows]; layouts = [r['layout_id'] for r in rows]
        primary.append({'condition': condition, 'budget': candidates.get(condition, {}).get('budget'),
                        'cell': cell, 'successes': sum(values), 'n': len(rows), 'rate': float(np.mean(values)),
                        **interval(values, layouts)})
        for layout in sorted(set(layouts)):
            subset = [r for r in rows if r['layout_id'] == layout]
            per_layout.append({'condition': condition, 'cell': cell, 'layout_id': layout,
                               'successes': sum(r['success'] for r in subset), 'n': len(subset)})
        if condition not in ('base', 'clarified'):
            paired.append(contrast(rows, groups[('base', cell)], condition, 'base', cell))
    by_method = defaultdict(dict)
    for name, c in candidates.items():
        method = 'lora' if 'learning_rate' in c else 'weightsmith'
        by_method[method][c['budget']] = name
    for n in budgets:
        edit, lora = by_method['weightsmith'][n], by_method['lora'][n]
        for cell in ('original', 'unseen', 'reverse'):
            paired.append(contrast(groups[(edit, cell)], groups[(lora, cell)], edit, lora, cell))
        paired.append(contrast(groups[(edit, 'original')], groups[('clarified', 'original')], edit, 'clarified', 'original'))
    for method, entries in by_method.items():
        contrasts = list(zip(budgets[1:], budgets[:-1]))
        if len(budgets) > 2:
            contrasts.append((budgets[-1], budgets[0]))
        for larger, smaller in contrasts:
            for cell in ('original', 'unseen', 'reverse'):
                a, b = entries[larger], entries[smaller]
                scaling.append({'method': method, 'larger_budget': larger, 'smaller_budget': smaller,
                                **contrast(groups[(a, cell)], groups[(b, cell)], a, b, cell)})
    source = read(schedule['source_manifest'])
    all_lora = read(ROOT / 'lora-adaptation/candidates.json')['candidates']
    expected_test = 96 + 128*len(budgets)
    expected_dev = 32*len(budgets)
    if sum(r['n'] for r in receipts) != expected_test or sum(r['n'] for r in selection_receipts) != expected_dev:
        raise ValueError('Final experiment accounting differs from available declared milestones')
    report = {'cohort_id': protocol()['cohort_id'], 'status': 'complete_and_verified',
              'available_budgets': budgets, 'new_heldout_policy_episodes': expected_test,
              'new_selection_policy_episodes': expected_dev, 'primary': primary, 'paired': paired,
              'scaling': scaling, 'per_layout': per_layout, 'source_collection': source,
              'selection': read(ROOT / 'selection.json'), 'costs': receipts + selection_receipts,
              'training_reports': {c['candidate_id']: read(c['training_path']) for c in all_lora},
              'training_cost_boundary': 'Both learning rates at every available budget are included, including rejected candidates. Full trajectories contain more states than one decision per correction; report actual chunk counts. Wall-time components are not an end-to-end speed advantage.',
              'statistics': 'Paired, 20,000-draw bootstrap over 16 layout clusters, seed 17. Intervals condition on one fixed nested source ordering and one task; they do not estimate variance over independent correction sets. All declared budgets and cells reported, no winner selection.',
              'limitations': protocol()['limitations'] + [
                  'One unfamiliar phrasing is tested; this does not establish broad linguistic generalization.',
                  'Changed arrangements use the same objects and scene. No new-object or physical-robot claim.',
                  'Averaging more episode corrections keeps 72 fitted rows. This is a sample-budget experiment, not sequential online composition.',
                  'Nominally paired rollouts may have different camera images; none are excluded for this reason.'],
              'source_receipts': {**artifacts, **selection_artifacts},
              'analysis_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'protocol.json')}
    output.mkdir()
    save(output / 'report.json', report)
    for name, rows in (('primary', primary), ('paired', paired), ('scaling', scaling), ('per_layout', per_layout)):
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'report': str(output / 'report.json'), 'primary': primary}, indent=2))


if __name__ == '__main__':
    main()
