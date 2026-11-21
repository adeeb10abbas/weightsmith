"""Verify the complete few-pair study before estimating repair and retention.

Each source triplet is reported separately against its shared controls. Bootstrap
units are held-out layouts, never source-group copies of the same control trials.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from common import HERE, ROOT, canonical, goal, goal_index, goal_root, protocol, read, save, sha, verify


CELLS = ('original', 'unseen', 'competing')
GROUPS = ('g0', 'g1', 'g2')
METHODS = ('base', 'embedding', *(f'weightsmith_{g}' for g in GROUPS),
           *(f'residual_{g}' for g in GROUPS))


def _check_schedule(schedule, goal_id, stage):
    """Reject a complete-looking subset as well as duplicated conditions."""
    if (schedule.get('cohort_id') != protocol()['cohort_id']
            or schedule.get('goal_id') != goal_id or schedule.get('stage') != stage):
        raise ValueError('Schedule belongs to another cohort, goal, or stage')
    batches = schedule['batches']
    assignment = schedule['gpu_assignment']
    if len(assignment) != len(batches) or set(assignment) != {0, 1}:
        raise ValueError('Incomplete device assignment')
    keys = [(b['condition'], b['cell']) for b in batches]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicated condition/cell in the complete schedule')
    candidates = schedule['candidates']
    names = [c['candidate_id'] for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate candidate identity')
    if stage == 'selection':
        if len(candidates) != 12 or 'base' in names:
            raise ValueError('Development selection must include every nonzero candidate')
        for family in ('embedding', *(f'residual_{g}' for g in GROUPS)):
            if sorted(c['strength'] for c in candidates if c['family'] == family) != protocol()['baselines']['strengths']:
                raise ValueError('Incomplete development strength grid')
        expected = {(name, 'selection') for name in ['base', *names]}
    else:
        if set(names) != set(METHODS) - {'base'}:
            raise ValueError('Missing held-out source group or reference method')
        expected = {(name, cell) for name in METHODS
                    for cell in ('original_0', 'original_1', 'unseen', 'competing')}
        expected |= {('clarified', 'original_0'), ('clarified', 'original_1')}
    if set(keys) != expected:
        raise ValueError('Schedule omits or adds a registered condition/cell')
    expected_n = 208 if stage == 'selection' else 544
    if sum(b['n'] for b in batches) != expected_n:
        raise ValueError('Schedule episode accounting differs from the protocol')


def collect_stage(goal_id, stage):
    """Return (groups[(condition, cell)], batch_receipts, checked_artifacts).

    Original_0 and original_1 are joined into the 32-trial original cell.
    This is deliberately strict: unfinished workers do not yield partial arms.
    The selection caller imports this function before held-out freezing.
    """
    from evaluate import validate_evaluation_plan
    from worker import verify_global_freeze

    if stage not in ('selection', 'heldout'):
        raise ValueError('Expected selection or heldout')
    root = goal_root(goal_id)
    g = goal(goal_id)
    schedule_path = root / f'{stage}-frozen/schedule.json'
    schedule = read(schedule_path)
    _check_schedule(schedule, goal_id, stage)
    expected_split = 'development' if stage == 'selection' else 'heldout'
    candidates = {c['candidate_id']: c for c in schedule['candidates']}
    groups, receipts = defaultdict(list), []
    artifacts = {str(schedule_path): sha(schedule_path)}
    attempted, metrics_cache, layout_cache = [], {}, {}
    for gpu in (0, 1):
        worker_root = root / stage / f'g{gpu}'
        frozen_plan = root / f'{stage}-frozen/plan-g{gpu}.json'
        plan = read(worker_root / 'plan.json')
        validate_evaluation_plan(plan)
        frozen_artifacts = verify_global_freeze(plan)
        if not frozen_artifacts:
            raise ValueError('Evaluation requires a nonempty pre-existing freeze')
        if (plan != read(frozen_plan) or plan['goal_id'] != goal_id or plan['stage'] != stage
                or plan['split'] != expected_split or plan['schedule_sha256'] != canonical(schedule)):
            raise ValueError('Worker plan differs from its frozen complete schedule')
        assigned = [b for b, device in zip(schedule['batches'], schedule['gpu_assignment'], strict=True) if device == gpu]
        if plan['batches'] != assigned:
            raise ValueError('Worker violates the frozen device assignment')
        state = read(worker_root / 'status.json')
        if (state.get('status') != 'complete' or state.get('cohort_id') != protocol()['cohort_id']
                or state.get('split') != expected_split or state.get('gpu') != gpu
                or state.get('completed_batches') != len(assigned)
                or state.get('planned_episodes') != 16 * len(assigned)
                or state.get('completed_episodes') != 16 * len(assigned)):
            raise ValueError('Analysis requires all planned trials, without partial-arm selection')
        identity = read(worker_root / 'identity.json')
        verify(identity)
        if any(identity.get(path) != digest for path, digest in frozen_artifacts.items()):
            raise ValueError('Worker identity omits or changes a frozen artifact')
        artifacts.update(identity)
        artifacts[str(frozen_plan)] = sha(frozen_plan)
        for number, b in enumerate(assigned):
            batch = worker_root / f"batch-{number:02d}-{b['cell']}-{b['condition']}"
            receipt = read(batch / 'completed.json')
            rows = read(batch / 'episodes.json')
            logs = sorted(str(p) for p in batch.glob('sim-*.log'))
            if (receipt['episodes_sha256'] != sha(batch / 'episodes.json')
                    or (receipt['batch'], receipt['cell'], receipt['condition'], receipt['n']) != (number, b['cell'], b['condition'], 16)
                    or sorted(r['slot'] for r in rows) != list(range(b['slot_start'], b['slot_start'] + 16))
                    or len(logs) != 1 or receipt['sim_attempt_logs'] != logs):
                raise ValueError('Batch receipt, slot accounting, or single-attempt contract differs')
            server = json.loads(receipt['server'])
            if server.get('workspace') != b['workspace']:
                raise ValueError('Server loaded another candidate')
            candidate = candidates.get(b['condition'])
            if b['workspace'] is not None and (candidate is None or server.get('fingerprint') != candidate['fingerprint']):
                raise ValueError('Loaded candidate fingerprint differs from the frozen registry')
            if b['layouts'] not in layout_cache:
                layout_cache[b['layouts']] = read(b['layouts'])['layouts']
                artifacts[b['layouts']] = sha(b['layouts'])
            layouts = layout_cache[b['layouts']]
            if len(layouts) != 16 or len({x['id'] for x in layouts}) != 16:
                raise ValueError('Expected sixteen distinct frozen layouts')
            for row in rows:
                expected = {'cohort_id': protocol()['cohort_id'], 'goal_id': goal_id,
                    'case': g['runtime_case'], 'split': expected_split, 'condition': b['condition'],
                    'cell': b['cell'], 'input_prompt': b['input_prompt'], 'effective_prompt': b['effective_prompt'],
                    'selected_goal': b['goal'], 'task_name': b['task'], 'inference_batching': 'fixed16', 'ok': True}
                if any(row.get(k) != v for k, v in expected.items()):
                    raise ValueError('Episode identity differs from its frozen plan')
                env = row['slot'] % 16
                if (row['env'] != env or row['policy_seed'] != ((row['slot'] // 16) * 64 + env) * 100
                        or row['layout_id'] != layouts[env]['id'] or type(row['success']) is not bool
                        or row['success'] != row['strict_final'][b['goal']]):
                    raise ValueError('Episode seed, layout, or primary outcome differs')
                if not Path(row['obs']).resolve().is_relative_to(batch):
                    raise ValueError('Observation belongs to another batch')
                if sha(row['obs']) != row['obs_sha256']:
                    raise ValueError('Recorded policy observations changed')
                artifacts[row['obs']] = row['obs_sha256']
                if not Path(row['metrics']).resolve().is_relative_to(batch):
                    raise ValueError('Physical metrics belong to another batch')
                if row['metrics'] not in metrics_cache:
                    metrics_cache[row['metrics']] = (sha(row['metrics']), read(row['metrics']))
                digest, metrics = metrics_cache[row['metrics']]
                if digest != row['metrics_sha256']:
                    raise ValueError('Recorded physical scoring metrics changed')
                if (metrics['reset_calls'] != 2 or metrics['layout']['sha256'] != artifacts[b['layouts']]
                        or metrics['selected_goal'] != b['goal']
                        or metrics['layout']['layout_ids'][env] != row['layout_id']
                        or bool(metrics['strict_final'][b['goal']][env]) != row['success']):
                    raise ValueError('Physical scoring, reset, or layout receipt mismatch')
                artifacts[row['metrics']] = digest
                cell = 'original' if b['cell'].startswith('original_') else b['cell']
                groups[(b['condition'], cell)].append(row)
            if sum(r['success'] for r in rows) != receipt['successes']:
                raise ValueError('Success totals disagree')
            receipts.append({**receipt, 'goal_id': goal_id, 'gpu': gpu, 'split': expected_split,
                             'stage': stage, 'receipt': str(batch / 'completed.json')})
            attempted.append(b)
            for path in (batch / 'completed.json', batch / 'episodes.json', *map(Path, logs)):
                artifacts[str(path)] = sha(path)
        for path in (worker_root / 'plan.json', worker_root / 'identity.json', worker_root / 'status.json'):
            artifacts[str(path)] = sha(path)
    if sorted(map(canonical, attempted)) != sorted(map(canonical, schedule['batches'])):
        raise ValueError('Workers do not cover the complete schedule exactly once')
    for (_, cell), rows in groups.items():
        rows.sort(key=lambda r: r['slot'])
        n = 32 if cell == 'original' else 16
        counts = Counter(r['layout_id'] for r in rows)
        if len(rows) != n or len({r['slot'] for r in rows}) != n or len(counts) != 16 or set(counts.values()) != {n // 16}:
            raise ValueError('Missing trials or unbalanced layout replication')
    return dict(groups), receipts, artifacts


def interval(values, layout_ids, draws=None):
    """Percentile bootstrap of balanced layout means, with a fixed random seed."""
    if len(values) != len(layout_ids) or not len(values):
        raise ValueError('Values and layout identities must be nonempty and aligned')
    clusters = defaultdict(list)
    for value, layout in zip(values, layout_ids, strict=True):
        clusters[layout].append(float(value))
    if len(set(map(len, clusters.values()))) != 1:
        raise ValueError('Cluster bootstrap expects balanced layout replication')
    means = np.asarray([np.mean(clusters[k]) for k in sorted(clusters)])
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite outcomes')
    draws = protocol()['analysis']['bootstrap_draws'] if draws is None else draws
    if draws < 1:
        raise ValueError('Bootstrap draw count must be positive')
    sample = np.random.default_rng(17).integers(0, len(means), size=(draws, len(means)))
    low, high = np.quantile(means[sample].mean(axis=1), [.025, .975])
    return {'ci_low': float(low), 'ci_high': float(high), 'layout_clusters': len(means),
            'degenerate_ci': bool(np.isclose(low, high))}


def contrast(a, b, condition, reference, cell):
    a, b = sorted(a, key=lambda r: r['slot']), sorted(b, key=lambda r: r['slot'])
    if not a or len(a) != len(b) or len({r['slot'] for r in a}) != len(a):
        raise ValueError('Paired arms have different or duplicated trial identities')
    for x, y in zip(a, b, strict=True):
        if (x['slot'], x['layout_id'], x['policy_seed']) != (y['slot'], y['layout_id'], y['policy_seed']):
            raise ValueError('Paired trial identities differ')
    differences = [int(x['success']) - int(y['success']) for x, y in zip(a, b)]
    return {'condition': condition, 'reference': reference, 'cell': cell, 'n': len(a),
            'difference': float(np.mean(differences)),
            'fixed': sum(bool(x['success'] and not y['success']) for x, y in zip(a, b)),
            'broken': sum(bool(y['success'] and not x['success']) for x, y in zip(a, b)),
            'both_success': sum(bool(x['success'] and y['success']) for x, y in zip(a, b)),
            'both_failure': sum(bool(not x['success'] and not y['success']) for x, y in zip(a, b)),
            'first_camera_mismatches': sum(x['first_camera_sha256'] != y['first_camera_sha256'] for x, y in zip(a, b)),
            **interval(differences, [r['layout_id'] for r in a])}


def summarize(groups):
    """Describe each arm once; never turn the shared base into three cohorts."""
    expected = {(condition, cell) for condition in METHODS for cell in CELLS} | {('clarified', 'original')}
    if set(groups) != expected:
        raise ValueError('Incomplete held-out analysis matrix')
    primary, paired, per_layout, source_group_summary = [], [], [], []
    for (condition, cell), rows in sorted(groups.items()):
        values, layouts = [r['success'] for r in rows], [r['layout_id'] for r in rows]
        primary.append({'condition': condition, 'cell': cell, 'successes': sum(values),
                        'n': len(rows), 'rate': float(np.mean(values)), **interval(values, layouts)})
        for layout in sorted(set(layouts)):
            subset = [r for r in rows if r['layout_id'] == layout]
            per_layout.append({'condition': condition, 'cell': cell, 'layout_id': layout,
                               'successes': sum(r['success'] for r in subset), 'n': len(subset)})
        if condition != 'base':
            paired.append(contrast(rows, groups[('base', cell)], condition, 'base', cell))
    for group in GROUPS:
        name = f'weightsmith_{group}'
        for cell in CELLS:
            for reference in ('embedding', f'residual_{group}'):
                paired.append(contrast(groups[(name, cell)], groups[(reference, cell)], name, reference, cell))
        paired.append(contrast(groups[(name, 'original')], groups[('clarified', 'original')], name, 'clarified', 'original'))
    for method in ('weightsmith', 'residual'):
        for cell in CELLS:
            rates = [float(np.mean([r['success'] for r in groups[(f'{method}_{g}', cell)]])) for g in GROUPS]
            source_group_summary.append({'method': method, 'cell': cell, 'source_groups': 3,
                'mean_rate': float(np.mean(rates)), 'min_rate': min(rates), 'max_rate': max(rates),
                'interpretation': 'Descriptive across disjoint source triplets; shared layouts and controls, no pooled confidence interval.'})
    return {'primary': primary, 'paired': paired, 'per_layout': per_layout, 'source_group_summary': source_group_summary}


def source_accounting(goal_id):
    """Include all qualification attempts and failures, not only the nine used."""
    from prepare_sources import read_batch
    root = goal_root(goal_id)
    status_path = root / 'source-status.json'
    status = read(status_path)
    if status['status'] not in ('qualified', 'unqualified_at_cap'):
        raise ValueError('Source qualification is not complete')
    manifest_path = Path(status['manifest'])
    if sha(manifest_path) != status['manifest_sha256']:
        raise ValueError('Source manifest changed')
    source = read(manifest_path)
    verify(source['artifacts'])
    if (source['cohort_id'] != protocol()['cohort_id'] or source['goal_id'] != goal_id
            or not source['fresh_collection'] or source['protocol_sha256'] != sha(HERE / 'protocol.json')
            or source['qualified'] != (status['status'] == 'qualified')):
        raise ValueError('Source qualification identity differs')
    attempts = source['attempts']
    n = source['attempted_pairs']
    if (n != len(attempts) or n % 16 or not 16 <= n <= 48 or source['policy_episodes'] != 2 * n
            or status['attempted_pairs'] != n or status['eligible_pairs'] != source['eligible_pairs']
            or source['eligible_pairs'] != sum(bool(r['eligible']) for r in attempts)):
        raise ValueError('Source acquisition accounting differs')
    if not source['qualified'] and (n != 48 or source['eligible_pairs'] >= 9):
        raise ValueError('Unqualified goal did not exhaust its fixed source cap')
    selected = [r for r in attempts if r['eligible']][:9]
    if source['pairs'] != selected or (source['qualified'] and len(selected) != 9):
        raise ValueError('Source pool is not the first nine eligible pairs')
    if source['qualified']:
        from prepare_sources import partition_pairs
        expected_seed = protocol()['collection']['grouping_seed'] + goal_index(goal_id)
        if (source['grouping_seed'] != expected_seed
                or source['groups'] != partition_pairs([r['pair_id'] for r in selected], expected_seed)):
            raise ValueError('Source groups differ from their frozen disjoint partition')
        if n > 16 and sum(bool(r['eligible']) for r in attempts[:-16]) >= 9:
            raise ValueError('Source acquisition continued beyond the first qualifying block')
    elif source['groups']:
        raise ValueError('Unqualified goal contains source groups')
    receipts, reconstructed = [], []
    artifacts = {**source['artifacts'], str(status_path): sha(status_path), str(manifest_path): sha(manifest_path)}
    for block in range(n // 16):
        pairs = {}
        for cell in ('original', 'teacher'):
            _, batch, rows = read_batch(goal_id, block, cell)
            pairs[cell] = rows
            receipt = read(batch / 'completed.json')
            if receipt['successes'] != sum(r['success'] for r in rows):
                raise ValueError('Source success totals disagree')
            receipts.append({**receipt, 'goal_id': goal_id, 'stage': 'collection', 'receipt': str(batch / 'completed.json')})
            for row in rows:
                artifacts[row['obs']] = row['obs_sha256']
                artifacts[row['metrics']] = row['metrics_sha256']
        reconstructed.extend(zip(pairs['original'], pairs['teacher'], strict=True))
    for recorded, (original, teacher) in zip(attempts, reconstructed, strict=True):
        if recorded['original'] != original or recorded['teacher'] != teacher:
            raise ValueError('Source manifest no longer matches complete worker receipts')
    return {'status': status['status'], 'qualified': source['qualified'], 'attempted_pairs': n,
        'eligible_pairs': source['eligible_pairs'], 'selected_pairs': len(selected) if source['qualified'] else 0,
        'construction_states': len(selected) if source['qualified'] else 0,
        'source_groups': source['groups'], 'source_policy_episodes': 2 * n,
        'source_original_successes': sum(r['original']['success'] for r in attempts),
        'source_clarified_successes': sum(r['teacher']['success'] for r in attempts),
        'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path)}, receipts, artifacts


def _cost_totals(receipts):
    return {stage: {'policy_episodes': sum(r['n'] for r in receipts if r['stage'] == stage),
        **{field: sum(float(r[field]) for r in receipts if r['stage'] == stage)
           for field in ('wall_seconds', 'queue_seconds', 'execution_seconds')}}
        for stage in ('collection', 'selection', 'heldout')}


def analyze_goal(goal_id):
    source, source_receipts, artifacts = source_accounting(goal_id)
    report = {'cohort_id': protocol()['cohort_id'], 'goal_id': goal_id, 'source_collection': source,
              'analysis_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'protocol.json')}
    if not source['qualified']:
        report.update(status='qualification_failed_verified', new_heldout_policy_episodes=0,
                      new_selection_policy_episodes=0, costs=_cost_totals(source_receipts), source_receipts=artifacts)
        return report
    global_path = ROOT / 'all-heldout-frozen.json'
    global_freeze = read(global_path)
    if (global_freeze.get('cohort_id') != protocol()['cohort_id'] or global_freeze.get('status') != 'frozen'
            or set(global_freeze['goals']) != set(protocol()['goal_ids'])
            or not global_freeze['goals'][goal_id]['qualified']):
        raise ValueError('Missing complete pre-test freeze for all registered goals')
    verify(global_freeze['artifacts'])
    artifacts.update(global_freeze['artifacts'])
    artifacts[str(global_path)] = sha(global_path)
    groups, receipts, heldout_artifacts = collect_stage(goal_id, 'heldout')
    selection_groups, selection_receipts, selection_artifacts = collect_stage(goal_id, 'selection')
    schedule = read(goal_root(goal_id) / 'heldout-frozen/schedule.json')
    selection_schedule = read(goal_root(goal_id) / 'selection-frozen/schedule.json')
    selection_path = goal_root(goal_id) / 'selection.json'
    selection = read(selection_path)
    from evaluate import select_candidates
    expected_selection = select_candidates(selection_schedule['candidates'], {k[0]: v for k, v in selection_groups.items()})
    if (selection['heldout_used'] is not False or selection['candidates'] != expected_selection
            or selection['development_episodes'] != 208
            or [c for c in schedule['candidates'] if c['candidate_id'] in {'embedding', 'residual_g0', 'residual_g1', 'residual_g2'}] != expected_selection):
        raise ValueError('Held-out references differ from the complete development selection')
    if schedule['source_manifest_sha256'] != source['source_manifest_sha256']:
        raise ValueError('Evaluation uses a different source pool')
    all_receipts = source_receipts + selection_receipts + receipts
    report.update(status='complete_and_verified', new_heldout_policy_episodes=sum(r['n'] for r in receipts),
        new_selection_policy_episodes=sum(r['n'] for r in selection_receipts), **summarize(groups),
        candidates=schedule['candidates'], selection=selection, costs=_cost_totals(all_receipts),
        batch_costs=all_receipts, source_receipts={**artifacts, **heldout_artifacts, **selection_artifacts,
                                                  str(selection_path): sha(selection_path)},
        statistics='Paired percentile 95% intervals from 20,000 bootstrap draws over 16 layout clusters, seed 17. Original-request trials have two noise replicates per layout. Each source triplet has its own estimate; shared controls and layouts are not independent cohorts. Intervals are descriptive and not multiplicity-adjusted; no equivalence or non-inferiority margin was registered.',
        cost_boundary='All attempted source pairs and all development strengths are counted. Batch wall and execution times are summed resource durations and may overlap across GPUs; they are not end-to-end elapsed time. Native qualification and activation-capture work are outside policy-episode counts.',
        limitations=['Three historically informed goals in one model and existing scenes; same objects with changed arrangements.',
            'Only one unseen phrasing and one competing goal per repair are tested.',
            'Three source sets are disjoint but share the held-out scenes and control episodes; group means and ranges are descriptive.',
            'Nominal scene/noise pairing does not imply identical rendered observations; camera mismatches are retained and reported.',
            'Zero-strength selected references use unchanged weights in separately executed nominal arms; rollout differences are not evidence of a distinct intervention.',
            'A nonsignificant competing-goal difference does not establish preservation or equivalence.'])
    return report


def aggregate_reports(reports):
    if [r['goal_id'] for r in reports] != protocol()['goal_ids']:
        raise ValueError('Aggregate must retain every registered goal, in order')
    if any(r['status'] not in ('complete_and_verified', 'qualification_failed_verified') for r in reports):
        raise ValueError('Cannot aggregate unfinished or unverified goals')
    return {'cohort_id': protocol()['cohort_id'], 'status': 'complete_and_verified',
        'registered_goals': len(reports), 'qualified_goals': sum(r['source_collection']['qualified'] for r in reports),
        'goals': reports, 'new_source_policy_episodes': sum(r['source_collection']['source_policy_episodes'] for r in reports),
        'new_selection_policy_episodes': sum(r['new_selection_policy_episodes'] for r in reports),
        'new_heldout_policy_episodes': sum(r['new_heldout_policy_episodes'] for r in reports),
        'shared_control_accounting': 'Each qualified goal contributes one 64-trial base cohort, one 64-trial embedding cohort, and one 32-trial clarified cohort. Source-group comparisons reuse these trials; they do not multiply the control sample size.',
        'scope': 'All registered goals, including failed source qualification. No pooled cross-goal or source-group success interval.',
        'analysis_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'protocol.json')}


def write_report(output, report):
    output = Path(output)
    save(output / 'report.json', report, exclusive=True)
    for name in ('primary', 'paired', 'per_layout', 'source_group_summary'):
        rows = report.get(name, [])
        if rows:
            path = output / f'{name}.csv'
            if path.exists():
                continue  # The immutable JSON above is authoritative on repeat invocation.
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--goal', choices=protocol()['goal_ids'])
    choice.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        reports = [analyze_goal(gid) for gid in protocol()['goal_ids']]
        for report in reports:
            write_report(goal_root(report['goal_id']) / 'analysis-v1', report)
        report = aggregate_reports(reports)
        output = ROOT / 'analysis-v1'
    else:
        report = analyze_goal(args.goal)
        output = goal_root(args.goal) / 'analysis-v1'
    write_report(output, report)
    print(json.dumps({'report': str(output / 'report.json'), 'status': report['status']}, indent=2))


if __name__ == '__main__':
    main()
