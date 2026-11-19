"""Freeze the full selection and test schedules before any corresponding rollouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

from common import HERE, POOL, PRIOR, ROOT, canonical, load, protocol, read, save, sha, source_inventory


def candidate_identity(candidate):
    from weightsmith import config as C
    from weightsmith.edits import fingerprint
    workspace = Path(candidate['workspace']).resolve()
    manifest = read(workspace / 'workspace.json')
    base = read(C.BASE_WS / 'workspace.json')
    for key in ('base_vlm_sha256', 'checkpoint', 'total_parameters_and_buffers'):
        if manifest[key] != base[key]:
            raise ValueError(f'Candidate base mismatch: {key}')
    if (workspace / 'base').resolve() != (C.BASE_WS / 'base').resolve():
        raise ValueError('Candidate does not use the declared immutable base')
    edits = sorted((workspace / 'edits').glob('*.npy'))
    names = {v['file']: k for k, v in manifest['tensors'].items()}
    expected = {f'language.blocks.{i:02d}.mlp.linear' for i in protocol()['edit']['layers']}
    if len(edits) != len(expected) or {names[p.name] for p in edits} != expected:
        raise ValueError('Candidate writes outside the specified projections')
    artifacts = {str(p): sha(p) for p in [workspace / 'workspace.json', *edits]}
    for path, digest in candidate.get('artifacts', {}).items():
        if sha(path) != digest:
            raise ValueError(f'Candidate artifact changed: {path}')
        artifacts[path] = digest
    return {**candidate, 'workspace': str(workspace), 'fingerprint': fingerprint(workspace / 'edits'),
            'artifacts': artifacts}


def registry(path, method):
    record = read(path)
    if not record.get('source_manifest') or not record.get('source_manifest_sha256'):
        raise ValueError('Registry must bind its complete development source manifest')
    source = read(record['source_manifest'])
    if (sha(record['source_manifest']) != record['source_manifest_sha256']
            or source.get('cohort_id') != protocol()['cohort_id'] or source.get('split') != 'development'):
        raise ValueError('Registry source identity differs from this development cohort')
    candidates = [candidate_identity(c) for c in record['candidates']]
    budgets = sorted({int(c['budget']) for c in candidates})
    if not budgets or any(n not in protocol()['budgets'] for n in budgets):
        raise ValueError('Registry has no declared correction budget')
    expected = protocol()['budgets'][:len(budgets)]
    if budgets != expected:
        raise ValueError('Available budgets must form the predetermined nested prefix')
    if budgets != source['correction_budgets_available']:
        raise ValueError('Every available source budget must finish before selection/testing')
    if method == 'lora' and record.get('completed_budgets') != budgets:
        raise ValueError('LoRA registry is not complete for every available source budget')
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    for n in budgets:
        rows = [c for c in candidates if c['budget'] == n]
        if method == 'lora':
            if len(rows) != 2 or sorted(c['learning_rate'] for c in rows) != [1e-4, 1e-3]:
                raise ValueError('Each LoRA budget requires both fixed learning rates')
        elif len(rows) != 1:
            raise ValueError('Exactly one pooled edit is permitted per budget')
    artifacts = {str(Path(path).resolve()): sha(path)}
    for c in candidates:
        if (c.get('source_manifest_sha256', record['source_manifest_sha256']) != record['source_manifest_sha256']
                or c.get('source_manifest', record['source_manifest']) != record['source_manifest']):
            raise ValueError('A candidate was fitted using a different source manifest')
        artifacts.update(c['artifacts'])
        for key in ('training_path', 'seal_path'):
            if c.get(key):
                artifacts[c[key]] = sha(c[key])
    source = record.get('source_manifest')
    if source:
        digest = sha(source)
        if record.get('source_manifest_sha256', digest) != digest:
            raise ValueError('Source manifest changed after fitting')
        artifacts[source] = digest
    return candidates, artifacts, record


def test_cells(condition, workspace):
    p = protocol(); e = p['evaluation']; start = e['heldout_slot_start']
    layout = str(ROOT / 'design/changed-layouts.json')
    cells = []
    for i in range(2):
        cells.append({'cell': f'original_{i}', 'condition': condition,
                      'task': p['task'], 'goal': p['goal'], 'input_prompt': p['original_prompt'],
                      'effective_prompt': p['clarified_prompt'] if condition == 'clarified' else p['original_prompt'],
                      'workspace': workspace, 'layouts': layout, 'slot_start': start + 16*i, 'n': 16})
    if condition != 'clarified':
        for cell, prompt, task, goal, offset in (
            ('unseen', e['unseen_prompts'][0], p['task'], p['goal'], 32),
            ('reverse', e['competing_prompt'], e['competing_task'], e['competing_goal'], 48),
        ):
            cells.append({'cell': cell, 'condition': condition, 'task': task, 'goal': goal,
                          'input_prompt': prompt, 'effective_prompt': prompt,
                          'workspace': workspace, 'layouts': layout, 'slot_start': start + offset, 'n': 16})
    return cells


def validate_plan(plan):
    p = protocol()
    if (plan.get('cohort_id') != p['cohort_id'] or plan.get('case') != 'temporal'
            or plan.get('inference_batching') != 'fixed16'
            or plan.get('protocol_sha256') != sha(HERE / 'protocol.json')):
        raise ValueError('Evaluation plan identity mismatch')
    batches = plan['batches']
    if not batches or any(b.get('n') != 16 or b['slot_start'] % 16 for b in batches):
        raise ValueError('Evaluation requires whole, aligned fixed16 batches')
    if len({(b['condition'], b['cell']) for b in batches}) != len(batches):
        raise ValueError('Duplicate candidate/cell in a worker plan')
    if plan['split'] == 'development':
        for b in batches:
            if (b['cell'] != 'selection' or b['task'] != p['task'] or b['goal'] != p['goal']
                    or b['input_prompt'] != p['original_prompt'] or b['effective_prompt'] != p['original_prompt']
                    or b['layouts'] != str(ROOT / 'design/development-layouts.json')
                    or b['slot_start'] != p['evaluation']['selection_slot_start'] or not b.get('workspace')):
                raise ValueError('Invalid development selection cell')
    elif plan['split'] == 'heldout':
        for b in batches:
            expected = {c['cell']: c for c in test_cells(b['condition'], b['workspace'])}
            if b != expected.get(b['cell']):
                raise ValueError('Held-out cell differs from the predeclared schedule')
            if b['condition'] in ('base', 'clarified') and b['workspace'] is not None:
                raise ValueError('Unchanged control has edited weights')
    else:
        raise ValueError('Unknown evaluation split')


def seal_schedule(stage, batches, artifacts, metadata):
    dest = ROOT / f'{stage}-frozen'
    if dest.exists():
        raise FileExistsError(dest)
    sources = source_inventory() + [HERE / name for name in ('evaluate.py', 'analyze.py')]
    sources += sorted((ROOT / 'design').glob('*.json'))
    artifacts = {**{str(p.resolve()): sha(p) for p in sources}, **artifacts}
    conditions = list(dict.fromkeys(b['condition'] for b in batches))
    cell_index = {'original_0': 0, 'original_1': 1, 'unseen': 2, 'reverse': 3}
    assignments = [((i//2 + i%2) % 2 if stage == 'selection'
                    else (conditions.index(b['condition']) + cell_index[b['cell']]) % 2)
                   for i, b in enumerate(batches)]
    schedule = {'cohort_id': protocol()['cohort_id'], 'stage': stage, 'batches': batches,
                'gpu_assignment': assignments, **metadata}
    # Every condition's two original-request replicates run on opposite GPUs.
    for gpu in (0, 1):
        plan = {'cohort_id': protocol()['cohort_id'], 'case': 'temporal',
                'split': 'development' if stage == 'selection' else 'heldout',
                'inference_batching': 'fixed16', 'protocol_sha256': sha(HERE / 'protocol.json'),
                'batches': [b for b, assigned in zip(batches, assignments) if assigned == gpu],
                'schedule_sha256': canonical(schedule)}
        if not plan['batches']:
            continue
        validate_plan(plan)
        freeze = dest / f'freeze-g{gpu}.json'
        save(freeze, {'artifacts': artifacts, 'plan_sha256': {'temporal': canonical(plan)}}, exclusive=True)
        save(dest / f'plan-g{gpu}.json', {**plan, 'global_freeze': {'path': str(freeze), 'sha256': sha(freeze)}}, exclusive=True)
    save(dest / 'schedule.json', schedule, exclusive=True)
    return schedule


def prepare_selection():
    p = protocol()
    candidates, artifacts, _ = registry(ROOT / 'lora-adaptation/candidates.json', 'lora')
    batches = []
    for c in sorted(candidates, key=lambda x: (x['budget'], x['learning_rate'])):
        batches.append({'cell': 'selection', 'condition': c['candidate_id'], 'task': p['task'],
                        'goal': p['goal'], 'input_prompt': p['original_prompt'], 'effective_prompt': p['original_prompt'],
                        'workspace': c['workspace'], 'layouts': str(ROOT / 'design/development-layouts.json'),
                        'slot_start': p['evaluation']['selection_slot_start'], 'n': 16})
    return seal_schedule('selection', batches, artifacts, {'candidate_budgets': {c['candidate_id']: c['budget'] for c in candidates}})


def select_candidates(candidates, groups):
    by_budget = defaultdict(list)
    for c in candidates:
        rows = groups[c['candidate_id']]
        if len(rows) != 16 or len({r['slot'] for r in rows}) != 16:
            raise ValueError('Every learning rate needs all 16 selection episodes')
        by_budget[c['budget']].append({**c, 'development_successes': sum(bool(r['success']) for r in rows)})
    return [sorted(values, key=lambda c: (-c['development_successes'], c['learning_rate']))[0]
            for _, values in sorted(by_budget.items())]


def prepare_heldout():
    from analyze import collect_stage
    import preflight
    reset = preflight.verify()
    lora, artifacts, lr_record = registry(ROOT / 'lora-adaptation/candidates.json', 'lora')
    edits, edit_artifacts, edit_record = registry(ROOT / 'edits/candidates.json', 'edit')
    artifacts.update(edit_artifacts)
    artifacts.update(reset['source_identity'])
    for item in reset['starts']:
        artifacts.update(item['evidence_sha256'])
    for path in (HERE / 'preflight.py', ROOT / 'preflight/receipt.json', ROOT / 'preflight/source-freeze.json'):
        artifacts[str(path)] = sha(path)
    if sorted({c['budget'] for c in lora}) != sorted(c['budget'] for c in edits):
        raise ValueError('Methods must use the same available correction budgets')
    if lr_record.get('source_manifest_sha256') != edit_record.get('source_manifest_sha256'):
        raise ValueError('Methods were not constructed from the same source manifest')
    groups, receipts, selection_artifacts = collect_stage('selection')
    selected = select_candidates(lora, {key[0]: value for key, value in groups.items()})
    chosen = {'rule': 'Most development successes; ties choose lower learning rate.',
              'heldout_used': False, 'candidates': selected,
              'all_development_counts': {key[0]: sum(r['success'] for r in rows) for key, rows in groups.items()},
              'development_episodes': sum(r['n'] for r in receipts)}
    save(ROOT / 'selection.json', chosen, exclusive=True)
    artifacts[str(ROOT / 'selection.json')] = sha(ROOT / 'selection.json')
    artifacts.update({k: v for k, v in selection_artifacts.items() if not k.endswith('.npz')})
    candidates = sorted([*edits, *selected], key=lambda c: (c['budget'], c['candidate_id']))
    controls = [('base', None), ('clarified', None)]
    conditions = controls + [(c['candidate_id'], c['workspace']) for c in candidates]
    # Rotate execution order across the four cells; no outcome-dependent ordering.
    all_cells = {condition: test_cells(condition, workspace) for condition, workspace in conditions}
    batches = []
    for index in range(4):
        order = conditions[index:] + conditions[:index]
        if index % 2:
            order = list(reversed(order))
        for condition, _ in order:
            if index < len(all_cells[condition]):
                batches.append(all_cells[condition][index])
    return seal_schedule('heldout', batches, artifacts,
                         {'candidates': candidates, 'available_budgets': sorted(c['budget'] for c in edits),
                          'primary_contrasts': 'Each edit versus base and trajectory LoRA at the same N on changed-layout original instructions; N12 minus N1 when both are available.',
                          'source_manifest': edit_record['source_manifest'],
                          'source_manifest_sha256': edit_record['source_manifest_sha256']})


def worker(plan_path, root, gpu):
    plan = read(plan_path); validate_plan(plan)
    expected = ROOT / ('selection' if plan['split'] == 'development' else 'heldout') / f'g{gpu}'
    if Path(root).resolve() != expected:
        raise ValueError('Evaluation output differs from the declared cohort')
    # config.py reads these variables at import time, before prior.main runs.
    os.environ.update(WSP_POOL_PORT=str(8700 + 10*gpu), WSP_SERVER_GPU=str(gpu),
                      WSP_SIM_GPU=str(gpu), WSP_EVAL_EXTRA_GPUS='', WSP_TASK_DIRS=str(PRIOR / 'runtime'))
    prior = load('scaling_evaluation_worker', PRIOR / 'worker.py')
    prior.verify_global_freeze(plan)
    import weightsmith
    weightsmith.pool = load('weightsmith.pool', POOL)
    sys.argv = [str(HERE / 'evaluate.py'), '--plan', str(plan_path), '--root', str(root), '--gpu', str(gpu)]
    prior.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    sub.add_parser('selection'); sub.add_parser('heldout')
    work = sub.add_parser('worker')
    work.add_argument('--plan', type=Path, required=True)
    work.add_argument('--root', type=Path, required=True)
    work.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    if args.stage == 'selection': prepare_selection()
    elif args.stage == 'heldout': prepare_heldout()
    else: worker(args.plan, args.root, args.gpu)


if __name__ == '__main__':
    main()
