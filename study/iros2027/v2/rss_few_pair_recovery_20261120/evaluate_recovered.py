"""Use the certified punctuation fix without changing the frozen first runtime.

The original evaluator's schedule, source selection, scoring, and statistical
rules remain authoritative. Only baseline artifact loading and the runtime
identity of newly created schedules are extended. Existing temporal selection
stays sealed under its original worker and candidate artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from recovery import (HERE, STUDY, RECROOT, AMENDMENT, goal_root, protocol,
                      read, sha, verify, worker_commands as original_worker_commands)
import evaluate as E

V2 = HERE / 'baselines_v2'
GATE = RECROOT / 'baseline-v2-qualification/passed.json'
BASELINE_AMENDMENT = RECROOT / 'design/baseline-runtime-v2-amendment.json'
WORKER = HERE / 'worker_recovered.py'

ORIGINAL_CANDIDATE = getattr(E, '_fewpair_original_candidate_identity', E.candidate_identity)
ORIGINAL_SEAL = getattr(E, '_fewpair_original_seal_schedule', E.seal_schedule)


def artifact_api():
    name = '_fewpair_recovered_baseline_artifacts'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, V2 / 'artifacts.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def candidate_identity(candidate, goal_id):
    """Keep the legacy candidate record shape for exact selection replay."""
    workspace = Path(candidate['workspace']).resolve()
    if not (workspace / 'baseline.json').is_file():
        return ORIGINAL_CANDIDATE(candidate, goal_id)
    if not workspace.is_relative_to(goal_root(goal_id) / 'adaptation'):
        raise ValueError('Baseline candidate is outside this registered goal')
    result = dict(candidate)
    spec, _ = artifact_api().load_artifact(workspace, verify_sources=True)
    expected = 'embedding_mean' if result['method'] == 'embedding' else 'residual_cells'
    if result['method'] not in ('embedding', 'residual') or spec['method'] != expected:
        raise ValueError('Baseline method differs from the registered candidate')
    if spec['strength'] != result['strength']:
        raise ValueError('Baseline strength differs from the registered candidate')
    if expected == 'residual_cells' and spec['layer'] != protocol()['baselines']['residual_layer']:
        raise ValueError('Baseline uses an undeclared residual layer')
    verify(result.get('artifacts', {}))
    result['workspace'] = str(workspace)
    result['fingerprint'] = sha(workspace / 'baseline.json')
    result['artifacts'] = {**result.get('artifacts', {}),
                           **{str(p): sha(p) for p in (workspace / 'baseline.json', workspace / 'direction.npy')}}
    return result


def runtime_artifacts():
    gate = read(GATE)
    if (gate.get('status') != 'pass' or gate.get('new_policy_episodes') != 0
            or gate.get('kernel_batch') != 16 or not gate.get('artifacts')):
        raise ValueError('The native V2 compatibility qualification has not passed')
    parity = gate.get('legacy_candidate_max_abs_errors', {})
    expected_prompts = {(gid, cell) for gid in protocol()['goal_ids']
                        for cell in ('original', 'teacher', 'unseen', 'competing')}
    actual_prompts = {(row.get('goal_id'), row.get('cell'))
                      for row in gate.get('registered_prompt_cases', [])}
    if (gate.get('legacy_exact_parity') is not True or len(parity) != 12
            or any(value != 0.0 for value in parity.values())
            or actual_prompts != expected_prompts):
        raise ValueError('The V2 gate does not certify every legacy candidate and registered prompt')
    errors = gate.get('max_abs_physical_action_errors', {})
    for name in ('fixed16_native_wrapper', 'embedding_zero', 'residual_zero',
                 'restore_native', 'clarified_embedding_replacement'):
        error = errors.get(name, float('inf'))
        if not 0 <= error <= 1e-6:
            raise ValueError('The native V2 control did not pass: ' + name)
    verify(gate['artifacts'])
    amendment = read(BASELINE_AMENDMENT)
    if amendment.get('status') != 'authorized_technical_recovery_before_heldout':
        raise ValueError('The punctuation recovery has no frozen amendment')
    verify(amendment.get('preserved_failure_artifacts', {}))
    verify(amendment.get('original_runtime_artifacts', {}))
    files = [Path(__file__).resolve(), WORKER, GATE, AMENDMENT, BASELINE_AMENDMENT,
             *sorted(V2.glob('*.py'))]
    if not (V2 / 'batched_server.py').is_file():
        raise ValueError('The V2 policy server entry point is missing')
    return {**gate['artifacts'], **amendment.get('preserved_failure_artifacts', {}),
            **amendment.get('original_runtime_artifacts', {}),
            **{str(path.resolve()): sha(path) for path in files}}


def seal_schedule(goal_id, stage, batches, artifacts, metadata):
    destination = goal_root(goal_id) / f'{stage}-frozen/schedule.json'
    if destination.exists():
        # A running or completed V1 temporal development schedule remains exact.
        existing = read(destination)
        if not existing.get('worker_entrypoint'):
            if goal_id != 'temporal' or stage != 'selection':
                raise ValueError('An unexpected legacy schedule exists in the recovered phases')
        elif existing['worker_entrypoint'] != str(WORKER):
            raise ValueError('A frozen schedule names another recovery worker')
        return ORIGINAL_SEAL(goal_id, stage, batches, artifacts, metadata)
    extra = runtime_artifacts()
    return ORIGINAL_SEAL(goal_id, stage, batches, {**artifacts, **extra}, {
        **metadata, 'worker_entrypoint': str(WORKER), 'baseline_runtime': 'punctuation-v2',
        'baseline_runtime_gate': str(GATE), 'baseline_runtime_gate_sha256': sha(GATE),
        'baseline_recovery_amendment': str(BASELINE_AMENDMENT),
        'baseline_recovery_amendment_sha256': sha(BASELINE_AMENDMENT)})


def install():
    """Install two narrow in-process hooks; no original file is modified."""
    E._fewpair_original_candidate_identity = ORIGINAL_CANDIDATE
    E._fewpair_original_seal_schedule = ORIGINAL_SEAL
    E.candidate_identity = candidate_identity
    E.seal_schedule = seal_schedule
    return E


def worker_commands(goal_id, stage):
    schedule = read(goal_root(goal_id) / f'{stage}-frozen/schedule.json')
    entrypoint = schedule.get('worker_entrypoint')
    commands = original_worker_commands(goal_id, stage)
    if entrypoint is None:
        if (goal_id, stage) != ('temporal', 'selection'):
            raise ValueError('Only the existing temporal selection may use the legacy worker')
        return commands
    if entrypoint != str(WORKER):
        raise ValueError('The schedule names a different worker')
    for command in commands:
        if command[1] != str(STUDY / 'worker.py'):
            raise ValueError('Unexpected original worker command')
        command[1] = str(WORKER)
    return commands


def analyze(goal_id=None):
    install()
    import analyze as A
    if goal_id is None:
        reports = [A.analyze_goal(gid) for gid in protocol()['goal_ids']]
        for report in reports:
            A.write_report(goal_root(report['goal_id']) / 'analysis-v1', report)
        report = A.aggregate_reports(reports)
        output = A.ROOT / 'analysis-v1'
    else:
        report = A.analyze_goal(goal_id)
        output = goal_root(goal_id) / 'analysis-v1'
    A.write_report(output, report)
    print(json.dumps({'report': str(output / 'report.json'), 'status': report['status']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selection', 'heldout', 'freeze-all-heldout', 'analyze', 'aggregate'))
    parser.add_argument('--goal', choices=protocol()['goal_ids'])
    args = parser.parse_args()
    evaluator = install()
    if args.stage == 'freeze-all-heldout':
        evaluator.freeze_all_heldout()
    elif args.stage == 'aggregate':
        analyze()
    elif args.goal is None:
        parser.error('--goal is required for this stage')
    elif args.stage == 'selection':
        evaluator.prepare_selection(args.goal)
    elif args.stage == 'heldout':
        evaluator.prepare_heldout(args.goal)
    else:
        analyze(args.goal)


if __name__ == '__main__':
    main()
