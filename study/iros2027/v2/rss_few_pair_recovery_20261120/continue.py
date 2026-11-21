"""Finish the original comparison after the explicitly recorded recovery.

Every registered goal must have a verified disposition before the global test
freeze. Completed trials can be verified again; partial workers are never
silently replayed, and a technically blocked goal is never silently dropped.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import signal
import sys
import time

from recovery import (HERE, STUDY, OLDROOT, RECROOT, AMENDMENT, amendment,
                      command, goal_root, protocol, read, save, sha, verify,
                      run_commands, stop_on_signal)


def recovered_command(stage, *args):
    return [sys.executable, str(HERE / 'evaluate_recovered.py'), stage, *map(str, args)]


def validate_ready_record(goal_id, path):
    """Bind readiness to the exact source pool, recovery, and completed evidence."""
    path = Path(path)
    ready = read(path)
    source_path = goal_root(goal_id) / 'source-status.json'
    source = read(source_path)
    if ready.get('goal_id') != goal_id or ready.get('heldout_policy_episodes') != 0:
        raise ValueError('Readiness belongs to another goal or already used held-out trials')
    if ready.get('status') not in ('ready', 'unqualified_at_cap'):
        raise ValueError('Goal readiness is not a completed disposition')
    expected_source_status = 'qualified' if ready['status'] == 'ready' else 'unqualified_at_cap'
    if source.get('status') != expected_source_status:
        raise ValueError('Readiness disagrees with source qualification')
    manifest = Path(source['manifest']).resolve()
    if not manifest.is_relative_to(goal_root(goal_id) / 'sources'):
        raise ValueError('Readiness identifies a foreign source pool')
    if (ready.get('source_manifest') != str(manifest)
            or ready.get('source_manifest_sha256') != source['manifest_sha256']
            or sha(manifest) != source['manifest_sha256']):
        raise ValueError('Source pool changed before the held-out freeze')
    artifacts = ready.get('artifacts', {})
    if (artifacts.get(str(AMENDMENT)) != sha(AMENDMENT)
            or artifacts.get(str(manifest)) != sha(manifest)):
        raise ValueError('Readiness must bind the source manifest and recovery amendment')
    if ready['status'] == 'ready' and ready.get('development_episodes') != 208:
        raise ValueError('All 208 development-selection episodes are required')
    verify(artifacts)
    return {**ready, 'ready_path': str(path.resolve()), 'ready_sha256': sha(path),
            'source_status_path': str(source_path), 'source_status_sha256': sha(source_path)}


def verify_full_ready(goal_id, path):
    ready = validate_ready_record(goal_id, path)
    import evaluate_recovered
    evaluate_recovered.install()
    from analyze import source_accounting, collect_stage
    source, _, _ = source_accounting(goal_id)
    if ready['status'] == 'unqualified_at_cap':
        if source['qualified'] or source['attempted_pairs'] != 48:
            raise ValueError('Failed qualification must be the registered full-cap outcome')
        return ready
    if not source['qualified']:
        raise ValueError('Qualified readiness has no qualified source manifest')
    from evaluate import registries, reset_artifacts
    registries(goal_id)
    reset_artifacts(goal_id, 'development')
    reset_artifacts(goal_id, 'heldout')
    _, receipts, _ = collect_stage(goal_id, 'selection')
    if len(receipts) != 13 or sum(r['n'] for r in receipts) != 208:
        raise ValueError('Incomplete verified development selection')
    return ready


def readiness_paths():
    """Do not mistake an absent technical result for failed qualification."""
    paths, statuses = {}, {}
    for goal_id in protocol()['goal_ids']:
        directory = RECROOT / goal_id
        for filename in ('prepare-status.json', 'collection-status.json'):
            path = directory / filename
            if path.exists():
                status = read(path)
                if status.get('status') in ('stopped', 'failed'):
                    raise RuntimeError(f'{goal_id} stopped during recovery: {path}: {status.get("reason", "")}')
        path = directory / 'ready-for-heldout.json'
        if path.exists():
            paths[goal_id] = path
            statuses[goal_id] = read(path).get('status')
        else:
            statuses[goal_id] = 'pending'
    return paths, statuses


def wait_for_readiness(timeout_seconds, poll_seconds=10, update=None):
    deadline = time.monotonic() + timeout_seconds
    while True:
        paths, statuses = readiness_paths()
        if update:
            update(status='waiting_for_readiness', stage='development_selection', goals=statuses)
        if set(paths) == set(protocol()['goal_ids']):
            return {goal_id: verify_full_ready(goal_id, paths[goal_id]) for goal_id in protocol()['goal_ids']}
        if time.monotonic() >= deadline:
            raise TimeoutError('Every registered goal must finish preparation or capped qualification; no goal was dropped')
        time.sleep(poll_seconds)


def verified_global_freeze(readiness):
    path = OLDROOT / 'all-heldout-frozen.json'
    frozen = read(path)
    if (frozen.get('status') != 'frozen' or frozen.get('cohort_id') != protocol()['cohort_id']
            or set(frozen['goals']) != set(protocol()['goal_ids'])):
        raise ValueError('The global freeze does not cover every registered goal')
    verify(frozen['artifacts'])
    qualified = []
    for goal_id in protocol()['goal_ids']:
        expected = readiness[goal_id]['status'] == 'ready'
        if frozen['goals'][goal_id]['qualified'] is not expected:
            raise ValueError('The frozen goal disposition differs from verified readiness')
        if expected:
            qualified.append(goal_id)
    if frozen['planned_heldout_episodes'] != 544 * len(qualified):
        raise ValueError('The global frozen trial count changed')
    return path, qualified


def phase(stage, commands, update, timeout):
    """Skip only a completed process phase; never overwrite a failed attempt."""
    directory = RECROOT / 'continuation' / stage
    receipt = directory / 'completed.json'
    if receipt.exists():
        previous = read(receipt)
        if previous.get('status') != 'complete' or previous.get('commands') != commands:
            raise ValueError('A previous phase receipt belongs to different commands')
        return
    if directory.exists():
        raise ValueError('Partial continuation phase retained: ' + str(directory))
    update(status='running', stage=stage, commands=commands)
    run_commands(commands, directory, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-timeout-seconds', type=int, default=10800)
    parser.add_argument('--heldout-timeout-seconds', type=int, default=43200)
    args = parser.parse_args()
    if not 1 <= args.ready_timeout_seconds <= 21600:
        parser.error('Readiness timeout must be between 1 second and 6 hours')
    if not 1 <= args.heldout_timeout_seconds <= 86400:
        parser.error('Held-out timeout must be between 1 second and 24 hours')
    amendment()
    import evaluate_recovered
    evaluate_recovered.install()
    RECROOT.mkdir(parents=True, exist_ok=True)
    lock = (RECROOT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    state_path = RECROOT / 'supervisor-status.json'
    if state_path.exists():
        previous = read(state_path)
        if previous.get('status') == 'complete_with_verified_report':
            if sha(previous['report']) != previous['report_sha256']:
                raise ValueError('Completed recovery report changed')
            print({'status': 'already_complete', 'report': previous['report']}, flush=True)
            return
        raise ValueError('A previous supervisor attempt is preserved; record any further recovery separately')
    own_artifacts = {str(Path(__file__).resolve()): sha(__file__), str(AMENDMENT): sha(AMENDMENT),
                     str(HERE / 'recovery.py'): sha(HERE / 'recovery.py'),
                     str(HERE / 'evaluate_recovered.py'): sha(HERE / 'evaluate_recovered.py'),
                     str(HERE / 'worker_recovered.py'): sha(HERE / 'worker_recovered.py')}
    state = {'status': 'starting', 'started_unix': time.time(), 'artifacts': own_artifacts,
             'ready_timeout_seconds': args.ready_timeout_seconds,
             'heldout_timeout_seconds': args.heldout_timeout_seconds}

    def update(**values):
        state.update(values, updated_unix=time.time())
        save(state_path, state)

    try:
        update(stage='checking_recovery_amendment')
        readiness = wait_for_readiness(args.ready_timeout_seconds, update=update)
        verify(own_artifacts)
        readiness_artifacts = {
            record['ready_path']: record['ready_sha256'] for record in readiness.values()}
        readiness_artifacts.update({record['source_status_path']: record['source_status_sha256']
                                    for record in readiness.values()})
        save(RECROOT / 'design/all-goals-ready.json', {
            'status': 'verified', 'goals': readiness, 'artifacts': {**own_artifacts, **readiness_artifacts},
            'heldout_policy_episodes': 0}, exclusive=True)
        phase('freeze-all-heldout', [recovered_command('freeze-all-heldout')], update, 3600)
        freeze_path, qualified = verified_global_freeze(readiness)
        save(RECROOT / 'design/heldout-release.json', {
            'status': 'released', 'qualified_goals': qualified,
            'global_freeze': str(freeze_path), 'global_freeze_sha256': sha(freeze_path),
            'artifacts': {**own_artifacts, **readiness_artifacts, str(freeze_path): sha(freeze_path)}}, exclusive=True)
        update(qualified_goals=qualified, planned_heldout_episodes=544 * len(qualified))
        deadline = time.monotonic() + args.heldout_timeout_seconds
        for goal_id in qualified:
            verify(own_artifacts)
            verify(readiness_artifacts)
            verified_global_freeze(readiness)
            commands = evaluate_recovered.worker_commands(goal_id, 'heldout')
            if commands:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Held-out execution deadline reached; no partial trial was replayed')
                phase(goal_id + '-heldout', commands, update, remaining)
            # Fresh analysis also verifies any previously completed workers.
            phase(goal_id + '-analysis', [recovered_command('analyze', '--goal', goal_id)], update, 3600)
        phase('aggregate-analysis', [recovered_command('aggregate')], update, 3600)
        report = OLDROOT / 'analysis-v1/report.json'
        result = read(report)
        if (result.get('status') != 'complete_and_verified'
                or result.get('registered_goals') != len(protocol()['goal_ids'])
                or result.get('new_heldout_policy_episodes') != 544 * len(qualified)
                or result.get('new_selection_policy_episodes') != 208 * len(qualified)):
            raise ValueError('Final report does not cover the entire registered comparison')
        verify(own_artifacts)
        update(status='complete_with_verified_report', stage='complete', report=str(report),
               report_sha256=sha(report), completed_unix=time.time())
        print({'status': state['status'], 'report': str(report)}, flush=True)
    except BaseException as exc:
        update(status='stopped', reason=repr(exc))
        raise


if __name__ == '__main__':
    main()
