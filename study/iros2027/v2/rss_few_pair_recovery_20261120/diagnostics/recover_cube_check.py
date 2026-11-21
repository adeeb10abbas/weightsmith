"""One authorized, separately accounted retry of the auxiliary cube matrix."""
from __future__ import annotations

import fcntl
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent
sys.path.insert(0, str(RECOVERY))
from recovery import (RECROOT, STUDY, amendment, goal_root, protocol, read, save, sha,
                      verify, run_commands, stop_on_signal)
from common import freeze_plan

CANONICAL = RECROOT / 'development-edit-check/cube_behind'
OUT = CANONICAL / 'recovery-v3'
OLDOWNER = goal_root('cube_behind') / 'development-edit-check/g1'
OWNER = goal_root('cube_behind') / 'development-edit-check-recovery-v3/g1'
WORKER = HERE / 'worker_audited.py'
AMENDMENT = OUT / 'amendment.json'
CONDITIONS = ['base', 'weightsmith_g0', 'weightsmith_g1', 'weightsmith_g2']


def validate_failed_attempt(old_owner, old_plan, controller_status):
    state = read(old_owner / 'status.json')
    if state['status'] != 'stopped' or state['completed_episodes'] != 0 or state['batch'] != 0:
        raise ValueError('Recovery is limited to the recorded first-batch technical stop')
    if 'Source/candidate changed during batch' not in state['reason']:
        raise ValueError('Different failure requires a separate recovery')
    if controller_status['status'] != 'stopped':
        raise ValueError('The original controller is not stopped')
    if [b['condition'] for b in old_plan['batches']] != CONDITIONS:
        raise ValueError('Original four-condition matrix differs')
    if any(b['n'] != 16 for b in old_plan['batches']):
        raise ValueError('Original batch size differs')
    batches = sorted(p for p in old_owner.glob('batch-*') if p.is_dir())
    if len(batches) != 1 or batches[0].name != 'batch-00-selection-base':
        raise ValueError('Expected exactly one prior technical batch')
    batch = batches[0]
    if (batch / 'completed.json').exists() or (batch / 'episodes.json').exists():
        raise ValueError('Prior trials must remain unreceipted and unpromoted')
    rows = read(batch / 'raw-rows.json')
    spec = old_plan['batches'][0]
    if len(rows) != 16 or sorted(r['slot'] for r in rows) != list(range(spec['slot_start'], spec['slot_start'] + 16)):
        raise ValueError('Prior technical-attempt accounting differs from 16 trials')
    return 16


def snapshot_failure(old_owner, canonical, destination):
    # Enumerate first: the destination will be a new child of canonical.
    sources = [(p, 'worker' / Path(p.relative_to(old_owner)))
               for p in sorted(old_owner.rglob('*')) if p.is_file()]
    sources += [(p, 'controller' / Path(p.relative_to(canonical)))
                for p in sorted(canonical.rglob('*')) if p.is_file()
                and not p.is_relative_to(destination.parent)]
    destination.mkdir(parents=True, exist_ok=False)
    mapping, original_pins, snapshot_pins = [], {}, {}
    for original, relative in sources:
        checksum = sha(original)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha(target) != checksum or sha(original) != checksum:
            raise ValueError('Failure evidence changed while being preserved: ' + str(original))
        target.chmod(0o444)
        original_pins[str(original)] = checksum
        snapshot_pins[str(target)] = checksum
        mapping.append({'original': str(original), 'snapshot': str(target),
                        'sha256': checksum, 'bytes': target.stat().st_size})
    record = {'status': 'preserved_and_verified', 'created_unix': time.time(),
              'new_policy_episodes': 0, 'files': mapping,
              'original_artifacts': original_pins, 'artifacts': snapshot_pins}
    path = destination / 'inventory.json'
    save(path, record, exclusive=True)
    path.chmod(0o444)
    for directory in sorted((p for p in destination.rglob('*') if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    destination.chmod(0o555)
    return path, record


def main():
    amendment()
    if OUT.exists() or OWNER.exists() or (CANONICAL / 'report.json').exists():
        raise ValueError('Recovery is single-attempt; existing state requires separate review')
    old_lock = (OLDOWNER / 'worker.lock').open('r')
    fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller_lock = (CANONICAL / 'check.lock').open('r')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_plan = read(CANONICAL / 'plan.json')
    prior_status = read(CANONICAL / 'status.json')
    quarantined = validate_failed_attempt(OLDOWNER, old_plan, prior_status)
    identity = read(OLDOWNER / 'identity.json')
    verify(identity)
    from worker import verify_global_freeze
    verify_global_freeze(old_plan)
    snapshot_path, snapshot = snapshot_failure(OLDOWNER, CANONICAL, OUT / 'failure-snapshot')
    frozen_prior = {p: h for p, h in snapshot['original_artifacts'].items()
                    if Path(p).is_relative_to(OLDOWNER)}
    amendment_record = {
        'status': 'authorized_auxiliary_technical_recovery', 'created_unix': time.time(),
        'cohort_id': protocol()['cohort_id'], 'goal_id': 'cube_behind',
        'reason': 'Original batch stopped at an unlocalized post-run hash discrepancy; all pins now match, but the failed guard cannot be cleared retrospectively.',
        'quarantined_policy_episodes': quarantined,
        'quarantine_rule': 'Preserve all original raw outcomes; never promote, pool, or count them as confirmed comparisons.',
        'new_policy_episode_cap': 64, 'auxiliary_policy_episode_cap': 144,
        'original_confirmatory_policy_episode_cap': 2544, 'combined_policy_episode_cap': 2688,
        'conditions': CONDITIONS, 'unchanged': ['all four conditions', 'weights and source groups',
            'prompts', 'layouts and noise seeds', 'fixed16 inference', 'scoring', 'no selection or tuning'],
        'worker_change': 'Original worker with import-root routing and complete per-path integrity diagnostics; discrepancy stops execution without a receipt.',
        'failure_snapshot': str(snapshot_path), 'old_owner': str(OLDOWNER), 'new_owner': str(OWNER),
        'original_controller_status': prior_status,
        'artifacts': {**frozen_prior, str(snapshot_path): sha(snapshot_path),
            str(CANONICAL / 'plan.json'): sha(CANONICAL / 'plan.json'),
            str(CANONICAL / 'plan-freeze.json'): sha(CANONICAL / 'plan-freeze.json'),
            str(WORKER): sha(WORKER), str(Path(__file__).resolve()): sha(__file__)}}
    save(AMENDMENT, amendment_record, exclusive=True)
    AMENDMENT.chmod(0o444)
    plan = {k: v for k, v in old_plan.items() if k != 'global_freeze'}
    plan['audit_root'] = str(OWNER / 'source-checks')
    plan['recovery'] = {'amendment': str(AMENDMENT), 'amendment_sha256': sha(AMENDMENT),
                        'quarantined_policy_episodes': quarantined, 'attempt': 'v3'}
    artifacts = {**identity, **amendment_record['artifacts'],
                 **snapshot['artifacts'], str(AMENDMENT): sha(AMENDMENT)}
    path = freeze_plan(OUT / 'plan.json', plan, artifacts)
    if read(path)['batches'] != old_plan['batches'] or read(path)['candidates'] != old_plan['candidates']:
        raise ValueError('Recovery changed the original matrix')
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    state = {'status': 'running', 'goal_id': 'cube_behind', 'attempt': 'v3',
             'started_unix': time.time(), 'stage': 'running_development_policy_trials',
             'quarantined_policy_episodes': quarantined, 'owner': str(OWNER),
             'amendment': str(AMENDMENT), 'prior_failure_snapshot': str(snapshot_path)}
    def update(**changes):
        state.update(changes, updated_unix=time.time())
        save(OUT / 'status.json', state)
        save(CANONICAL / 'status.json', state)
    try:
        update()
        run_commands([[sys.executable, str(WORKER), '--plan', str(path), '--root', str(OWNER), '--gpu', '1']],
                     OUT / 'execution', timeout=10800)
        from diagnose_edits import verify_batches
        report = verify_batches('cube_behind', read(path), OWNER)
        verify(frozen_prior)
        verify(snapshot['artifacts'])
        report['recovery'] = {'attempt': 'v3', 'amendment': str(AMENDMENT),
            'quarantined_policy_episodes': quarantined, 'quarantined_trials_in_analysis': False,
            'original_failed_owner': str(OLDOWNER), 'verified_owner': str(OWNER)}
        report['artifacts'].update({str(AMENDMENT): sha(AMENDMENT), str(path): sha(path),
                                   str(snapshot_path): sha(snapshot_path)})
        save(OUT / 'report.json', report, exclusive=True)
        save(CANONICAL / 'report.json', report, exclusive=True)
        update(status='complete', stage='complete', report=str(CANONICAL / 'report.json'))
    except BaseException as exc:
        update(status='stopped', reason=repr(exc))
        raise


if __name__ == '__main__':
    main()
