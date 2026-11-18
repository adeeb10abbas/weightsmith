"""Inventory the interrupted predecessor without modifying or pooling its evidence."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = REPO / 'projects/rss-goal-expansion-20261117'
    dest = REPO / 'projects/rss-goal-expansion-v2-20261117/prior-v1'
    output = dest / 'inventory.json'
    if output.exists():
        raise FileExistsError(output)
    evidence = {}

    def retain(path):
        path = Path(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        target = dest / 'files' / sha / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Evidence copy changed')
        target.write_bytes(raw)
        evidence[str(path)] = {'sha256': sha, 'bytes': len(raw), 'copy': str(target)}
        return json.loads(raw) if path.suffix == '.json' else raw.decode(errors='replace')

    receipts = sorted(old.glob('qualification/*/*/batch-*/completed.json'))
    if len(receipts) != 2 or any('/cube_behind/initial/' not in str(p) for p in receipts):
        raise ValueError('Predecessor changed: review additional evidence explicitly')
    completed = []
    for path in receipts:
        receipt = retain(path)
        episodes_path = path.with_name('episodes.json')
        episodes = retain(episodes_path)
        if digest(episodes_path) != receipt['episodes_sha256'] or len(episodes) != 16:
            raise ValueError('Invalid predecessor receipt')
        if sum(bool(r['success']) for r in episodes) != receipt['successes']:
            raise ValueError('Invalid predecessor outcomes')
        for row in episodes:
            retain(row['metrics'])
        for log in path.parent.glob('*.log'):
            retain(log)
        completed.append({**receipt, 'receipt_path': str(path),
                          'recorded_policy_decisions': sum(r['decision_count'] for r in episodes)})
    failed = old / 'qualification/glasses_pickup/initial'
    state = retain(failed / 'status.json')
    batch = failed / 'batch-00-original-base'
    logs = sorted(batch.glob('sim-*.log'))
    if len(logs) != 4 or list(batch.glob('obs-*/*.npz')) or (batch / 'completed.json').exists():
        raise ValueError('Failed-stage evidence changed; do not assume zero policy exposure')
    attempts = []
    for i, path in enumerate(logs):
        text = retain(path)
        reset_path = batch / f'obs-l0a{i}/layout_r0_reset1.json'
        reset = retain(reset_path) if reset_path.exists() else None
        elapsed = [int(x.replace(',', '')) / 1000 for x in re.findall(r'\[([\d,]+)ms\]', text)]
        attempts.append({'log': str(path), 'reset_receipt': str(reset_path) if reset else None,
            'status': 'invalid reset before policy' if reset else 'interrupted during scene initialization',
            'recorded_policy_observation_files': 0,
            'log_observed_simulator_elapsed_seconds': max(elapsed) if elapsed else None,
            'elapsed_limit': 'Last printed simulator time, not exact process duration or GPU compute time.'})
    retain(batch / 'server-l0.log')
    for p in (old / 'qualification-source-freeze.json', old / 'qualification-status.json',
              old / 'integration-interruption-20261117.json',
              dest / 'preservation.json', dest / 'services-before-stop.txt',
              dest / 'qualification-journal.jsonl', dest / 'qualification-journal-summary.json'):
        retain(p)
    journal = json.loads((dest / 'qualification-journal-summary.json').read_text())
    stop = [r['unix_seconds'] for r in journal if r['message'].startswith('Stopped ')][-1]
    cpu = [int(r['cpu_usage_nsec']) / 1e9 for r in journal if r['cpu_usage_nsec'] is not None][-1]
    record = {'schema': 'weightsmith-interrupted-goal-cohort-v1',
        'cohort': 'rss-goal-expansion-20261117', 'source_commit': '16c87cb',
        'disposition': 'Aborted setup cohort; all policy qualification must repeat in successor. Never pooled.',
        'counts': {'completed_policy_episodes': 32, 'completed_policy_batches': 2,
            'recorded_policy_decisions': sum(x['recorded_policy_decisions'] for x in completed),
            'failed_or_interrupted_simulator_starts': 4, 'completed_glasses_policy_episodes': 0,
            'repeat_qualification_episodes': 0, 'qualified_edits': 0, 'heldout_episodes': 0},
        'completed': completed, 'pre_policy_attempts': attempts,
        'exposure': 'Operator observed cube-behind original2/16 and teacher15/16, and glasses invalid-reset diagnostics. The glasses server loaded, but all attempts stopped before the first infer_batch call; no recorded action/observation trajectories. Server initialization is not policy evaluation.',
        'costs': {'completed_execution_seconds': sum(x['execution_seconds'] for x in completed),
            'completed_queue_seconds': sum(x['queue_seconds'] for x in completed),
            'failed_stage_queue_seconds': state['queue_seconds'],
            'failed_stage_execution_to_service_stopped_seconds': stop - state['execution_started_at'],
            'failed_stage_execution_start_unix': state['execution_started_at'],
            'service_stopped_unix': stop, 'qualification_service_cpu_seconds': cpu,
            'limits': 'Failed-stage interval includes server startup, all simulator retries and cleanup, bounded by systemd stop completion. It is wall time, not GPU compute or per-attempt exact timing. CPU is whole qualification service, overlapping other listed costs; do not sum CPU with wall time. Operator/source engineering unmetered.'},
        'evidence': evidence}
    if list(old.glob('*adaptation*')) or list(old.glob('*heldout*')):
        raise ValueError('Unexpected downstream predecessor execution')
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'sha256': digest(output), 'counts': record['counts'], 'costs': record['costs']}))


if __name__ == '__main__':
    main()
