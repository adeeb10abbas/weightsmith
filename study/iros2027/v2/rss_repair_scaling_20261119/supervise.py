"""Run the bounded source expansion, fresh adaptations, selection, and frozen tests."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import HERE, REPO, ROOT, protocol, read, save, sha


def completed(path, status_values):
    return Path(path).is_file() and read(path).get('status') in status_values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay-gate', type=Path, default=ROOT / 'replay-gate/passed.json')
    ap.add_argument('--reset-gate', type=Path, default=ROOT / 'preflight/receipt.json')
    ap.add_argument('--gate-timeout-seconds', type=int, default=3600)
    args = ap.parse_args()
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children = []
    state = {'status': 'starting', 'started_unix': time.time(), 'budgets': protocol()['budgets'],
             'maximum_source_episodes': 96, 'maximum_selection_episodes': 128,
             'maximum_heldout_episodes': 608, 'code_sha256': sha(__file__)}

    def update(**values):
        state.update(values, updated_unix=time.time())
        save(ROOT / 'supervisor-status.json', state)

    def stop(signum, frame):
        raise InterruptedError(f'User/service signal {signum}; partial artifacts remain preserved')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def run_many(stage, commands):
        nonlocal children
        update(status='running', stage=stage, commands=commands)
        streams = []; children = []
        try:
            for i, command in enumerate(commands):
                stream = (ROOT / f'supervisor-{stage}-{i}.log').open('a')
                streams.append(stream)
                env = {**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
                children.append(subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                                 env=env, start_new_session=True))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError(f'{stage} failed; no automatic replay of incomplete stages')
                time.sleep(5)
            if any(child.returncode for child in children):
                raise RuntimeError(f'{stage} failed')
        finally:
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for child in children:
                if child.poll() is None:
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
            for child in children:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            for stream in streams:
                stream.close()
            children = []

    def gate(path, stage):
        update(status='waiting_for_validated_gate', stage=stage, gate=str(path))
        deadline = time.monotonic() + args.gate_timeout_seconds
        while not path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f'{stage} was not certified before the bounded wait ended')
            if completed(ROOT / 'collection-status.json', {'stopped'}):
                raise RuntimeError('Source collection stopped; inspect preserved failure')
            time.sleep(10)
        record = read(path)
        if record.get('status') not in ('passed', 'complete_and_verified'):
            raise ValueError(f'{stage} did not pass: {path}')
        for artifact, digest in record.get('artifacts', {}).items():
            if sha(artifact) != digest:
                raise ValueError(f'{stage} certificate artifact changed: {artifact}')
        return record

    def run_workers(stage):
        commands = []
        for gpu in (0, 1):
            plan = ROOT / f'{stage}-frozen/plan-g{gpu}.json'
            if not plan.exists():
                continue
            output = ROOT / stage / f'g{gpu}'
            if completed(output / 'status.json', {'complete'}):
                continue
            commands.append([sys.executable, str(HERE / 'evaluate.py'), 'worker', '--plan', str(plan),
                             '--root', str(output), '--gpu', str(gpu)])
        if commands:
            run_many(stage, commands)

    try:
        update(status='waiting_for_source_collection', stage='initial-source-block')
        deadline = time.monotonic() + args.gate_timeout_seconds
        while not completed(ROOT / 'collection-status.json', {'complete'}):
            if completed(ROOT / 'collection-status.json', {'stopped'}):
                raise RuntimeError('Initial collection stopped; inspect preserved artifacts')
            if time.monotonic() > deadline:
                raise TimeoutError('Initial collection did not complete before bounded wait')
            time.sleep(10)
        reset = gate(args.reset_gate, 'changed-arrangement-reset')
        if reset.get('layouts_sha256') != sha(ROOT / 'design/changed-layouts.json'):
            raise ValueError('Reset certificate does not cover the frozen changed arrangements')
        source_status = read(ROOT / 'source-status.json')
        if source_status['eligible_pairs'] == 0:
            run_many('source-expansion-zero-initial', [[sys.executable, str(HERE / 'collect.py'), 'run',
                                                       '--gpus', '0', '1', '--max-blocks', '3']])
            source_status = read(ROOT / 'source-status.json')
            if source_status['eligible_pairs'] == 0:
                update(status='complete_without_eligible_corrections', stage='source-qualification',
                       source_counts=source_status)
                return
        replay = gate(args.replay_gate, 'fresh-native-replay')
        certified = Path(replay.get('source_manifest', ''))
        if (not certified.is_file() or not certified.resolve().is_relative_to(ROOT / 'sources')
                or replay.get('source_manifest_sha256') != sha(certified)):
            raise ValueError('Replay certificate belongs to a different source cohort')
        # No acquisition based on held-out performance. Preserve first eligible
        # pairs and stop at 12 usable sources or the fixed 48-pair cap.
        source_status = read(ROOT / 'source-status.json')
        if source_status['eligible_pairs'] < 12 and source_status['attempted_pairs'] < 48:
            run_many('source-expansion', [[sys.executable, str(HERE / 'collect.py'), 'run',
                                          '--gpus', '0', '1', '--max-blocks', '3']])
        source_status = read(ROOT / 'source-status.json')
        source_manifest = Path(source_status['manifest'])
        source = read(source_manifest)
        if not source['correction_budgets_available']:
            update(status='complete_without_eligible_corrections', stage='source-qualification',
                   source_manifest=str(source_manifest), source_counts=source_status)
            return
        update(source_manifest=str(source_manifest), source_manifest_sha256=sha(source_manifest),
               available_budgets=source['correction_budgets_available'], source_counts=source_status)
        commands = []
        if not (ROOT / 'edits/candidates.json').is_file():
            commands.append([sys.executable, str(HERE / 'build_edits.py'), '--source-manifest', str(source_manifest), '--gpu', '0'])
        if not completed(ROOT / 'lora-adaptation/status.json', {'complete'}):
            commands.append([sys.executable, str(HERE / 'lora/run.py'), '--source-manifest', str(source_manifest),
                             '--output', str(ROOT / 'lora-adaptation'), '--gpu', '1'])
        if commands:
            run_many('adaptation', commands)
        if not (ROOT / 'selection-frozen/schedule.json').is_file():
            run_many('freeze-selection', [[sys.executable, str(HERE / 'evaluate.py'), 'selection']])
        run_workers('selection')
        if not (ROOT / 'heldout-frozen/schedule.json').is_file():
            run_many('freeze-heldout', [[sys.executable, str(HERE / 'evaluate.py'), 'heldout']])
        run_workers('heldout')
        report = ROOT / 'analysis-v1/report.json'
        if not report.is_file():
            run_many('analysis', [[sys.executable, str(HERE / 'analyze.py')]])
        if read(report).get('status') != 'complete_and_verified':
            raise ValueError('Final result is not fully verified')
        update(status='complete_with_verified_report', stage='complete', report=str(report), report_sha256=sha(report))
    except BaseException as error:
        update(status='stopped', reason=repr(error))
        raise


if __name__ == '__main__':
    main()
