"""Six-start, policy-free geometry check; never retry or search layouts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from worker import digest, save


def source_identity():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    paths = [Path(__file__), HERE / 'protocol.json',
             *sorted((HERE / 'runtime').glob('*.py')),
             *sorted((HERE / 'runtime/layouts').glob('*.json')),
             *[Path(x) for x in protocol['source_paths']]]
    return {str(p.resolve()): digest(p) for p in paths}


def verify_start(case, split, item):
    from runtime.layouts import validate_record
    if item['case'] != case or item['split'] != split or item['returncode'] != 0 or item.get('exception'):
        raise ValueError('Wrong preflight split or failed process')
    for path, sha in item['evidence_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Changed preflight evidence')
    summary = json.loads(Path(item['summary']).read_text())
    if (summary.get('status') != 'complete' or summary.get('policy_calls') != 0
            or summary.get('action_calls') != 0 or summary.get('reset_cycles') != 2
            or summary.get('forbidden_modules_loaded')):
        raise ValueError('Preflight did not complete both policy-free resets')
    spec = json.loads((HERE / f'runtime/layouts/{case}_{split}.json').read_text())
    expected = {x['id'] for x in spec['layouts']}
    for reset in (1, 2):
        record = json.loads((Path(item['summary']).parent / f'layout_r0_reset{reset}.json').read_text())
        validate_record(record)
        if (record['num_envs'] != 16 or set(record['layout_ids']) != expected
                or record['sha256'] != digest(HERE / f'runtime/layouts/{case}_{split}.json')):
            raise ValueError('Incomplete or changed layout reset coverage')


def verify(root):
    root = Path(root).resolve()
    receipt = root / 'preflight/complete.json'
    data = json.loads(receipt.read_text())
    if (data.get('status') != 'complete' or data.get('policy_calls') != 0
            or data.get('simulator_starts') != 6 or data.get('source_identity') != source_identity()):
        raise ValueError('Missing or changed passing reset preflight')
    protocol = json.loads((HERE/'protocol.json').read_text())
    expected = [(case, split) for case in protocol['case_order'] for split in ('development','heldout')]
    for (case, split), item in zip(expected, data['starts'], strict=True):
        verify_start(case, split, item)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(1,), required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.verify_only:
        value = verify(root)
        print(json.dumps({'status': value['status'], 'simulator_starts': 6, 'policy_calls': 0}))
        return
    from weightsmith import config
    from weightsmith.locks import GpuLock
    output = root / 'preflight'
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    save(output / 'source-freeze.json', identity)
    state = {'status': 'prepared', 'started_at': time.time(), 'source_identity': identity,
             'policy_calls': 0, 'simulator_starts': 0, 'starts': []}
    save(output / 'status.json', state)
    try:
        protocol = json.loads((HERE/'protocol.json').read_text())
        for case, split in [(c,s) for c in protocol['case_order'] for s in ('development','heldout')]:
            if source_identity() != identity:
                raise ValueError('Preflight sources changed')
            destination = output / f'{case}-{split}'
            destination.mkdir()
            layout = HERE / f'runtime/layouts/{case}_{split}.json'
            command = [str(config.SIM_PY), str(HERE / 'runtime/reset_preflight.py'),
                '--layouts', str(layout), '--output', str(destination), '--headless',
                '--device', f'cuda:{args.gpu}', '--/rtx/verifyDriverVersion/enabled=false',
                '--/renderer/multiGpu/enabled=false', '--/renderer/multiGpu/autoEnable=false',
                f'--/renderer/activeGpu={args.gpu}']
            queued = time.time()
            state.update(status='waiting_gpu', active_case=case, active_split=split)
            save(output / 'status.json', state)
            env = os.environ.copy()
            for key in ('OPENAI_API_KEY', 'OWSP_PROMPT', 'WSP_START_LAYOUT_IDS', 'WSP_START_LAYOUTS'):
                env.pop(key, None)
            env.update(OMNI_KIT_ACCEPT_EULA='Y', VK_ICD_FILENAMES=config.VK_ICD,
                       OWSP_OBS_DIR=str(destination))
            with GpuLock(args.gpu):
                begin = time.time()
                state.update(status='running', simulator_starts=state['simulator_starts'] + 1)
                save(output / 'status.json', state)
                save(destination / 'launch.json', {'command': command, 'cwd': str(config.ROBOLAB),
                     'start_unix': begin, 'queue_seconds': begin - queued, 'policy_server': False})
                returncode, error = None, None
                try:
                    with (destination / 'sim.log').open('x') as log:
                        proc = subprocess.run(command, cwd=config.ROBOLAB, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, timeout=900)
                    returncode = proc.returncode
                except BaseException as exc:
                    error = repr(exc)
                    raise
                finally:
                    item = {'case': case, 'split': split, 'returncode': returncode, 'exception': error,
                            'queue_seconds': begin - queued, 'execution_seconds': time.time() - begin,
                            'summary': str(destination / 'summary.json'),
                            'evidence_sha256': {str(p): digest(p) for p in sorted(destination.iterdir()) if p.is_file()}}
                    state['starts'].append(item)
                    save(output / 'status.json', state)
            if returncode != 0:
                raise RuntimeError(f'{split} reset failed; no further starts or geometry search')
            verify_start(case, split, item)
        if source_identity() != identity:
            raise ValueError('Preflight sources changed during checks')
        state.update(status='complete', completed_at=time.time())
        save(output / 'complete.json', state)
        verify(root)
        save(output / 'status.json', state)
        print(json.dumps({'status': 'complete', 'simulator_starts': 6, 'policy_calls': 0}))
    except BaseException as exc:
        state.update(status='stopped', reason=repr(exc), stopped_at=time.time())
        save(output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
