"""Native, policy-free reset qualification for the frozen evaluation layouts.

Run with --gpu 1 only after source collection has released that GPU. This file
contains both the CPU supervisor and its isolated Isaac child (--native).
Neither creates or revises layouts, serves a policy, or executes robot actions.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from common import HERE, PRIOR, ROOT, load, protocol, read, save, sha, source_inventory

FORBIDDEN = ('policies.pi0_family.client', 'weightsmith.batched_client',
             'weightsmith.batched_server', 'weightsmith.rollout', 'openpi')
SPLITS = (('changed', 'changed-layouts.json', 'heldout_slot_start'),
          ('development', 'development-layouts.json', 'selection_slot_start'))


def identity():
    paths = [*source_inventory(), Path(__file__),
             *(ROOT / 'design' / filename for _, filename, _ in SPLITS)]
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def check_layout(path, split):
    spec = read(path)
    ids = [r['id'] for r in spec['layouts']]
    expected_split = 'heldout' if split == 'changed' else 'development'
    if (spec['case'] != 'temporal' or spec['split'] != expected_split
            or len(ids) != 16 or len(set(ids)) != 16
            or set(spec['objects']) != {'rubiks_cube', 'banana', 'bowl'}):
        raise ValueError('Expected all sixteen frozen temporal layouts')
    return spec


def check_measurement(record, extra, spec, layout, start, reset):
    runtime = load('scaling_reset_validation', PRIOR / 'runtime/layouts.py')
    runtime.validate_record(record)
    ids = [x['id'] for x in spec['layouts']]
    if (record.get('case') != 'temporal' or record['num_envs'] != 16
            or record['reset_call'] != reset or record['layout_ids'] != ids
            or record['slots'] != list(range(start, start + 16))
            or record['sha256'] != sha(layout)
            or set(record['objects']) != set(spec['objects'])):
        raise ValueError('Reset did not cover the exact frozen slots/layouts')
    for name in spec['objects']:
        actual = record['objects'][name]
        expected = np.asarray([x['poses'][name]['pos'] for x in spec['layouts']])
        expected_quat = np.asarray([x['poses'][name]['quat'] for x in spec['layouts']])
        if (not np.allclose(actual['written_pos'], expected, rtol=0, atol=1e-6)
                or not np.allclose(actual['written_quat'], expected_quat, rtol=0, atol=1e-6)):
            raise ValueError(f'Wrong requested pose for {name}')
        # The existing runtime checks XY, orientation, drift, and speed. Also
        # reject vertical collapse and unsupported objects in this reset audit.
        settled = np.asarray(actual['settled_pos'])
        if np.max(np.abs(settled[:, 2] - expected[:, 2])) > .025:
            raise ValueError(f'Over 25 mm vertical reset displacement: {name}')
        contact = extra['table_contact'][name]
        if len(contact) != 16 or not all(contact):
            raise ValueError(f'Object lacks measured table contact: {name}')
    predicates = record['initial_predicates']
    if set(predicates) != {'rubiks_cube_placed', 'banana_placed'}:
        raise ValueError('Missing actual initial placement predicates')
    if any(len(v) != 16 or any(v) for v in predicates.values()):
        raise ValueError('A placement predicate is initially true')
    if (set(extra['initial_strict_goals']) != {'cube_then_banana', 'banana_then_cube', 'cube_only'}
            or any(len(v) != 16 or any(v) for v in extra['initial_strict_goals'].values())):
        raise ValueError('A sequence goal is initially true')
    frames = extra['frames']
    cameras = {f['camera'] for f in frames}
    if len(cameras) < 2 or any({f['env'] for f in frames if f['camera'] == c} != set(range(16)) for c in cameras):
        raise ValueError('Missing rendered observations for one or more layouts')
    for frame in frames:
        if sha(frame['path']) != frame['sha256'] or frame['layout_id'] != ids[frame['env']]:
            raise ValueError('Changed or misassigned reset frame')


def verify_start(item):
    split, filename, slot_key = next(x for x in SPLITS if x[0] == item['split'])
    layout = ROOT / 'design' / filename
    spec = check_layout(layout, split)
    if item['returncode'] != 0 or item.get('exception'):
        raise ValueError('Native reset process did not exit successfully')
    for path, digest in item['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Changed reset evidence: {path}')
    destination = Path(item['summary']).parent
    summary = read(item['summary'])
    if (summary['status'] != 'complete' or summary['reset_cycles'] != 2
            or summary['policy_calls'] != 0 or summary['action_calls'] != 0
            or summary['forbidden_modules_loaded']):
        raise ValueError('Both policy-free reset cycles must complete')
    for reset in (1, 2):
        check_measurement(read(destination / f'layout_r0_reset{reset}.json'),
                          read(destination / f'measured-reset{reset}.json'), spec,
                          layout, protocol()['evaluation'][slot_key], reset)


def verify_receipt(receipt):
    if (receipt['status'] != 'passed' or receipt['policy_calls'] != 0
            or receipt['simulator_starts'] != 2 or receipt['source_identity'] != identity()
            or receipt['layouts_sha256'] != sha(ROOT / 'design/changed-layouts.json')
            or [x['split'] for x in receipt['starts']] != [x[0] for x in SPLITS]):
        raise ValueError('Missing or changed passing reset qualification')
    for item in receipt['starts']:
        verify_start(item)
    return receipt


def verify():
    return verify_receipt(read(ROOT / 'preflight/receipt.json'))


def controller(gpu):
    from weightsmith import config
    from weightsmith.locks import GpuLock
    destination = ROOT / 'preflight'
    destination.mkdir(parents=True, exist_ok=False)
    frozen = identity()
    save(destination / 'source-freeze.json', frozen, exclusive=True)
    state = {'status': 'prepared', 'cohort_id': protocol()['cohort_id'],
             'source_identity': frozen, 'started_at': time.time(), 'gpu': gpu,
             'policy_calls': 0, 'simulator_starts': 0, 'starts': [],
             'layouts_sha256': sha(ROOT / 'design/changed-layouts.json'),
             'scope': '32 distinct layouts; two policy-free resets each; no robot-policy episodes'}
    save(destination / 'status.json', state)
    try:
        for split, filename, slot_key in SPLITS:
            if identity() != frozen:
                raise ValueError('Preflight sources changed')
            layout = ROOT / 'design' / filename
            check_layout(layout, split)
            output = destination / split
            output.mkdir()
            command = [str(config.SIM_PY), '-B', str(Path(__file__).resolve()), '--native',
                       '--layouts', str(layout), '--output', str(output), '--split', split,
                       '--slot-start', str(protocol()['evaluation'][slot_key]),
                       '--headless', '--device', f'cuda:{gpu}',
                       '--/rtx/verifyDriverVersion/enabled=false', '--/renderer/multiGpu/enabled=false',
                       '--/renderer/multiGpu/autoEnable=false', f'--/renderer/activeGpu={gpu}']
            env = os.environ.copy()
            for key in ('OPENAI_API_KEY', 'OWSP_PROMPT', 'WSP_START_LAYOUT_IDS', 'WSP_START_LAYOUTS', 'CUDA_VISIBLE_DEVICES'):
                env.pop(key, None)
            env.update(OMNI_KIT_ACCEPT_EULA='Y', VK_ICD_FILENAMES=config.VK_ICD,
                       OWSP_OBS_DIR=str(output), PYTHONDONTWRITEBYTECODE='1')
            queued = time.time()
            state.update(status='waiting_gpu', active_split=split)
            save(destination / 'status.json', state)
            with GpuLock(gpu):
                began = time.time()
                state.update(status='running', simulator_starts=state['simulator_starts'] + 1)
                save(destination / 'status.json', state)
                save(output / 'launch.json', {'command': command, 'cwd': str(config.ROBOLAB),
                                             'started_at': began, 'queue_seconds': began - queued,
                                             'policy_server': False})
                code, error, proc = None, None, None
                try:
                    with (output / 'sim.log').open('x') as log:
                        proc = subprocess.Popen(command, cwd=config.ROBOLAB, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                        code = proc.wait(timeout=900)
                except BaseException as exc:
                    error = repr(exc)
                    raise
                finally:
                    if proc is not None and proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=25)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait(timeout=5)
                    item = {'split': split, 'returncode': code, 'exception': error,
                            'summary': str(output / 'summary.json'), 'queue_seconds': began - queued,
                            'execution_seconds': time.time() - began,
                            'evidence_sha256': {str(p): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
                    state['starts'].append(item)
                    save(destination / 'status.json', state)
            if code != 0:
                raise RuntimeError(f'{split} reset failed; no retry, resampling, or later start')
            verify_start(item)
        if identity() != frozen:
            raise ValueError('Sources changed during native checks')
        state.update(status='passed', completed_at=time.time(), active_split=None)
        verify_receipt(state)
        save(destination / 'status.json', state)
        save(destination / 'receipt.json', state, exclusive=True)
        print(json.dumps({'status': 'passed', 'policy_calls': 0, 'simulator_starts': 2,
                          'receipt': str(destination / 'receipt.json')}))
    except BaseException as exc:
        state.update(status='stopped', reason=repr(exc), stopped_at=time.time())
        save(destination / 'status.json', state)
        raise


def native():
    class NoPolicyImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == x or fullname.startswith(x + '.') for x in FORBIDDEN):
                raise RuntimeError('Policy import forbidden in reset preflight: ' + fullname)
    sys.meta_path.insert(0, NoPolicyImports())
    import cv2  # Required before IsaacLab.
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--layouts', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=[x[0] for x in SPLITS], required=True)
    ap.add_argument('--slot-start', type=int, required=True)
    AppLauncher.add_app_launcher_args(ap)
    args, _ = ap.parse_known_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    state = {'status': 'starting', 'policy_calls': 0, 'action_calls': 0, 'reset_cycles': 0,
             'num_envs': 16, 'layouts': str(args.layouts), 'started_at': time.time()}
    try:
        import torch
        import robolab.constants as constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        sys.path.insert(0, str(PRIOR / 'runtime'))
        from layouts import apply, load as load_layouts
        constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        constants.RECORD_IMAGE_DATA = False
        constants.VERBOSE = False
        constants.DEBUG = False
        constants.set_output_dir(str(args.output))
        os.environ['OWSP_OBS_DIR'] = str(args.output)
        check_layout(args.layouts, args.split)
        spec = load_layouts(args.layouts)
        if args.slot_start % 16:
            raise ValueError('Reset slots must match fixed16 evaluation groups')
        auto_register_droid_envs(task=[str(PRIOR / 'runtime/RSSCubeThenBananaTask.py')])
        env, _ = create_env('RSSCubeThenBananaTask', device=args.device, seed=0, num_envs=16,
                            instruction_type='default', policy=None, renderer='realtime', rendering_mode='performance')
        world = get_world(env)
        def no_actions(*a, **kw):
            state['action_calls'] += 1
            raise RuntimeError('Robot action step forbidden in reset preflight')
        env.step = no_actions
        with torch.inference_mode():
            for reset in (1, 2):
                env._rss_scorer = None
                env.reset()
                record = apply(env, spec, group=args.slot_start // 16, run=0, call=reset)
                frames = []
                frame_dir = args.output / f'frames-reset{reset}'
                frame_dir.mkdir()
                for name, camera in env.scene.sensors.items():
                    if 'Camera' not in type(camera).__name__ or 'rgb' not in camera.data.output:
                        continue
                    rgb = camera.data.output['rgb'].detach().cpu().numpy()
                    if rgb.shape[0] != 16 or rgb.shape[-1] < 3 or not np.isfinite(rgb).all():
                        raise ValueError('Invalid native camera output')
                    if rgb.dtype != np.uint8:
                        raise ValueError('Expected native uint8 RGB camera frames')
                    for i, image in enumerate(rgb):
                        path = frame_dir / f'{name}-env{i:02d}.png'
                        if not cv2.imwrite(str(path), cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)):
                            raise RuntimeError('Failed to save native reset image')
                        frames.append({'camera': name, 'env': i, 'layout_id': record['layout_ids'][i],
                                       'path': str(path), 'sha256': sha(path), 'shape': list(image.shape)})
                extra = {'reset': reset, 'layout_ids': record['layout_ids'], 'frames': frames,
                         'table_contact': {name: world.in_contact(name, 'table', .1, env_id=None).detach().cpu().tolist()
                                           for name in spec['objects']},
                         'table_contact_force_world_N': {name: world.get_contact_force(name, 'table', env_id=None).detach().cpu().tolist()
                                                        for name in spec['objects']},
                         'initial_strict_goals': env._rss_scorer.record()['strict_final']}
                save(args.output / f'measured-reset{reset}.json', extra)
                check_measurement(record, extra, spec, args.layouts, args.slot_start, reset)
                state['reset_cycles'] += 1
        state.update(status='complete', completed_at=time.time(),
                     forbidden_modules_loaded=[x for x in sys.modules if any(x == y or x.startswith(y + '.') for y in FORBIDDEN)])
    except BaseException as exc:
        state.update(status='failed', reason=repr(exc), stopped_at=time.time())
        raise
    finally:
        save(args.output / 'summary.json', state)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    if '--native' in sys.argv:
        native()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--gpu', type=int, choices=(0, 1))
        parser.add_argument('--verify-only', action='store_true')
        args = parser.parse_args()
        if args.verify_only:
            result = verify()
            print(json.dumps({'status': result['status'], 'policy_calls': 0, 'simulator_starts': 2}))
        elif args.gpu is None:
            parser.error('--gpu is required for native reset checks')
        else:
            controller(args.gpu)
