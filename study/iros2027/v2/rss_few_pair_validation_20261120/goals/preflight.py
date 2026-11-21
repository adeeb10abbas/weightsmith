"""Check every declared layout twice without importing or calling a policy.

Run inside the existing RoboLab/Isaac environment, with its normal working
directory. The caller owns the GPU lock and process/output log. Nothing in this
entry point changes old task, runtime, source, or outcome files.
"""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

try:
    from .registry import GOAL_IDS, get_goal, runtime_sources
except ImportError:
    from registry import GOAL_IDS, get_goal, runtime_sources

FORBIDDEN = ('policies.pi0_family.client', 'weightsmith.batched_client',
             'weightsmith.batched_server', 'weightsmith.rollout', 'openpi')


class NoPolicyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            raise RuntimeError('Policy import forbidden in reset preflight: ' + fullname)


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main(args):
    goal = get_goal(args.goal)
    runtime = Path(goal['runtime_dir'])
    sys.path.insert(0, str(runtime))
    os.environ['WSP_TASK_DIRS'] = str(runtime)
    import robolab.constants as constants
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.world.world_state import get_world
    from layouts import apply, load
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists() or list(output.glob('layout_r*.json')):
        raise FileExistsError('Preflight output already contains evidence; use a fresh attempt directory')
    spec = load(args.layouts)
    if spec.get('goal_id') != args.goal or spec['case'] != goal['runtime_case']:
        raise ValueError('Layout identity differs from preflight goal')
    if len(spec['layouts']) % 16 or len(spec['layouts']) == 0:
        raise ValueError('Every layout must fit into complete groups of sixteen')
    constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    constants.RECORD_IMAGE_DATA = False
    constants.VERBOSE = False
    constants.DEBUG = False
    constants.set_output_dir(str(output))
    os.environ['OWSP_OBS_DIR'] = str(output)
    identities = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [*runtime_sources(args.goal), args.layouts.resolve()]}
    state = {'status': 'starting', 'goal_id': args.goal,
             'task': goal['task'], 'runtime_dir': str(runtime),
             'policy_calls': 0, 'action_calls': 0, 'reset_cycles': 0,
             'planned_reset_cycles': len(spec['layouts']) // 16 * 2,
             'num_envs': 16, 'layouts': str(args.layouts.resolve()),
             'layout_ids': [row['id'] for row in spec['layouts']],
             'source_sha256': identities, 'started_at': time.time(),
             'resets': []}
    save(output / 'summary.json', state)
    env = None
    try:
        auto_register_droid_envs(task=[str(runtime / (goal['task'] + '.py'))])
        env, _ = create_env(goal['task'], device=args.device, seed=0, num_envs=16,
                            instruction_type='default', policy=None,
                            renderer='realtime', rendering_mode='performance')

        def forbid_action(*a, **kw):
            state['action_calls'] += 1
            raise RuntimeError('Environment action step forbidden in reset preflight')

        env.step = forbid_action
        state['status'] = 'running'
        with torch.inference_mode():
            for group in range(len(spec['layouts']) // 16):
                for cycle in (1, 2):
                    env._rss_scorer = None
                    env.reset()
                    record = apply(env, spec, group=group, run=group, call=cycle)
                    world = get_world(env)
                    support = {}
                    for name, values in record['objects'].items():
                        written = np.asarray(values['written_pos'])
                        settled = np.asarray(values['settled_pos'])
                        if np.max(np.abs(settled[:, 2] - written[:, 2])) > .025:
                            raise RuntimeError('Vertical reset displacement over 25mm: ' + name)
                        support[name] = world.in_contact(name, 'table', .1, env_id=None).detach().cpu().tolist()
                        if len(support[name]) != 16 or not all(support[name]):
                            raise RuntimeError('Missing table support at reset: ' + name)
                    state['resets'].append({'group': group, 'cycle': cycle,
                                            'layout_ids': record['layout_ids'], 'table_contact': support,
                                            'vertical_displacement_checked': True})
                    state['reset_cycles'] += 1
                    save(output / 'summary.json', state)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
               for path, digest in identities.items()):
            raise RuntimeError('Preflight runtime or input changed during execution')
        forbidden = [name for name in sys.modules
                     if any(name == x or name.startswith(x + '.') for x in FORBIDDEN)]
        if forbidden:
            raise RuntimeError('Policy module entered policy-free reset process')
        state.update(status='complete', completed_at=time.time(),
                     forbidden_modules_loaded=forbidden)
    except BaseException as exc:
        state.update(status='failed', reason=repr(exc), stopped_at=time.time())
        raise
    finally:
        save(output / 'summary.json', state)
        if env is not None:
            env.close()


if __name__ == '__main__':
    sys.meta_path.insert(0, NoPolicyImports())
    import cv2  # noqa: F401,E402; required before IsaacLab
    from isaaclab.app import AppLauncher  # noqa: E402
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', choices=GOAL_IDS, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        main(args)
    except BaseException:
        traceback.print_exc()
        app.close()
        sys.exit(1)
    app.close()
