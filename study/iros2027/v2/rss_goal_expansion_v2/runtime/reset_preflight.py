"""Instantiate the declared scene and inspect two resets; no policy backend exists here."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import traceback

FORBIDDEN = ('policies.pi0_family.client', 'weightsmith.batched_client',
             'weightsmith.batched_server', 'weightsmith.rollout', 'openpi')


class NoPolicyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == x or fullname.startswith(x + '.') for x in FORBIDDEN):
            raise RuntimeError('Policy import forbidden in reset preflight: ' + fullname)


sys.meta_path.insert(0, NoPolicyImports())
import cv2  # noqa: E402,F401; required before IsaacLab
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--layouts', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import robolab.constants as constants
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.world.world_state import get_world
    from layouts import apply, load
    import torch
    output = args.output.resolve()
    constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    constants.RECORD_IMAGE_DATA = False
    constants.VERBOSE = False
    constants.DEBUG = False
    constants.set_output_dir(str(output))
    os.environ['OWSP_OBS_DIR'] = str(output)
    auto_register_droid_envs(task=[str(HERE / 'RSSGGlassesPickupTask.py')])
    spec = load(args.layouts)
    env, _ = create_env('RSSGGlassesPickupTask', device=args.device, seed=0, num_envs=16,
                        instruction_type='default', policy=None, renderer='realtime', rendering_mode='performance')
    state = {'status': 'running', 'policy_calls': 0, 'action_calls': 0,
             'reset_cycles': 0, 'num_envs': 16, 'layouts': str(args.layouts),
             'layout_ids': [x['id'] for x in spec['layouts']], 'started_at': time.time()}
    # Scene diagnostics are reads after the original update. No forces or poses are changed.
    update = env.scene.update
    tick = [0]
    phase = ['initial']
    world = get_world(env)
    diagnostics = (output / 'contact-diagnostics.jsonl').open('x')

    def sample():
        row = {'phase': phase[0], 'sample': tick[0], 'objects': {}, 'contacts': {}}
        for name in ('glasses', 'rubiks_cube', 'marker', 'table'):
            data = env.scene[name].data
            row['objects'][name] = {
                'pos': (data.root_pos_w - env.scene.env_origins).detach().cpu().tolist(),
                'quat_wxyz': data.root_quat_w.detach().cpu().tolist(),
                'linear_velocity': data.root_lin_vel_w.detach().cpu().tolist()}
        for a, b in (('rubiks_cube', 'marker'), ('rubiks_cube', 'table'), ('glasses', 'table')):
            row['contacts'][a + ':' + b] = {
                'above_0p1N': world.in_contact(a, b, .1, env_id=None).detach().cpu().tolist(),
                'net_force_world_N': world.get_contact_force(a, b, env_id=None).detach().cpu().tolist()}
        diagnostics.write(json.dumps(row) + '\n')
        diagnostics.flush()
        tick[0] += 1

    def observe_update(*a, **kw):
        result = update(*a, **kw)
        sample()
        return result

    def forbid_action(*a, **kw):
        state['action_calls'] += 1
        raise RuntimeError('Environment action step forbidden in reset preflight')

    env.step = forbid_action
    try:
        with torch.inference_mode():
            for reset in (1, 2):
                phase[0] = f'reset{reset}-stock'
                env._rss_scorer = None
                env.reset()
                sample()
                phase[0] = f'reset{reset}-layout'
                env.scene.update = observe_update
                try:
                    apply(env, spec, group=0, run=0, call=reset)
                finally:
                    env.scene.update = update
                state['reset_cycles'] += 1
        state.update(status='complete', completed_at=time.time(),
            forbidden_modules_loaded=[x for x in sys.modules if any(x == y or x.startswith(y + '.') for y in FORBIDDEN)])
    except BaseException as exc:
        state.update(status='failed', reason=repr(exc), stopped_at=time.time())
        raise
    finally:
        diagnostics.close()
        (output / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')
        env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        app.close()
        sys.exit(1)
    app.close()
