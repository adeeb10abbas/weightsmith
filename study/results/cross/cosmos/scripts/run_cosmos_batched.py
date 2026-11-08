"""RoboLab evaluation of Cosmos3 Edge Policy DROID with weightsmith's batched slot scheme (Isaac Sim venv).

Adapted from weightsmith/run_batched.py (pi0.5) and RoboLab's policies/cosmos3/run.py: envs are registered with the
three cameras the Cosmos3 client reads (wrist and the two over-the-shoulder views, WRIST_LEFT_RIGHT; the stock
run.py also renders a head camera the client never uses), task classes may come from WSP_TASK_DIRS (task modules
kept outside the RoboLab checkout, as in the sgdw study), and the client is BatchedCosmos3Client (cosmos_client.py).
"""
import argparse
import os
import pathlib
import sys
import traceback

import cv2  # noqa: F401 -- must be imported before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Cosmos3 Edge with batched requests.")
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", type=int, default=8500)
parser.add_argument("--open-loop-horizon", type=int, default=None)
parser.add_argument("--sampling-seed-base", type=int, default=0)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cosmos_client import BatchedCosmos3Client  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = False


def external_task_files(names):
    """Task class names -> task module paths from WSP_TASK_DIRS (as weightsmith/run_batched.py)."""
    if names is None:
        return None
    dirs = [d for d in os.environ.get("WSP_TASK_DIRS", "").split(os.pathsep) if d.strip()]
    out = []
    for name in names:
        hit = None
        if dirs and "/" not in name and not name.endswith(".py"):
            for d in dirs:
                for f in sorted(pathlib.Path(d).expanduser().glob("*.py")):
                    if f"class {name}(" in f.read_text():
                        hit = str(f.resolve())
                        break
                if hit:
                    break
        out.append(hit or name)
    return out


auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=external_task_files(args_cli.task), cameras=WRIST_LEFT_RIGHT)


def make_client(args):
    return BatchedCosmos3Client(remote_host=args.remote_host, remote_port=args.remote_port,
                                open_loop_horizon=args.open_loop_horizon, sampling_seed_base=args.sampling_seed_base)


def main():
    run_evaluation(args_cli, policy="cosmos3", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[RoboLab] Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
