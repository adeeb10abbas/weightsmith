"""Evaluate GR00T N1.7 DROID on RoboLab with batched, paired-seed requests (Isaac Sim python, cwd = RoboLab).

RoboLab's ``policies/gr00t/run.py`` plus: ``--sampling-seed-base`` (paired noise seeds), the batched client in
groot_batched_client.py, and ``WSP_TASK_DIRS`` (task modules kept outside the RoboLab checkout, as in
weightsmith/run_batched.py), so the shared RoboLab checkout is never modified.
"""

import argparse
import pathlib
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "gr00t"

parser = argparse.ArgumentParser(description="Evaluate GR00T N1.7 DROID (batched, paired seeds).")
parser.add_argument("--remote-host", "--remote_host", type=str, default="127.0.0.1")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8600)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=8)
parser.add_argument("--sampling-seed-base", "--sampling_seed_base", type=int, default=None,
                    help="Episode-zero seed group; run r of env e at replan k uses ((base+r)*64+e)*100+k.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from groot_batched_client import BatchedGR00TClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug


def external_task_files(names):
    """Task class names defined in ``*.py`` files under ``WSP_TASK_DIRS`` resolve to those files."""
    import os
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


auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=external_task_files(args_cli.task),
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> BatchedGR00TClient:
    return BatchedGR00TClient(remote_host=args.remote_host, remote_port=args.remote_port,
                              open_loop_horizon=args.open_loop_horizon,
                              sampling_seed_base=args.sampling_seed_base)


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
