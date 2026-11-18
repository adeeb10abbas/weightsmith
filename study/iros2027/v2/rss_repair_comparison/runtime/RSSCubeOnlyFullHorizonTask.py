"""Cube-only preservation scored after the complete 60-second action budget.

Separate task identity preserves the older early-stop diagnostic unchanged.
"""
from pathlib import Path
import sys

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.task.decorators import atomic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as _tasks


def at_horizon(success, elapsed, horizon):
    return success & (elapsed >= horizon)


@atomic
def full_horizon_cube_only(env, env_id=None):
    # Update first events, reference motion and banana preservation on every tick.
    success = _tasks.strict_goal(env, "temporal", "cube_only", env_id=None)
    result = at_horizon(success, env.episode_length_buf, env.max_episode_length)
    return result if env_id is None else bool(result[env_id])


@configclass
class FullHorizonCubeOnlyTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=full_horizon_cube_only)


class RSSCubeOnlyFullHorizonTask(_tasks.RSSCubeOnlyTask):
    terminations = FullHorizonCubeOnlyTerms
    episode_length_s = 60
