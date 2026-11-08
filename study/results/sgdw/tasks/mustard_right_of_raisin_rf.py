# Study goal S4-R of "Same Goal, Different Words" (RWS-20260926) as a RoboLab task for the weightsmith runtime.
#
# Scene, objects and horizon are those of the stock MustardAboveRaisinTask (mustard_raisin_box.usda, 40 s at 15 Hz),
# which the study used for scene S4. The goal is the study-added "mustard bottle to the right of the raisin box"
# (robot frame). Success mirrors the study's "stable placement" (experiments/robolab_workshop/scoring.py and
# sim_env.study_predicate at world-action-model-steerability@9a6fcde): the relation holds (45-degree cone, robot
# frame), the mustard touches the table, the gripper does not touch it, and it moves slower than 2 cm/s (finite
# difference of its root position between control ticks), for one continuous second.
#
# The default instruction is the study's reference-first (RF) wording, so this task's own instruction is the one
# being repaired; the target-first (TF) and direct (DIR) wordings are given as weightsmith prompt overrides.
# Loaded from outside the RoboLab checkout through WSP_TASK_DIRS (see weightsmith/run_batched.py).

import math
from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task import predicate_logic as pl
from robolab.core.task.conditionals import object_dropped, object_grabbed, object_right_of
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world

RF = "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF = "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
DIR = "Place the mustard bottle on the table to the right of the raisin box."


@atomic
def stable_goal(env, kind: str, object: str, reference_object: str, support: str = "table",
                gripper_name: str = "gripper", cone_deg: int = 45, force_threshold: float = 0.1,
                speed_mps: float = 0.02, dwell_s: float = 1.0, env_id=None):
    """Study 'stable placement': goal predicate and speed < speed_mps for dwell_s of consecutive control ticks."""
    world = get_world(env)
    if kind in ("left_of", "right_of", "in_front_of", "behind"):
        relation = getattr(pl, kind)(world, object, reference_object, "robot", False, cone_deg, env_id=None)
        supported = world.in_contact(object, support, force_threshold, env_id=None)
    elif kind == "stacked":
        relation = pl.in_opentop_container(world, object, reference_object, 0.01, env_id=None)
        supported = world.in_contact(object, reference_object, force_threshold, env_id=None)
    else:
        raise ValueError(kind)
    held = world.in_contact(object, gripper_name, force_threshold, env_id=None)
    pos = world.get_pose(object, env_id=None)[0].clone()  # env-local root positions [N, 3], as the study records
    key = f"_sgdw_stable_{object}_{kind}"
    st = getattr(env, key, None)
    if st is None or st["pos"].shape != pos.shape:
        st = {"step": None, "pos": pos.clone(), "count": torch.zeros(len(pos), dtype=torch.long, device=pos.device),
              "out": torch.zeros(len(pos), dtype=torch.bool, device=pos.device)}
        setattr(env, key, st)
    step = int(env.common_step_counter)
    if st["step"] == step:  # evaluated once per control tick, whoever calls
        return st["out"]
    st["step"] = step
    fresh = env.episode_length_buf.to(pos.device) <= 1
    speed = torch.linalg.norm(pos - st["pos"], dim=-1) / float(env.step_dt)
    speed = torch.where(fresh, torch.zeros_like(speed), speed)
    ok = relation.to(pos.device) & supported.to(pos.device) & ~held.to(pos.device) & (speed < speed_mps)
    count = torch.where(ok, st["count"] + 1, torch.zeros_like(st["count"]))
    count = torch.where(fresh, ok.long(), count)
    need = int(math.ceil(dwell_s / float(env.step_dt) - 1e-6)) + 1  # span (need-1) ticks >= dwell_s
    st["pos"], st["count"], st["out"] = pos, count, count >= need
    return st["out"]


@configclass
class MustardRightOfRaisinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stable_goal, params={"kind": "right_of", "object": "mustard_bottle",
                                                 "reference_object": "raisin_box"})


@dataclass
class MustardRightOfRaisinRFTask(Task):
    """S4-R: move the mustard bottle to the right of the raisin box (robot frame); RF wording by default."""
    contact_object_list = ["mustard_bottle", "raisin_box", "table"]
    scene = import_scene("mustard_raisin_box.usda", contact_object_list)
    terminations = MustardRightOfRaisinTerminations
    instruction = {"default": RF, "vague": RF, "specific": RF}
    episode_length_s: int = 40
    attributes = ['spatial']
    subtasks = [
        Subtask(
            name="mustard_right_of_raisin_box",
            conditions=[
                partial(object_grabbed, object="mustard_bottle"),
                partial(object_right_of, object="mustard_bottle", reference_object="raisin_box",
                        frame_of_reference="robot", mirrored=False),
                partial(object_dropped, object="mustard_bottle"),
            ],
            logical="all",
            score=1.0,
        )
    ]
