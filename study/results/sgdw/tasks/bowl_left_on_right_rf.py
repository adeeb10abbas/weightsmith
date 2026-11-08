# Study goal S5-LR of "Same Goal, Different Words" (RWS-20260926) as a RoboLab task for the weightsmith runtime.
#
# Scene, objects, horizon (20 s at 15 Hz) and native success objects are those of the stock
# BowlStackingLeftOnRightTask (bowls_2_table.usda; left bowl = bowl_2, right bowl = bowl_1, as in the study).
# Success mirrors the study's "stable placement" for its "stacked" kind: bowl_2 inside bowl_1's open-top hull
# (1 cm tolerance), touching bowl_1, not touching the gripper, slower than 2 cm/s, for one continuous second.
#
# The default instruction is the study's reference-first (RF) wording; TF and DIR are weightsmith prompt overrides.
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
from robolab.core.task.conditionals import object_in_container
from robolab.core.task.decorators import atomic
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world

RF = "Move the left bowl so that the right bowl is underneath and supporting the left bowl."
TF = "Move the left bowl so that the left bowl is stacked on the right bowl."
DIR = "Stack the left bowl on the right bowl"


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
class BowlLeftOnRightTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=stable_goal, params={"kind": "stacked", "object": "bowl_2", "reference_object": "bowl_1"})


@dataclass
class BowlLeftOnRightRFTask(Task):
    """S5-LR: stack the left bowl (bowl_2) on the right bowl (bowl_1); RF wording by default."""
    contact_object_list = ["bowl_1", "bowl_2", "table"]
    scene = import_scene("bowls_2_table.usda", contact_object_list)
    terminations = BowlLeftOnRightTerminations
    instruction = {"default": RF, "vague": RF, "specific": RF}
    episode_length_s: int = 20
    attributes = ['spatial']
    subtasks = [
        Subtask(
            name="stack_bowl_left_on_right",
            conditions=partial(object_in_container, object=["bowl_2"], container="bowl_1", gripper_name="gripper",
                               require_contact_with=True, require_gripper_detached=True),
            logical="all",
            score=1.0,
        )
    ]
