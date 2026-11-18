"""External RoboLab tasks for the separately scored RSS comparison cohort."""
from dataclasses import dataclass
from pathlib import Path
import sys

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task import predicate_logic as pl
from robolab.core.task.decorators import atomic
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import Scorer

OBJECTS = {"temporal": ("rubiks_cube", "banana", "bowl"),
           "spatial": ("mustard_bottle", "raisin_box")}


def observe(env, case):
    world = get_world(env)
    positions = {name: (env.scene[name].data.root_pos_w - env.scene.env_origins).detach().cpu().numpy()
                 for name in OBJECTS[case]}
    contact = lambda a, b: world.in_contact(a, b, .1, env_id=None)
    if case == "temporal":
        predicates = {name + "_placed": pl.in_opentop_container(world, name, "bowl", 0., env_id=None)
                      & contact(name, "bowl") & ~contact(name, "gripper")
                      for name in ("rubiks_cube", "banana")}
    else:
        predicates = {
            "right_of": pl.right_of(world, "mustard_bottle", "raisin_box", "robot", False, 45, env_id=None),
            "left_of": pl.left_of(world, "mustard_bottle", "raisin_box", "robot", False, 45, env_id=None),
            "on_raisin": world.is_supported_on_surface("mustard_bottle", "raisin_box", env_id=None)
                & pl.centroid_in_footprint(world, "mustard_bottle", "raisin_box", .01, env_id=None),
            "mustard_table_contact": contact("mustard_bottle", "table"),
            "mustard_raisin_contact": contact("mustard_bottle", "raisin_box"),
            "mustard_detached": ~contact("mustard_bottle", "gripper"),
        }
    return positions, {k: v.detach().cpu().numpy() for k, v in predicates.items()}


@atomic
def strict_goal(env, case, goal, env_id=None):
    scorer = getattr(env, "_rss_scorer", None)
    if scorer is None:  # Environment setup before the explicit layout reset.
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        if scorer.case != case:
            raise RuntimeError(f"Layout case {scorer.case} does not match task case {case}")
        positions, predicates = observe(env, case)
        frozen = getattr(env, "_pre_step_frozen", None)
        active = None if frozen is None else (~frozen).detach().cpu().numpy()
        scorer.update(int(env.common_step_counter), positions, predicates, active)
        env._rss_selected_goal = goal
        result = torch.as_tensor(scorer.current[goal], dtype=torch.bool, device=env.device)
    return result if env_id is None else bool(result[env_id])


@configclass
class CubeThenBananaTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "temporal", "goal": "cube_then_banana"})

@configclass
class BananaThenCubeTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "temporal", "goal": "banana_then_cube"})

@configclass
class CubeOnlyTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "temporal", "goal": "cube_only"})

@configclass
class MustardRightTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "spatial", "goal": "mustard_right"})

@configclass
class MustardLeftTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "spatial", "goal": "mustard_left"})

@configclass
class MustardOnRaisinTerms:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=strict_goal, params={"case": "spatial", "goal": "mustard_on_raisin"})


@dataclass
class RSSCubeThenBananaTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    scene = import_scene("rubiks_cube_banana_bowl.usda", contact_object_list)
    terminations = CubeThenBananaTerms
    instruction = {k: "Put the cube then the banana in the bowl." for k in ("default", "vague", "specific")}
    episode_length_s: int = 60
    attributes = ["conjunction"]
    subtasks = []

@dataclass
class RSSBananaThenCubeTask(RSSCubeThenBananaTask):
    terminations = BananaThenCubeTerms
    instruction = {k: "Put the banana in the bowl first, then put the cube in the bowl." for k in ("default", "vague", "specific")}

@dataclass
class RSSCubeOnlyTask(RSSCubeThenBananaTask):
    terminations = CubeOnlyTerms
    instruction = {k: "Put only the cube in the bowl. Leave the banana where it is." for k in ("default", "vague", "specific")}

@dataclass
class RSSMustardRightTask(Task):
    contact_object_list = ["mustard_bottle", "raisin_box", "table"]
    scene = import_scene("mustard_raisin_box.usda", contact_object_list)
    terminations = MustardRightTerms
    instruction = {k: "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle." for k in ("default", "vague", "specific")}
    episode_length_s: int = 40
    attributes = ["spatial"]
    subtasks = []

@dataclass
class RSSMustardLeftTask(RSSMustardRightTask):
    terminations = MustardLeftTerms
    instruction = {k: "Move the mustard bottle to the left of the raisin box. Leave the raisin box where it is." for k in ("default", "vague", "specific")}

@dataclass
class RSSMustardOnRaisinTask(RSSMustardRightTask):
    terminations = MustardOnRaisinTerms
    instruction = {k: "Place the mustard bottle on top of the raisin box. Leave the raisin box where it is." for k in ("default", "vague", "specific")}
