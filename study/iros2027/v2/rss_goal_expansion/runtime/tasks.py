"""Separately registered tasks for historically informed new goal repairs."""
from dataclasses import dataclass
from pathlib import Path
import sys
import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task import predicate_logic as pl
from robolab.core.task.conditionals import object_picked_up
from robolab.core.task.decorators import atomic
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring import Scorer

OBJECTS = {'cube_behind':('rubiks_cube','banana','bowl'),
           'glasses_pickup':('glasses','rubiks_cube'),
           'mustard_left_bin':('mustard','grey_bin_left','grey_bin_right')}


def observe(env,case):
    world = get_world(env)
    positions = {name:(env.scene[name].data.root_pos_w-env.scene.env_origins).detach().cpu().numpy() for name in OBJECTS[case]}
    contact = lambda a,b:world.in_contact(a,b,.1,env_id=None)
    if case == 'cube_behind':
        predicates = {
            'behind':pl.behind(world,'rubiks_cube','bowl','robot',False,45,env_id=None),
            'cube_table_contact':contact('rubiks_cube','table'),
            'cube_bowl_contact':contact('rubiks_cube','bowl'),
            'cube_detached':~contact('rubiks_cube','gripper'),
            'cube_in_bowl':pl.in_opentop_container(world,'rubiks_cube','bowl',.01,env_id=None),
        }
    elif case == 'mustard_left_bin':
        predicates = {
            'mustard_left':pl.in_opentop_container(world,'mustard','grey_bin_left',.01,env_id=None),
            'mustard_right':pl.in_opentop_container(world,'mustard','grey_bin_right',.01,env_id=None),
            'mustard_left_contact':contact('mustard','grey_bin_left'),
            'mustard_right_contact':contact('mustard','grey_bin_right'),
            'mustard_detached':~contact('mustard','gripper'),
        }
    else:
        predicates = {
            'glasses_picked_up':object_picked_up(env,'glasses','table',.05,env_id=None),
            'cube_picked_up':object_picked_up(env,'rubiks_cube','table',.05,env_id=None),
        }
    return positions,{k:v.detach().cpu().numpy() for k,v in predicates.items()}


@atomic
def strict_goal(env,case,goal,env_id=None):
    scorer = getattr(env,'_rss_scorer',None)
    if scorer is None:
        result = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    else:
        if scorer.case != case:
            raise RuntimeError(f'Layout case {scorer.case} does not match {case}')
        positions,predicates = observe(env,case)
        frozen = getattr(env,'_pre_step_frozen',None)
        active = None if frozen is None else (~frozen).detach().cpu().numpy()
        scorer.update(int(env.common_step_counter),positions,predicates,active)
        env._rss_selected_goal = goal
        result = torch.as_tensor(scorer.current[goal],dtype=torch.bool,device=env.device)
    return result if env_id is None else bool(result[env_id])


@configclass
class CubeBehindTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'cube_behind','goal':'cube_behind'})
@configclass
class CubeInBowlTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'cube_behind','goal':'cube_in_bowl'})
@configclass
class GlassesPickupTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'glasses_pickup','goal':'glasses_pickup'})
@configclass
class DeskCubePickupTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'glasses_pickup','goal':'cube_pickup'})
@configclass
class MustardLeftBinTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'mustard_left_bin','goal':'mustard_left_bin'})
@configclass
class MustardRightBinTerms:
    time_out = DoneTerm(func=mdp.time_out,time_out=True)
    success = DoneTerm(func=strict_goal,params={'case':'mustard_left_bin','goal':'mustard_right_bin'})

@dataclass
class RSSGCubeBehindTask(Task):
    contact_object_list = ['rubiks_cube','banana','bowl','table']
    scene = import_scene('rubiks_cube_banana_bowl.usda',contact_object_list)
    terminations = CubeBehindTerms
    instruction = {k:"Place the Rubik's cube so that the bowl is in front of the Rubik's cube." for k in ('default','specific','vague')}
    episode_length_s:int = 30
    attributes = ['spatial']
    subtasks = []
@dataclass
class RSSGCubeInBowlTask(RSSGCubeBehindTask):
    terminations = CubeInBowlTerms
    instruction = {k:'Put the rubiks cube in the bowl' for k in ('default','specific','vague')}
@dataclass
class RSSGGlassesPickupTask(Task):
    contact_object_list = ['ceramic_mug','glasses','keyboard','lizard_figurine','marker','remote_control','rubiks_cube','smartphone','wooden_bowl','spoon_big','computer_mouse','yogurt_cup','oatmeal_raisin_cookies','granola_bars','table']
    scene = import_scene('workdesk.usda',contact_object_list)
    terminations = GlassesPickupTerms
    instruction = {k:'Pick up the eye glasses' for k in ('default','specific','vague')}
    episode_length_s:int = 30
    attributes = ['semantics']
    subtasks = []
@dataclass
class RSSGDeskCubePickupTask(RSSGGlassesPickupTask):
    terminations = DeskCubePickupTerms
    instruction = {k:'Pick up the rubiks cube' for k in ('default','specific','vague')}
@dataclass
class RSSGMustardLeftBinTask(Task):
    contact_object_list = ['table','mustard','grey_bin_right','grey_bin_left']
    scene = import_scene('two_bin.usda',contact_object_list)
    terminations = MustardLeftBinTerms
    instruction = {k:'Put the mustard in the left bin' for k in ('default','specific','vague')}
    episode_length_s:int = 30
    attributes = ['spatial']
    subtasks = []
@dataclass
class RSSGMustardRightBinTask(RSSGMustardLeftBinTask):
    terminations = MustardRightBinTerms
    instruction = {k:'Put the mustard in the right bin' for k in ('default','specific','vague')}
