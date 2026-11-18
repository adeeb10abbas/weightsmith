"""New semantic goals on existing scenes; original benchmark files are unchanged."""
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
from robolab.core.task.conditionals import object_on_top
from robolab.core.task.task import Task
from robolab.core.world.world_state import get_world
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring import CASES

OBJECTS={'banana_left_plate':('banana','plate_large','bowl'),
         'butter_behind_raisin':('butter','raisin_box'),
         'left_bowl_on_right':('bowl_1','bowl_2')}

def observe(env,case):
    world=get_world(env)
    positions={name:(env.scene[name].data.root_pos_w-env.scene.env_origins).detach().cpu().numpy() for name in OBJECTS[case]}
    contact=lambda a,b:world.in_contact(a,b,.1,env_id=None)
    cfg=CASES[case];mover,ref=cfg['mover'],cfg['reference']
    if case=='banana_left_plate':
        target=pl.left_of(world,mover,ref,'robot',False,45,env_id=None)
        preservation=object_on_top(env,mover,ref,env_id=None)
    elif case=='butter_behind_raisin':
        target=pl.behind(world,mover,ref,'robot',False,45,env_id=None)
        preservation=object_on_top(env,mover,ref,env_id=None)
    else:
        target=pl.in_opentop_container(world,mover,ref,.01,env_id=None)
        preservation=pl.in_opentop_container(world,ref,mover,.01,env_id=None)
    pred={'target':target,'preservation':preservation,
          'mover_table':contact(mover,'table'),'mover_reference':contact(mover,ref),
          'mover_gripped':contact(mover,'gripper')}
    if case=='left_bowl_on_right':pred['left_bowl_identity']=pl.left_of(world,'bowl_2','bowl_1','robot',False,45,env_id=None)
    for name in OBJECTS[case]:pred['detached_'+name]=~contact(name,'gripper')
    return positions,{k:v.detach().cpu().numpy() for k,v in pred.items()}

@atomic
def strict_goal(env,case,goal,env_id=None):
    scorer=getattr(env,'_rss_scorer',None)
    if scorer is None:result=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    else:
        if scorer.case!=case:raise RuntimeError('Wrong scorer case')
        positions,predicates=observe(env,case)
        frozen=getattr(env,'_pre_step_frozen',None)
        active=None if frozen is None else (~frozen).detach().cpu().numpy()
        scorer.update(int(env.common_step_counter),positions,predicates,active)
        env._rss_selected_goal=goal
        result=torch.as_tensor(scorer.current[goal],dtype=torch.bool,device=env.device)
    return result if env_id is None else bool(result[env_id])

def terms(case,goal):
    @configclass
    class Terms:
        time_out=DoneTerm(func=mdp.time_out,time_out=True)
        success=DoneTerm(func=strict_goal,params={'case':case,'goal':goal})
    return Terms

@dataclass
class RSSWBananaLeftPlateTask(Task):
    contact_object_list=['bagel_00','bagel_06','banana','bowl','plate_large','table']
    scene=import_scene('bagel_plate_banana_bowl.usda',contact_object_list)
    terminations=terms('banana_left_plate','banana_left_plate')
    instruction={k:'Move the banana on the table so that the plate is to the right of the banana.' for k in ('default','specific','vague')}
    episode_length_s:int=40
    attributes=['spatial'];subtasks=[]
@dataclass
class RSSWBananaOnPlateTask(RSSWBananaLeftPlateTask):
    terminations=terms('banana_left_plate','banana_on_plate')
    instruction={k:'Pick up the banana and put it on the plate.' for k in ('default','specific','vague')}
@dataclass
class RSSWButterBehindRaisinTask(Task):
    contact_object_list=['butter','raisin_box','table']
    scene=import_scene('butter_raisin_box.usda',contact_object_list)
    terminations=terms('butter_behind_raisin','butter_behind_raisin')
    instruction={k:'Move the butter box on the table so that the raisin box is in front of the butter box.' for k in ('default','specific','vague')}
    episode_length_s:int=40
    attributes=['spatial'];subtasks=[]
@dataclass
class RSSWButterOnRaisinTask(RSSWButterBehindRaisinTask):
    terminations=terms('butter_behind_raisin','butter_on_raisin')
    instruction={k:'Pick up the butter box and place it on top of the raisin box.' for k in ('default','specific','vague')}
@dataclass
class RSSWLeftBowlOnRightTask(Task):
    contact_object_list=['bowl_1','bowl_2','table']
    scene=import_scene('bowls_2_table.usda',contact_object_list)
    terminations=terms('left_bowl_on_right','left_bowl_on_right')
    instruction={k:'Move the bowl on the left so that the bowl on the right contains it.' for k in ('default','specific','vague')}
    episode_length_s:int=30
    attributes=['spatial'];subtasks=[]
@dataclass
class RSSWRightBowlOnLeftTask(RSSWLeftBowlOnRightTask):
    terminations=terms('left_bowl_on_right','right_bowl_on_left')
    instruction={k:'Place the bowl on the right inside the bowl on the left.' for k in ('default','specific','vague')}
