"""Identities and phase-specific seals for the additive few-pair validation study."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / 'projects/rss-few-pair-validation-20261120'
PRIOR = HERE.parent / 'rss_repair_comparison'
SCALING = HERE.parent / 'rss_repair_scaling_20261119'
POOL = HERE.parent / 'rss_goal_expansion_v2/runtime/single_attempt_pool.py'
sys.path.insert(0, str(REPO))

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20): h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save(path, value, *, exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():
        if read(path)!=value: raise ValueError(f'Refusing to change frozen artifact: {path}')
        return
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

def protocol(): return read(HERE/'protocol.json')

def goal(goal_id):
    from goals.registry import get_goal
    if goal_id not in protocol()['goal_ids']: raise ValueError('Unregistered goal')
    return get_goal(goal_id)

def goal_index(goal_id): return protocol()['goal_ids'].index(goal_id)

def goal_root(goal_id):
    goal_index(goal_id)
    return ROOT/goal_id

def slot_start(goal_id,stage):
    p=protocol();i=goal_index(goal_id)
    if stage=='collection': return p['collection']['slot_start']+i*p['collection']['goal_slot_stride']
    return p['evaluation'][stage+'_slot_start']+i*p['evaluation']['goal_slot_stride']

def verify(artifacts):
    for path,digest in artifacts.items():
        if sha(path)!=digest: raise ValueError(f'Sealed artifact changed: {path}')

def native_sources(goal_id):
    from weightsmith import config as C
    from goals.registry import runtime_sources
    g=goal(goal_id);runtime=Path(g['runtime_dir'])
    inventory=load('fewpair_prior_inventory',PRIOR/'make_eval_plan.py')
    files=[*inventory.runtime_sources(),POOL,HERE/'protocol.json',HERE/'common.py',HERE/'worker.py',
           *runtime_sources(goal_id),*sorted(runtime.glob('*.py')),
           C.BASE_WS/'workspace.json',REPO/'third_party/pi05_droid_jointpos.sha256']
    return sorted(set(Path(x).resolve() for x in files))

def freeze_plan(path,plan,artifacts):
    path=Path(path);freeze=path.with_name(path.stem+'-freeze.json')
    save(freeze,{'artifacts':artifacts,'plan_sha256':{plan['case']:canonical(plan)}},exclusive=True)
    save(path,{**plan,'global_freeze':{'path':str(freeze),'sha256':sha(freeze)}},exclusive=True)
    return path

def validate_plan(plan):
    p=protocol();g=goal(plan['goal_id'])
    if plan.get('cohort_id')!=p['cohort_id'] or plan.get('protocol_sha256')!=sha(HERE/'protocol.json'):
        raise ValueError('Cohort/protocol mismatch')
    if plan.get('case')!=g['runtime_case'] or plan.get('runtime_dir')!=g['runtime_dir'] or plan.get('inference_batching')!='fixed16':
        raise ValueError('Runtime identity mismatch')
    if plan.get('split') not in ('development','heldout'): raise ValueError('Unknown split')
    if not plan['batches'] or any(b['n']!=16 or b['slot_start']%16 for b in plan['batches']):
        raise ValueError('Only complete aligned fixed16 batches permitted')
    for batch in plan['batches']:
        if not Path(batch['layouts']).resolve().is_relative_to(goal_root(plan['goal_id'])/'design'):
            raise ValueError('Foreign layout file')
        if batch['goal'] not in (g['goal'],g['competing_goal']): raise ValueError('Unregistered task goal')
        if batch['task'] not in (g['task'],g['competing_task']): raise ValueError('Unregistered task')
        if batch.get('workspace') and not Path(batch['workspace']).resolve().is_relative_to(goal_root(plan['goal_id'])/'adaptation'):
            raise ValueError('Foreign candidate')
