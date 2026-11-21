"""Seal development selection and the entire registered held-out comparison."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from common import HERE,ROOT,canonical,freeze_plan,goal,goal_root,native_sources,protocol,read,save,sha,slot_start,validate_plan,verify

def serving_fingerprint(workspace,manifest,edited_names):
    """Match native Server.load: tensor names and loaded array bytes, in order."""
    digest=hashlib.sha256()
    for name in sorted(edited_names):
        array=np.load(Path(workspace)/'edits'/manifest['tensors'][name]['file'],allow_pickle=False)
        digest.update(name.encode());digest.update(array.tobytes())
    return digest.hexdigest()

def candidate_identity(candidate,goal_id):
    from weightsmith import config as C
    from weightsmith.edits import fingerprint
    c=dict(candidate);workspace=Path(c['workspace']).resolve()
    if not workspace.is_relative_to(goal_root(goal_id)/'adaptation'):
        raise ValueError('Candidate outside this new goal')
    files=[]
    if (workspace/'baseline.json').is_file():
        from baselines.artifacts import load_artifact
        spec,_=load_artifact(workspace,verify_sources=True)
        expected='embedding_mean' if c['method']=='embedding' else 'residual_cells'
        if spec['method']!=expected or spec['strength']!=c['strength']:
            raise ValueError('Baseline registry differs from implementation')
        if expected=='residual_cells' and spec['layer']!=protocol()['baselines']['residual_layer']:
            raise ValueError('Undeclared residual layer')
        files=[workspace/'baseline.json',workspace/'direction.npy']
        c['fingerprint']=sha(workspace/'baseline.json')
    else:
        manifest=read(workspace/'workspace.json');base=read(C.BASE_WS/'workspace.json')
        for k in ('base_vlm_sha256','checkpoint','total_parameters_and_buffers'):
            if manifest[k]!=base[k]: raise ValueError('Edited candidate uses different base')
        if (workspace/'base').resolve()!=(C.BASE_WS/'base').resolve(): raise ValueError('Foreign immutable base')
        edits=sorted((workspace/'edits').glob('*.npy'));names={v['file']:k for k,v in manifest['tensors'].items()}
        expected={f'language.blocks.{i:02d}.mlp.linear' for i in protocol()['edit']['layers']}
        if len(edits)!=len(expected) or {names[p.name] for p in edits}!=expected:
            raise ValueError('Edited candidate writes outside the prescribed projections')
        from weightsmith.lab.openpi_vlm_full import validate_workspace
        checked=validate_workspace(workspace)
        if not checked['edited_tensors']: raise ValueError('Weightsmith candidate has no effective edit')
        files=[workspace/'workspace.json',*edits]
        c['file_fingerprint']=fingerprint(workspace/'edits')
        c['fingerprint']=serving_fingerprint(workspace,manifest,checked['edited_tensors'])
    verify(c.get('artifacts',{}))
    c['workspace']=str(workspace)
    c['artifacts']={**c.get('artifacts',{}),**{str(p):sha(p) for p in files}}
    return c

def registries(goal_id):
    root=goal_root(goal_id);source_status=read(root/'source-status.json')
    if source_status['status']!='qualified': raise ValueError('Goal has not qualified')
    manifest=Path(source_status['manifest']);source=read(manifest)
    if sha(manifest)!=source_status['manifest_sha256'] or len(source['pairs'])!=9:
        raise ValueError('Wrong source pool')
    artifacts={str(manifest):sha(manifest)};records={}
    for name in ('weightsmith','baselines'):
        path=root/f'adaptation/{name}/candidates.json';rec=read(path)
        if rec['source_manifest']!=str(manifest) or rec['source_manifest_sha256']!=sha(manifest):
            raise ValueError('Methods fitted with different source manifests')
        verify(rec['artifacts']);artifacts.update(rec['artifacts']);artifacts[str(path)]=sha(path)
        records[name]=[candidate_identity(c,goal_id) for c in rec['candidates']]
        for c in records[name]: artifacts.update(c['artifacts'])
    edits=records['weightsmith'];baselines=records['baselines']
    if sorted(c['candidate_id'] for c in edits)!=['weightsmith_g0','weightsmith_g1','weightsmith_g2']:
        raise ValueError('Need all three preassigned edits')
    if sorted(c['group_id'] for c in edits)!=['g0','g1','g2']:
        raise ValueError('Duplicate or missing source group')
    for c in edits:
        c['method']='weightsmith'
    families=['embedding','residual_g0','residual_g1','residual_g2']
    if set(c['family'] for c in baselines)!=set(families) or len(baselines)!=12:
        raise ValueError('Need the entire declared baseline grid')
    for family in families:
        if sorted(c['strength'] for c in baselines if c['family']==family)!=protocol()['baselines']['strengths']:
            raise ValueError('Baseline grid differs')
    if len({c['candidate_id'] for c in edits+baselines})!=15: raise ValueError('Candidate IDs not unique')
    return edits,baselines,artifacts,source

def test_cells(goal_id,condition,workspace):
    g=goal(goal_id);root=goal_root(goal_id);start=slot_start(goal_id,'heldout')
    cells=[]
    for i in range(2):
        cells.append({'cell':f'original_{i}','condition':condition,'task':g['task'],'goal':g['goal'],
            'input_prompt':g['original'],'effective_prompt':g['teacher'] if condition=='clarified' else g['original'],
            'workspace':workspace,'layouts':str(root/'design/heldout-layouts.json'),'slot_start':start+16*i,'n':16})
    if condition!='clarified':
        for cell,key,task,target,offset in [('unseen','unseen',g['task'],g['goal'],32),
                ('competing','competing',g['competing_task'],g['competing_goal'],48)]:
            cells.append({'cell':cell,'condition':condition,'task':task,'goal':target,
                'input_prompt':g[key],'effective_prompt':g[key],'workspace':workspace,
                'layouts':str(root/'design/heldout-layouts.json'),'slot_start':start+offset,'n':16})
    return cells

def selection_cell(goal_id,condition,workspace):
    g=goal(goal_id)
    return {'cell':'selection','condition':condition,'task':g['task'],'goal':g['goal'],
        'input_prompt':g['original'],'effective_prompt':g['original'],'workspace':workspace,
        'layouts':str(goal_root(goal_id)/'design/development-layouts.json'),
        'slot_start':slot_start(goal_id,'development'),'n':16}

def validate_evaluation_plan(plan):
    validate_plan(plan);stage=plan['stage'];gid=plan['goal_id']
    if stage not in ('selection','heldout'): raise ValueError('Invalid evaluation stage')
    if plan['split']!=('development' if stage=='selection' else 'heldout'): raise ValueError('Invalid evaluation split')
    if len({(b['condition'],b['cell']) for b in plan['batches']})!=len(plan['batches']):
        raise ValueError('Duplicate condition/cell')
    for b in plan['batches']:
        expected=selection_cell(gid,b['condition'],b['workspace']) if stage=='selection' else next(
            (x for x in test_cells(gid,b['condition'],b['workspace']) if x['cell']==b['cell']),None)
        if b!=expected: raise ValueError('Cell differs from registered protocol')
        if b['condition'] in ('base','clarified') and b['workspace'] is not None:
            raise ValueError('Unchanged control has an intervention')

def seal_schedule(goal_id,stage,batches,artifacts,metadata):
    root=goal_root(goal_id);g=goal(goal_id);dest=root/f'{stage}-frozen'
    if (dest/'schedule.json').exists():
        existing=read(dest/'schedule.json')
        if existing['batches']!=batches: raise ValueError('Frozen schedule differs')
        for gpu in (0,1):
            plan=read(dest/f'plan-g{gpu}.json')
            from worker import verify_global_freeze
            verify_global_freeze(plan)
        return existing
    if dest.exists(): raise ValueError('Partial schedule preserved')
    files=[*native_sources(goal_id),HERE/'evaluate.py',HERE/'analyze.py',HERE/'prepare_baselines.py',
           *sorted((HERE/'baselines').glob('*.py')),*sorted((HERE/'adaptation').glob('*.py')),
           *sorted((HERE/'adaptation').glob('*.json')),*sorted((root/'design').glob('*-layouts.json'))]
    artifacts={**artifacts,**{str(p.resolve()):sha(p) for p in files}}
    conditions=list(dict.fromkeys(b['condition'] for b in batches));cell_index={'original_0':0,'original_1':1,'unseen':2,'competing':3}
    assigned=[(i%2 if stage=='selection' else (conditions.index(b['condition'])+cell_index[b['cell']])%2)
              for i,b in enumerate(batches)]
    schedule={'cohort_id':protocol()['cohort_id'],'goal_id':goal_id,'stage':stage,'batches':batches,
              'gpu_assignment':assigned,**metadata}
    for gpu in (0,1):
        plan={'cohort_id':protocol()['cohort_id'],'goal_id':goal_id,'case':g['runtime_case'],
            'runtime_dir':g['runtime_dir'],'stage':stage,'split':'development' if stage=='selection' else 'heldout',
            'inference_batching':'fixed16','protocol_sha256':sha(HERE/'protocol.json'),
            'schedule_sha256':canonical(schedule),'batches':[b for b,a in zip(batches,assigned,strict=True) if a==gpu]}
        validate_evaluation_plan(plan);freeze_plan(dest/f'plan-g{gpu}.json',plan,artifacts)
    save(dest/'schedule.json',schedule,exclusive=True);return schedule

def reset_artifacts(goal_id,split):
    import preflight
    path=goal_root(goal_id)/f'preflight/{split}/receipt.json';record=read(path)
    verify(record['artifacts'])
    if record['status']!='passed' or record['returncode']!=0: raise ValueError('Reset preflight not passed')
    preflight.validate_summary(read(path.parent/'summary.json'),goal_id,goal_root(goal_id)/f'design/{split}-layouts.json')
    return {**record['artifacts'],str(path):sha(path)}

def prepare_selection(goal_id):
    _,baselines,artifacts,source=registries(goal_id);artifacts.update(reset_artifacts(goal_id,'development'))
    batches=[selection_cell(goal_id,'base',None)]+[selection_cell(goal_id,c['candidate_id'],c['workspace']) for c in baselines]
    assert sum(b['n'] for b in batches)==208
    return seal_schedule(goal_id,'selection',batches,artifacts,{'candidates':baselines,
        'source_manifest':read(goal_root(goal_id)/'source-status.json')['manifest'],
        'source_manifest_sha256':read(goal_root(goal_id)/'source-status.json')['manifest_sha256']})

def select_candidates(candidates,groups):
    expected={'base',*(c['candidate_id'] for c in candidates)}
    if set(groups)!=expected: raise ValueError('Incomplete development candidate grid')
    reference=sorted((r['slot'],r['layout_id'],r['policy_seed']) for r in groups['base'])
    if len(reference)!=16 or len({x[0] for x in reference})!=16: raise ValueError('Incomplete development base')
    for rows in groups.values():
        if sorted((r['slot'],r['layout_id'],r['policy_seed']) for r in rows)!=reference:
            raise ValueError('Development candidates lack matched slots/layouts/noise')
    selected=[]
    for family in ('embedding','residual_g0','residual_g1','residual_g2'):
        options=[{**c,'selected_candidate_id':c['candidate_id'],'development_successes':sum(bool(r['success']) for r in groups[c['candidate_id']])}
                 for c in candidates if c['family']==family]
        options.append({'candidate_id':'base','selected_candidate_id':'base','family':family,
            'method':'embedding' if family=='embedding' else 'residual',
            'group_id':None if family=='embedding' else family.removeprefix('residual_'),
            'strength':0.,'workspace':None,'development_successes':sum(bool(r['success']) for r in groups['base']),
            'zero_fallback':True})
        winner=min(options,key=lambda c:(-c['development_successes'],c['strength']))
        selected.append({**winner,'candidate_id':family})
    return selected

def prepare_heldout(goal_id):
    from analyze import collect_stage
    edits,baselines,artifacts,source=registries(goal_id);artifacts.update(reset_artifacts(goal_id,'heldout'))
    groups,receipts,evidence=collect_stage(goal_id,'selection')
    selected=select_candidates(baselines,{k[0]:v for k,v in groups.items()})
    choice={'rule':protocol()['baselines']['selection'],'heldout_used':False,'candidates':selected,
        'all_development_counts':{k[0]:sum(bool(r['success']) for r in v) for k,v in groups.items()},
        'development_episodes':sum(r['n'] for r in receipts)}
    path=goal_root(goal_id)/'selection.json';save(path,choice,exclusive=True)
    artifacts.update({p:h for p,h in evidence.items() if not p.endswith('.npz')});artifacts[str(path)]=sha(path)
    candidates=edits+selected;conditions=[('base',None),('clarified',None)]+[(c['candidate_id'],c['workspace']) for c in candidates]
    cells={name:test_cells(goal_id,name,ws) for name,ws in conditions};batches=[]
    for index in range(4):
        order=conditions[index:]+conditions[:index]
        if index%2: order=list(reversed(order))
        for name,_ in order:
            if index<len(cells[name]):batches.append(cells[name][index])
    if sum(b['n'] for b in batches)!=544: raise ValueError('Held-out accounting differs')
    return seal_schedule(goal_id,'heldout',batches,artifacts,{'candidates':candidates,'source_groups':source['groups'],
        'source_manifest':read(goal_root(goal_id)/'source-status.json')['manifest'],
        'source_manifest_sha256':read(goal_root(goal_id)/'source-status.json')['manifest_sha256'],
        'primary':'Each three-pair edit versus base, embedding offset, and its source-matched residual offset.',
        'secondary':'Unseen wording and competing goal; no claim of broad equivalence from nonsignificant differences.'})

def freeze_all_heldout():
    summaries={};artifacts={}
    for gid in protocol()['goal_ids']:
        path=goal_root(gid)/'source-status.json';status=read(path);artifacts[str(path)]=sha(path)
        if status['status']=='qualified':
            prepare_heldout(gid);dest=goal_root(gid)/'heldout-frozen'
            summaries[gid]={'qualified':True,'planned_episodes':544}
            for p in sorted(dest.glob('*.json')):artifacts[str(p)]=sha(p)
        elif status['status']=='unqualified_at_cap': summaries[gid]={'qualified':False,'reason':status}
        else: raise ValueError('All registered goals must finish qualification first')
    value={'cohort_id':protocol()['cohort_id'],'status':'frozen','goals':summaries,
           'artifacts':artifacts,'planned_heldout_episodes':sum(x.get('planned_episodes',0) for x in summaries.values())}
    save(ROOT/'all-heldout-frozen.json',value,exclusive=True);return value

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('selection','heldout','freeze-all-heldout'))
    ap.add_argument('--goal',choices=protocol()['goal_ids']);args=ap.parse_args()
    if args.stage=='freeze-all-heldout':freeze_all_heldout()
    elif args.goal is None:ap.error('--goal is required')
    elif args.stage=='selection':prepare_selection(args.goal)
    else:prepare_heldout(args.goal)
