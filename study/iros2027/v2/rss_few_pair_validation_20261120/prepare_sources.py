"""Verify fresh paired receipts; freeze disjoint triplets without using edited outcomes."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common import HERE, PRIOR, canonical, goal, goal_index, goal_root, load, protocol, read, save, sha, slot_start, verify
from goals.selection import select_failure_decision

def read_batch(goal_id,block,cell):
    p=protocol();g=goal(goal_id);owner=goal_root(goal_id)/f'collection/block-{block:02d}'/cell
    status=read(owner/'status.json');plan=read(owner/'plan.json')
    if status.get('status')!='complete' or status.get('completed_episodes')!=16 or status.get('completed_batches')!=1:
        raise ValueError(f'Incomplete source batch: {owner}')
    start=slot_start(goal_id,'collection')+16*block
    if plan['batches'][0]['slot_start']!=start or plan['cohort_id']!=p['cohort_id'] or plan['goal_id']!=goal_id or plan['stage']!='collection':
        raise ValueError('Wrong source plan')
    verify(read(owner/'identity.json'))
    batch=owner/f'batch-00-{cell}-base';receipt=read(batch/'completed.json');rows=read(batch/'episodes.json')
    if receipt['n']!=16 or receipt['episodes_sha256']!=sha(batch/'episodes.json'):
        raise ValueError('Source receipt identity changed')
    if len(rows)!=16 or {r['slot'] for r in rows}!=set(range(start,start+16)):
        raise ValueError('Source slots missing or duplicated')
    prompt=g['original'] if cell=='original' else g['teacher']
    for row in rows:
        expected={'cohort_id':p['cohort_id'],'goal_id':goal_id,'split':'development','condition':'base',
                  'inference_batching':'fixed16','task_name':g['task'],'selected_goal':g['goal'],
                  'effective_prompt':prompt,'ok':True,'cell':cell}
        if any(row.get(k)!=v for k,v in expected.items()): raise ValueError('Source row metadata mismatch')
        if not Path(row['obs']).resolve().is_relative_to(batch): raise ValueError('Foreign observation')
        if sha(row['obs'])!=row['obs_sha256'] or sha(row['metrics'])!=row['metrics_sha256']:
            raise ValueError('Source observation/metric changed')
        if bool(row['strict_final'][g['goal']])!=bool(row['success']): raise ValueError('Source score disagreement')
    return owner,batch,sorted(rows,key=lambda r:r['slot'])

def eligibility(goal_id,original,teacher):
    for key in ('slot','layout_id','policy_seed'):
        if original[key]!=teacher[key]: raise ValueError('Nominal pair mismatch')
    reasons=[]
    if original['success']: reasons.append('original_succeeded')
    if not teacher['success']: reasons.append('clarified_failed')
    with np.load(original['obs'],allow_pickle=False) as z:
        decision=select_failure_decision(goal_id,z['actions'],z['gripper'])
    if decision is None: reasons.append('no_registered_failure_state')
    return (None if reasons else int(decision)),reasons

def partition_pairs(pair_ids,seed):
    if len(pair_ids)!=9 or len(set(pair_ids))!=9: raise ValueError('Exactly9distinct source pairs required')
    order=np.random.default_rng(seed).permutation(9).tolist()
    return [{'group_id':f'g{i}','indices':order[3*i:3*i+3],
             'pair_ids':[pair_ids[k] for k in order[3*i:3*i+3]]} for i in range(3)]

def export(goal_id,blocks):
    p=protocol();g=goal(goal_id);root=goal_root(goal_id)
    if not 1<=blocks<=p['collection']['max_pairs_per_goal']//16: raise ValueError('Source cap exceeded')
    attempts=[];eligible=[];artifacts={}
    for block in range(blocks):
        batches={};rows={}
        for cell in ('original','teacher'):
            owner,batches[cell],rows[cell]=read_batch(goal_id,block,cell)
            for path in [owner/'status.json',owner/'identity.json',owner/'plan.json',batches[cell]/'episodes.json',batches[cell]/'completed.json']:
                artifacts[str(path)]=sha(path)
        for original,teacher in zip(rows['original'],rows['teacher'],strict=True):
            decision,reasons=eligibility(goal_id,original,teacher)
            rec={'pair_id':f'{goal_id}-{original["slot"]}','slot':original['slot'],'layout_id':original['layout_id'],
                 'eligible':decision is not None,'selected_decision':decision,'rejection_reasons':reasons,
                 'original':original,'teacher':teacher,**{f'{k}_batch':str(v) for k,v in batches.items()}}
            attempts.append(rec)
            if decision is not None: eligible.append(rec)
    selected=eligible[:9];qualified=len(selected)==9
    dest=root/f'sources/block-{blocks:02d}';manifest_path=dest/'source_manifest.json'
    grouping_seed=p['collection']['grouping_seed']+goal_index(goal_id)
    groups=partition_pairs([r['pair_id'] for r in selected],grouping_seed) if qualified else []
    manifest={'cohort_id':p['cohort_id'],'goal_id':goal_id,'case':g['runtime_case'],'split':'development',
              'protocol_sha256':sha(HERE/'protocol.json'),'fresh_collection':True,'attempted_pairs':len(attempts),
              'policy_episodes':2*len(attempts),'eligible_pairs':len(eligible),'qualified':qualified,
              'at_collection_cap':len(attempts)==p['collection']['max_pairs_per_goal'],
              'attempts':attempts,'pairs':selected,'groups':groups,'grouping_seed':grouping_seed,
              'spec':str(dest/'correction.json'),'artifacts':artifacts,'nominal_pairing_only':True}
    save(manifest_path,manifest,exclusive=True)
    if qualified:
        helper=load('fewpair_live_context',PRIOR/'prepare_shared.py');indices={};items=[]
        for pair in selected:
            row=pair['original'];batch=Path(pair['original_batch']);k=pair['selected_decision']
            if str(batch) not in indices: indices[str(batch)]=helper.live_context_index(batch,read(batch/'episodes.json'))
            context=helper.live_context(indices[str(batch)],row,k)
            context['live_batch_bucket']=16;context['inference_batching']='fixed16'
            context['live_context']['inference_batching']='fixed16';context['live_context_sha256']=canonical(context['live_context'])
            items.append({'item_id':f'{pair["pair_id"]}-original-k{k}','pair_id':pair['pair_id'],
                'obs_path':row['obs'],'decision_index':k,'split':'train','role':'repair','layout_id':row['layout_id'],
                'source_slot':row['slot'],'source_cell':'original','original_prompt':g['original'],
                'teacher_prompt':g['teacher'],'recorded_prompt':row['effective_prompt'],**context})
        spec={'cohort_id':p['cohort_id'],'goal_id':goal_id,'case':g['runtime_case'],'split':'development',
              'inference_batching':'fixed16','source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path),
              'protocol_sha256':sha(HERE/'protocol.json'),'correction_count':9,'distinct_pair_ids':[r['pair_id'] for r in selected],
              'arm':'correction','original_prompt':g['original'],'teacher_prompt':g['teacher'],'items':items,
              'groups':groups,'selection':'First9eligible; frozen random partition into3disjointtriplets; no edited outcomes.'}
        save(dest/'correction.json',spec,exclusive=True)
    status={'status':'qualified' if qualified else 'unqualified_at_cap' if manifest['at_collection_cap'] else 'collecting',
            'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'attempted_pairs':len(attempts),
            'eligible_pairs':len(eligible),'qualified':qualified}
    save(root/'source-status.json',status)
    return status

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--goal',required=True);ap.add_argument('--blocks',type=int,required=True)
    args=ap.parse_args();print(export(args.goal,args.blocks))
