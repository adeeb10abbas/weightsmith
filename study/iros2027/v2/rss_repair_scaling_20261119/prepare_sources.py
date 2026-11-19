"""Audit sealed source episodes and export nested correction/full-trajectory sets."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common import HERE, ROOT, PRIOR, canonical, load, protocol, read, save, sha
from weightsmith.distill import select_early_releases

def read_source_batch(owner, cell, block):
    owner = Path(owner); plan = read(owner / 'plan.json'); status = read(owner / 'status.json')
    p = protocol(); start = p['collection']['slot_start'] + 16 * block
    if status.get('status') != 'complete' or status.get('completed_episodes') != 16 or status.get('completed_batches') != 1:
        raise ValueError(f'Incomplete source owner: {owner}')
    batch = owner / f'batch-00-{cell}-base'
    receipt = read(batch / 'completed.json'); rows = read(batch / 'episodes.json')
    if receipt['n'] != 16 or receipt['episodes_sha256'] != sha(batch / 'episodes.json'):
        raise ValueError('Source receipt mismatch')
    if plan.get('split') != 'development' or plan.get('cohort_id') != p['cohort_id'] or plan['batches'][0]['slot_start'] != start:
        raise ValueError('Source cohort or slots differ')
    expected_prompt = p['original_prompt'] if cell == 'original' else p['clarified_prompt']
    if len(rows) != 16 or {r['slot'] for r in rows} != set(range(start, start+16)):
        raise ValueError('Source slots incomplete or duplicated')
    identity = read(owner / 'identity.json')
    for path, checksum in identity.items():
        if sha(path) != checksum: raise ValueError(f'Source runtime changed: {path}')
    for row in rows:
        if any(row.get(k) != v for k,v in {'cohort_id':p['cohort_id'],'split':'development','case':'temporal',
              'cell':cell,'condition':'base','effective_prompt':expected_prompt,'task_name':p['task'],
              'inference_batching':'fixed16','ok':True,'selected_goal':p['goal']}.items()):
            raise ValueError('Source episode metadata differs')
        if not Path(row['obs']).resolve().is_relative_to(batch): raise ValueError('Foreign source observation')
        if sha(row['obs']) != row['obs_sha256'] or sha(row['metrics']) != row['metrics_sha256']:
            raise ValueError('Source observation or metrics changed')
        if row['strict_final'][p['goal']] != row['success']: raise ValueError('Source scoring disagreement')
    return batch, sorted(rows,key=lambda r:r['slot'])

def eligible(original, teacher):
    if original['slot'] != teacher['slot'] or original['layout_id'] != teacher['layout_id'] or original['policy_seed'] != teacher['policy_seed']:
        raise ValueError('Nominal pair identities differ')
    reasons=[]
    if original['success']: reasons.append('original_succeeded')
    if not teacher['success']: reasons.append('clarified_failed')
    d=protocol()['collection']['decision']
    with np.load(original['obs'],allow_pickle=False) as z:
        chosen=select_early_releases(z['actions'],z['gripper'],max_hold=d['max_hold'],min_grip=d['min_grip'],max_grip=d['max_grip'],first_only=True)
    if not chosen: reasons.append('no_early_release_decision')
    return (chosen[0] if not reasons else None),reasons

def inspect(max_blocks):
    p=protocol(); attempts=[]; pairs=[]; artifacts={}
    if not 1 <= max_blocks <= p['collection']['max_pairs']//16: raise ValueError('Collection cap exceeded')
    for block in range(max_blocks):
        owners={cell:ROOT/f'collection/block-{block:02d}'/cell for cell in ('original','teacher')}
        batches={}; rows={}
        for cell in owners:
            batches[cell],rows[cell]=read_source_batch(owners[cell],cell,block)
            for name in ('episodes.json','completed.json'):
                path=batches[cell]/name; artifacts[str(path)]=sha(path)
            for name in ('plan.json','status.json','identity.json'):
                path=owners[cell]/name; artifacts[str(path)]=sha(path)
        for original,teacher in zip(rows['original'],rows['teacher'],strict=True):
            decision,reasons=eligible(original,teacher)
            rec={'pair_id':f'temporal-{original["slot"]}','slot':original['slot'],'layout_id':original['layout_id'],
                 'eligible':decision is not None,'rejection_reasons':reasons,'selected_decision':decision,
                 'original':original,'teacher':teacher,'original_batch':str(batches['original']),
                 'teacher_batch':str(batches['teacher'])}
            attempts.append(rec)
            if decision is not None: pairs.append(rec)
    return {'cohort_id':p['cohort_id'],'split':'development','protocol_sha256':sha(HERE/'protocol.json'),
            'blocks':max_blocks,'attempted_pairs':len(attempts),'policy_episodes':2*len(attempts),
            'eligible_pairs':len(pairs),'correction_budgets_available':[n for n in p['budgets'] if n<=len(pairs)],
            'at_collection_cap':len(attempts)==p['collection']['max_pairs'],
            'correction_unit':p['correction_unit'],'attempts':attempts,'pairs':pairs[:max(p['budgets'])],
            'artifacts':artifacts,'nominal_pairing_only':True}

def export(max_blocks):
    result=inspect(max_blocks); out=ROOT/f'sources/block-{max_blocks:02d}'
    result['specs']={str(n):{arm:str(out/f'n{n}/{arm}.json') for arm in ('correction','full_trajectory')}
                     for n in result['correction_budgets_available']}
    save(out/'source_manifest.json',result,exclusive=True)
    helper=load('scaling_live_context',PRIOR/'prepare_shared.py'); indices={}
    p=protocol()
    def item(pair,cell,k):
        row=pair[cell]; batch=Path(pair[cell+'_batch'])
        if str(batch) not in indices:
            indices[str(batch)]=helper.live_context_index(batch,read(batch/'episodes.json'))
        context=helper.live_context(indices[str(batch)],row,k)
        context['live_batch_bucket']=16
        context['inference_batching']='fixed16'
        context['live_context']['inference_batching']='fixed16'
        context['live_context_sha256']=canonical(context['live_context'])
        return {'item_id':f'{pair["pair_id"]}-{cell}-k{k}','pair_id':pair['pair_id'],
                'obs_path':row['obs'],'decision_index':k,'split':'train','role':'repair',
                'layout_id':row['layout_id'],'source_slot':row['slot'],'source_cell':cell,
                'original_prompt':p['original_prompt'],'teacher_prompt':p['clarified_prompt'],
                'recorded_prompt':row['effective_prompt'],**context}
    for n in result['correction_budgets_available']:
        pairs=result['pairs'][:n]
        for arm in ('correction','full_trajectory'):
            items=[]
            for pair in pairs:
                if arm=='correction': items.append(item(pair,'original',pair['selected_decision']))
                else:
                    for k in range(pair['teacher']['decision_count']): items.append(item(pair,'teacher',k))
            spec={'cohort_id':p['cohort_id'],'case':'temporal','split':'development','inference_batching':'fixed16',
                  'source_manifest':str(out/'source_manifest.json'),'source_manifest_sha256':sha(out/'source_manifest.json'),
                  'protocol_sha256':sha(HERE/'protocol.json'),'correction_count':n,'distinct_pair_ids':[x['pair_id'] for x in pairs],
                  'arm':arm,'original_prompt':p['original_prompt'],'teacher_prompt':p['clarified_prompt'],
                  'items':items,'selection':'Nested first eligible source episode pairs. Every source item is training; development selection is separate closed-loop rollouts.'}
            save(out/f'n{n}/{arm}.json',spec,exclusive=True)
    save(ROOT/'source-status.json',{'status':'sources_exported','manifest':str(out/'source_manifest.json'),
         'manifest_sha256':sha(out/'source_manifest.json'),'attempted_pairs':result['attempted_pairs'],
         'eligible_pairs':result['eligible_pairs'],'budgets_available':result['correction_budgets_available']})
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=('inspect','export'))
    ap.add_argument('--max-blocks',type=int,required=True);args=ap.parse_args()
    result=export(args.max_blocks) if args.command=='export' else inspect(args.max_blocks)
    print({k:result[k] for k in ('attempted_pairs','policy_episodes','eligible_pairs','correction_budgets_available')},flush=True)

if __name__=='__main__': main()
