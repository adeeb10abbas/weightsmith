"""Select measured development phases for the original 96-key spatial recipe."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parents[1]/'rss_goal_expansion_v2/adaptation/prepare.py'
sp=importlib.util.spec_from_file_location('qualified_development_readers',LEGACY)
readers=importlib.util.module_from_spec(sp);sp.loader.exec_module(readers)
sha=readers.sha
helpers=readers.helpers

class CoverageError(ValueError):
    """The fixed source budget cannot be filled without changing the rule."""

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def phase_indices(steps,gripped,clearance,*,failed_original=False):
    steps=np.asarray(steps,dtype=int)
    grip=np.asarray(gripped,dtype=bool);height=np.asarray(clearance,dtype=float)
    if not len(steps) or np.any(np.diff(steps)<=0) or steps[0]!=0:raise ValueError('Invalid policy step index')
    if grip.shape!=height.shape or not np.isfinite(height).all() or steps[-1]>=len(grip):raise ValueError('Phase history does not cover policy steps')
    contact=np.flatnonzero(grip)
    result={}
    if len(contact):
        event=int(contact[0]);k=int(np.searchsorted(steps,event,side='left')-1)
        if k<0:raise ValueError('Object already gripped before the first policy decision')
        result['grasp']=k
        if k>0:result['precontact']=k-1
        transport=np.flatnonzero(grip[steps]&(height[steps]>=.025))
        if len(transport):result['transport']=int(transport[0])
    elif failed_original:
        result['precontact']=len(steps)-1
    return result

def prepare(case,protocol_path,teacher_batch,preservation_batch,output):
    protocol_path=Path(protocol_path).resolve();protocol=json.loads(protocol_path.read_text());cfg=protocol['cases'][case]
    teacher_batch=Path(teacher_batch).resolve();gatepath=teacher_batch.parents[1]/'qualified.json'
    gate=json.loads(gatepath.read_text())
    if gate['case']!=case or gate['teacher_batch']!=str(teacher_batch):raise ValueError('Wrong qualification source')
    items=[];observations={};metrics_hashes={};receipts={};owners={};coverage={};used=set()
    for split,round_,needed in [('train',gate['rounds'][0],6),('validation',gate['rounds'][1],2)]:
        pools={phase:[] for phase in ('precontact','grasp','transport')}
        for cell in ('original','teacher'):
            batch=Path(round_[cell]);rows,layout_ids,owner_hashes=readers.read_batch(batch,case,cell,cfg[cell])
            context_index=helpers.live_context_index(batch,rows)
            receipts[str(batch)]=sha(batch/'completed.json');owners.update(owner_hashes)
            for row in sorted(rows,key=lambda r:r['slot']):
                eligible=(cell=='original' and not row['success']) or (cell=='teacher' and row['success'])
                if not eligible:continue
                path=Path(row['obs']).resolve();metricpath=Path(row['metrics']).resolve()
                if sha(path)!=row['obs_sha256'] or sha(metricpath)!=row['metrics_sha256']:raise ValueError('Source data changed')
                metrics=json.loads(metricpath.read_text());history=metrics['phase_history']
                if [h['policy_step'] for h in history]!=list(range(len(history))):raise ValueError('Incomplete phase history')
                env=row['env']
                with np.load(path,allow_pickle=False) as z:steps=z['step'].copy()
                phases=phase_indices(steps,[h['mover_gripped'][env] for h in history],
                    [h['mover_height_above_reset_m'][env] for h in history],failed_original=cell=='original')
                for phase,k in phases.items():
                    if phase!='precontact' and cell!='teacher':continue
                    pools[phase].append((0 if cell=='original' else 1,row['slot'],row,k,context_index,path,metricpath,cell))
        coverage[split]={phase:len(pool) for phase,pool in pools.items()}
        for phase,pool in pools.items():
            selected=[];episodes=set()
            for _,_,row,k,context_index,path,metricpath,cell in sorted(pool,key=lambda x:x[:2]):
                identity=(str(path),k)
                if str(path) in episodes or identity in used:continue
                selected.append((row,k,context_index,path,metricpath,cell));episodes.add(str(path));used.add(identity)
                if len(selected)==needed:break
            if len(selected)!=needed:
                raise CoverageError(f'{case}: {split}/{phase} needs {needed} distinct source episodes; found {len(selected)}; coverage={coverage}')
            for row,k,context_index,path,metricpath,cell in selected:
                context=helpers.live_context(context_index,row,k)
                context['live_batch_bucket']=16;context['inference_batching']='fixed16'
                context['live_context']['inference_batching']='fixed16'
                context['live_context_sha256']=canonical(context['live_context'])
                items.append({'item_id':f'{case}-{split}-{phase}-{cell}-{row["slot"]}-k{k}',
                    'obs_path':str(path),'decision_index':k,'split':split,'role':'repair','phase':phase,
                    'layout_id':row['layout_id'],'source_cell':cell,'source_slot':row['slot'],
                    'original_prompt':cfg['original'],'teacher_prompt':cfg['teacher'],
                    'recorded_prompt':cfg[cell],**context})
                observations[str(path)]=row['obs_sha256'];metrics_hashes[str(metricpath)]=row['metrics_sha256']
    counts={f'{split}_{phase}':sum(x['split']==split and x['phase']==phase for x in items)
            for split in ('train','validation') for phase in ('precontact','grasp','transport')}
    result={'cohort_id':protocol['cohort']+'-'+case+'-adaptation-v1','case':case,'split':'development',
        'inference_batching':'fixed16','reuse_or_new':'new','selection_rule':protocol['adaptation']['source_selection'],
        'privileged_annotation':protocol['adaptation']['annotation'],'protocol_sha256':sha(protocol_path),
        'selector_sha256':sha(__file__),'context_helper_sha256':sha(readers.ORIGINAL),
        'source_observation_sha256':observations,'phase_metric_sha256':metrics_hashes,
        'source_receipt_sha256':receipts,'source_owner_sha256':owners,'counts':counts,'coverage':coverage,
        'preservation_examples_used_in_fit':0,'items':items}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('case','protocol','teacher-batch','preservation-batch','output'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();v=prepare(a.case,a.protocol,a.teacher_batch,a.preservation_batch,a.output)
    print(json.dumps({'case':a.case,'counts':v['counts']}))
