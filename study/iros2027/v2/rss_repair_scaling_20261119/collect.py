"""Seal then collect bounded fresh paired episodes; never retries a partial batch."""
from __future__ import annotations
import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from common import HERE,ROOT,REPO,canonical,layouts,protocol,read,save,sha,source_inventory
from prepare_sources import export

def prepare():
    p=protocol();root=ROOT/'design'
    names={'collection':'collection','near':'development','far':'changed'}
    for kind,name in names.items():
        save(root/f'{name}-layouts.json',layouts(kind),exclusive=True)
    all_layouts=[r['poses'] for name in names.values() for r in read(root/f'{name}-layouts.json')['layouts']]
    if len(set(canonical(x) for x in all_layouts))!=len(all_layouts):raise ValueError('Overlapping layout definitions')
    files=source_inventory()+[root/'collection-layouts.json']
    seal={'cohort_id':p['cohort_id'],'protocol_sha256':sha(HERE/'protocol.json'),
          'stage':'source_collection','artifacts':{str(f.resolve()):sha(f) for f in files},
          'max_pairs':p['collection']['max_pairs'],'max_policy_episodes':2*p['collection']['max_pairs'],
          'created_unix':time.time()}
    path=root/'collection-seal.json'
    if path.exists():
        previous=read(path)
        if previous['artifacts']!=seal['artifacts']:raise ValueError('Source collection already sealed with different inputs')
    else:save(path,seal,exclusive=True)
    print({'collection_seal':str(path),'max_policy_episodes':seal['max_policy_episodes']},flush=True)

def plan(block,cell):
    p=protocol();seal=read(ROOT/'design/collection-seal.json')
    for path,h in seal['artifacts'].items():
        if sha(path)!=h:raise ValueError(f'Sealed collection source changed: {path}')
    owner=ROOT/f'collection/block-{block:02d}'/cell
    b={'cell':cell,'condition':'base','task':p['task'],'goal':p['goal'],'input_prompt':p['original_prompt'],
       'effective_prompt':p['original_prompt'] if cell=='original' else p['clarified_prompt'],'workspace':None,
       'layouts':str(ROOT/'design/collection-layouts.json'),'slot_start':p['collection']['slot_start']+16*block,'n':16}
    value={'cohort_id':p['cohort_id'],'split':'development','case':'temporal','inference_batching':'fixed16',
           'protocol_sha256':sha(HERE/'protocol.json'),'batches':[b]}
    manifest={'plan_sha256':{'temporal':canonical(value)},'artifacts':seal['artifacts']}
    global_path=ROOT/f'design/block-{block:02d}-{cell}-freeze.json';save(global_path,manifest,exclusive=True)
    value['global_freeze']={'path':str(global_path),'sha256':sha(global_path)}
    path=ROOT/f'design/block-{block:02d}-{cell}-plan.json';save(path,value,exclusive=True)
    return path,owner

def run(args):
    p=protocol();max_blocks=args.max_blocks
    if not 1<=max_blocks<=p['collection']['max_pairs']//16:raise ValueError('Requested collection exceeds frozen cap')
    if sorted(args.gpus)!=[0,1]:raise ValueError('Parallel paired collection requires exactly GPUs0 and1')
    lock=(ROOT/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=[]
    state={'status':'starting','max_blocks':max_blocks,'max_pairs':16*max_blocks,'started_unix':time.time()}
    try:
        for block in range(max_blocks):
            processes=[];logs=[]
            for cell,gpu in zip(('original','teacher'),args.gpus,strict=True):
                path,owner=plan(block,cell)
                if (owner/'status.json').exists() and read(owner/'status.json').get('status')=='complete':continue
                owner.parent.mkdir(parents=True,exist_ok=True)
                log=(owner.parent/f'{cell}.log').open('a');logs.append(log)
                cmd=[sys.executable,str(HERE/'worker.py'),'--plan',str(path),'--root',str(owner),'--gpu',str(gpu)]
                processes.append(subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT))
            state.update(status='running',block=block,pids=[p.pid for p in processes]);save(ROOT/'collection-status.json',state)
            # Fail promptly if either paired arm fails; keep all partial artifacts.
            while any(q.poll() is None for q in processes):
                if any(q.poll() not in (None,0) for q in processes):raise RuntimeError('A source worker failed; inspect preserved original/teacher logs')
                time.sleep(3)
            if any(q.returncode for q in processes):raise RuntimeError('Source worker failed')
            for log in logs:log.close()
            result=export(block+1)
            state.update(completed_pairs=result['attempted_pairs'],eligible_pairs=result['eligible_pairs']);save(ROOT/'collection-status.json',state)
            if result['eligible_pairs']>=max(p['budgets']):break
        state.update(status='complete',completed_unix=time.time());save(ROOT/'collection-status.json',state)
    except BaseException as exc:
        for q in processes:
            if q.poll() is None:q.terminate()
        for q in processes:
            if q.poll() is None:
                try:q.wait(timeout=40)
                except subprocess.TimeoutExpired:q.kill()
        state.update(status='stopped',reason=repr(exc));save(ROOT/'collection-status.json',state)
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');p=sub.add_parser('run');p.add_argument('--gpus',nargs=2,type=int,default=[0,1]);p.add_argument('--max-blocks',type=int,default=1)
    args=ap.parse_args();prepare() if args.command=='prepare' else run(args)

if __name__=='__main__':main()
