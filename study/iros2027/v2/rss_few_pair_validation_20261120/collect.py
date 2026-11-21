"""Fresh bounded paired acquisition, sealed before execution, without automatic retries."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE,ROOT,REPO,canonical,freeze_plan,goal,goal_index,goal_root,native_sources,protocol,read,save,sha,slot_start,verify
from goals import build_layouts
from prepare_sources import export

def prepare():
    p=protocol();reports=[]
    for goal_id in p['goal_ids']:
        root=goal_root(goal_id);g=goal(goal_id);design=root/'design'
        definitions=[]
        for split,n,seed in (
            ('collection',48,p['collection']['generator_seed']),
            ('development',16,p['evaluation']['development_seed']),
            ('heldout',16,p['evaluation']['heldout_seed'])):
            spec=build_layouts(goal_id,split,n=n,seed=seed+100*goal_index(goal_id))
            path=design/f'{split}-layouts.json';save(path,spec,exclusive=True)
            definitions.extend(canonical(x['poses']) for x in spec['layouts'])
        if len(set(definitions))!=len(definitions): raise ValueError('Source/development/test layout overlap')
        files=[*native_sources(goal_id),HERE/'collect.py',HERE/'prepare_sources.py',HERE/'preflight.py',*design.glob('*-layouts.json')]
        artifacts={str(f.resolve()):sha(f) for f in sorted(set(files))}
        seal={'cohort_id':p['cohort_id'],'goal_id':goal_id,'protocol_sha256':sha(HERE/'protocol.json'),
              'stage':'source_collection','artifacts':artifacts,'max_pairs':48,'max_policy_episodes':96}
        save(design/'collection-seal.json',seal,exclusive=True)
        reports.append({'goal':goal_id,'seal':str(design/'collection-seal.json')})
    save(ROOT/'design-prepared.json',{'protocol_sha256':sha(HERE/'protocol.json'),'goals':reports},exclusive=True)
    return reports

def plan(goal_id,block,cell):
    p=protocol();g=goal(goal_id);root=goal_root(goal_id)
    seal=read(root/'design/collection-seal.json');verify(seal['artifacts'])
    if not 0<=block<3 or cell not in ('original','teacher'): raise ValueError('Invalid source batch')
    spec={'cell':cell,'condition':'base','task':g['task'],'goal':g['goal'],
          'input_prompt':g['original'],'effective_prompt':g['original'] if cell=='original' else g['teacher'],
          'workspace':None,'layouts':str(root/'design/collection-layouts.json'),
          'slot_start':slot_start(goal_id,'collection')+16*block,'n':16}
    value={'cohort_id':p['cohort_id'],'goal_id':goal_id,'case':g['runtime_case'],'runtime_dir':g['runtime_dir'],
           'stage':'collection','split':'development','inference_batching':'fixed16',
           'protocol_sha256':sha(HERE/'protocol.json'),'batches':[spec]}
    path=freeze_plan(root/f'design/block-{block:02d}-{cell}-plan.json',value,seal['artifacts'])
    return path,root/f'collection/block-{block:02d}'/cell

def stop_children(processes):
    for process in processes:
        if process.poll() is None:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for process in processes:
        if process.poll() is None:
            try: process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass

def run_goal(goal_id,timeout_seconds=5400):
    root=goal_root(goal_id);state={'goal_id':goal_id,'status':'starting','started_unix':time.time()}
    processes=[];logs=[]
    try:
        import preflight
        state.update(status='checking_collection_resets');save(root/'collection-status.json',state)
        preflight.run(goal_id,'collection',1)
        for block in range(3):
            processes=[];logs=[]
            for gpu,cell in enumerate(('original','teacher')):
                path,owner=plan(goal_id,block,cell)
                if (owner/'status.json').exists() and read(owner/'status.json')['status']=='complete': continue
                owner.parent.mkdir(parents=True,exist_ok=True)
                log=(owner.parent/f'{cell}.log').open('a');logs.append(log)
                cmd=[sys.executable,str(HERE/'worker.py'),'--plan',str(path),'--root',str(owner),'--gpu',str(gpu)]
                processes.append(subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            state.update(status='running',block=block,pids=[x.pid for x in processes]);save(root/'collection-status.json',state)
            deadline=time.monotonic()+timeout_seconds
            while any(x.poll() is None for x in processes):
                if any(x.poll() not in (None,0) for x in processes): raise RuntimeError('Source worker failed; no automatic retry')
                if time.monotonic()>deadline: raise TimeoutError('Source block exceeded declared technical timeout')
                time.sleep(3)
            if any(x.returncode for x in processes): raise RuntimeError('Source worker failed')
            for log in logs: log.close()
            status=export(goal_id,block+1);state.update(status)
            save(root/'collection-status.json',state)
            print({'goal_id':goal_id,**status},flush=True)
            if status['qualified'] or status['status']=='unqualified_at_cap': break
        state.update(status='complete',qualification=read(root/'source-status.json')['status'],completed_unix=time.time())
        save(root/'collection-status.json',state)
    except BaseException as exc:
        stop_children(processes)
        for log in logs: log.close()
        state.update(status='stopped',reason=repr(exc));save(root/'collection-status.json',state);raise
    return state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--goal',choices=protocol()['goal_ids']);args=ap.parse_args()
    if args.command=='prepare': print(prepare());return
    if not (ROOT/'design-prepared.json').exists(): raise RuntimeError('Prepare and audit the frozen design first')
    lock=(ROOT/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    state={'status':'running','started_unix':time.time(),'goals':{}}
    try:
        for goal_id in ([args.goal] if args.goal else protocol()['goal_ids']):
            state['current_goal']=goal_id;save(ROOT/'collection-status.json',state)
            state['goals'][goal_id]=run_goal(goal_id);save(ROOT/'collection-status.json',state)
        state.update(status='complete',completed_unix=time.time())
    except BaseException as exc:
        state.update(status='stopped',reason=repr(exc));raise
    finally:save(ROOT/'collection-status.json',state)

if __name__=='__main__':main()
