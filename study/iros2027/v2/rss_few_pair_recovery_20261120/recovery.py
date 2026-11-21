"""Additive recovery records and process ownership; frozen study files stay intact."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'rss_few_pair_validation_20261120'
sys.path.insert(0,str(STUDY))
from common import REPO,ROOT as OLDROOT,goal_root,protocol,read,save,sha,verify
RECROOT=REPO/'projects/rss-few-pair-recovery-20261120'
AMENDMENT=RECROOT/'design/amendment.json'

def amendment():
    if AMENDMENT.exists():
        record=read(AMENDMENT);verify(record['preserved_stop_artifacts']);return record
    paths=[OLDROOT/'collection-status.json',OLDROOT/'supervisor-status.json',
        OLDROOT/'mustard_right/collection-status.json',
        *sorted((OLDROOT/'mustard_right/preflight/collection').glob('*.json')),
        STUDY/'protocol.json',HERE/'recovery.py']
    record={'cohort_id':protocol()['cohort_id'],'status':'authorized_recovery',
        'created_unix':time.time(),
        'user_request':'Test the new edits then dont sit around keep exploring until we make a solid case I KNOW theres a good paper here',
        'reason':'Mustard source preflight stopped before any mustard policy call; two goals already have fully qualified source manifests.',
        'changes':['Allow each qualified goal to fit and perform development selection while the remaining source goal undergoes technical reset diagnosis.',
                   'Keep all-goal candidate selection and the complete held-out schedules frozen before the first held-out episode.',
                   'Use an explicit, separately evidenced reset recovery if the original preflight lifecycle differs from actual fresh-process policy batches.'],
        'unchanged':['source pairs and triplet assignment','registered prompts and success predicates','edit recipe',
                     'baseline strengths and development selection','held-out tasks, layouts and noise','episode caps'],
        'partial_attempt_policy':'Retain every partial or failed attempt; no silent reruns, discarded policy outcomes, or tolerance relaxation.',
        'preserved_stop_artifacts':{str(p.resolve()):sha(p) for p in paths}}
    save(AMENDMENT,record,exclusive=True);return record

def command(script,*args):return [sys.executable,str(STUDY/script),*map(str,args)]

def run_commands(commands,log_dir,timeout=21600):
    log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    children=[];streams=[];started=time.time()
    try:
        for i,cmd in enumerate(commands):
            log=log_dir/f'process-{i}.log'
            if log.exists():raise ValueError('Preserved attempt already exists: '+str(log))
            stream=log.open('x');streams.append(stream)
            children.append(subprocess.Popen(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'),
                start_new_session=True))
        deadline=time.monotonic()+timeout
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Child failed; inspect '+str(log_dir))
            if time.monotonic()>deadline:raise TimeoutError('Technical timeout: '+str(log_dir))
            time.sleep(3)
        if any(p.returncode for p in children):raise RuntimeError('Child failed; inspect '+str(log_dir))
        save(log_dir/'completed.json',{'status':'complete','commands':commands,'wall_seconds':time.time()-started},exclusive=True)
    finally:
        for p in children:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
        for stream in streams:stream.close()

def stop_on_signal(signum,frame):raise InterruptedError(f'Signal {signum}; evidence retained')

def worker_commands(gid,stage):
    commands=[]
    for gpu in (0,1):
        root=goal_root(gid)/stage/f'g{gpu}'
        if (root/'status.json').exists():
            if read(root/'status.json')['status']=='complete':continue
            raise ValueError('Partial worker retained; separate recovery required: '+str(root))
        plan=goal_root(gid)/f'{stage}-frozen/plan-g{gpu}.json'
        from worker import verify_global_freeze
        verify_global_freeze(read(plan))
        commands.append(command('worker.py','--plan',plan,'--root',root,'--gpu',gpu))
    return commands
