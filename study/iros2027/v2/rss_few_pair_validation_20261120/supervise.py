"""Continue a completed acquisition into gated fitting and a sealed comparison."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE,REPO,ROOT,goal_root,protocol,read,save,sha,verify

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collection-timeout-seconds',type=int,default=21600)
    args=ap.parse_args();ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'starting','started_unix':time.time(),'limits':protocol()['limits'],'code_sha256':sha(__file__)}
    def update(**kw):
        state.update(kw,updated_unix=time.time());save(ROOT/'supervisor-status.json',state)
    def stop(signum,frame):raise InterruptedError(f'Signal {signum}; partial artifacts preserved')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run_many(stage,commands,timeout=21600):
        children=[];streams=[];update(status='running',stage=stage,commands=commands)
        try:
            for i,command in enumerate(commands):
                stream=(ROOT/f'supervisor-{stage}-{i}.log').open('a');streams.append(stream)
                children.append(subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'),start_new_session=True))
            deadline=time.monotonic()+timeout
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError(stage+' failed; no automatic retry')
                if time.monotonic()>deadline:raise TimeoutError(stage+' exceeded technical timeout')
                time.sleep(3)
            if any(c.returncode for c in children):raise RuntimeError(stage+' failed; no automatic retry')
        finally:
            for c in children:
                if c.poll() is None:
                    try:os.killpg(c.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for c in children:
                if c.poll() is None:
                    try:c.wait(timeout=40)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(c.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
            for stream in streams:stream.close()
    def command(script,*args):return [sys.executable,str(HERE/script),*map(str,args)]
    def workers(gid,stage):
        if stage=='heldout':
            gate=read(ROOT/'all-heldout-frozen.json');verify(gate['artifacts'])
            if gate['status']!='frozen' or not gate['goals'][gid]['qualified']:raise ValueError('Global test freeze missing')
        commands=[]
        for gpu in (0,1):
            root=goal_root(gid)/stage/f'g{gpu}'
            if (root/'status.json').exists():
                if read(root/'status.json')['status']=='complete':continue
                raise ValueError('Partial worker preserved: '+str(root))
            commands.append(command('worker.py','--plan',goal_root(gid)/f'{stage}-frozen/plan-g{gpu}.json',
                                    '--root',root,'--gpu',gpu))
        if commands:run_many(gid+'-'+stage,commands,timeout=43200)
    try:
        update(status='waiting_for_collection',stage='source-qualification')
        deadline=time.monotonic()+args.collection_timeout_seconds
        while True:
            path=ROOT/'collection-status.json'
            if path.exists():
                status=read(path)['status']
                if status=='complete':break
                if status=='stopped':raise RuntimeError('Collection stopped; inspect preserved failure')
            if time.monotonic()>deadline:raise TimeoutError('Collection did not finish within declared wait')
            time.sleep(10)
        qualified=[];qualification={}
        for gid in protocol()['goal_ids']:
            status=read(goal_root(gid)/'source-status.json');qualification[gid]=status
            if status['status']=='qualified':qualified.append(gid)
            elif status['status']!='unqualified_at_cap':raise ValueError('Incomplete goal qualification')
        update(qualified_goals=qualified,qualification=qualification)
        # All goals and all source groups are determined before edited outcomes.
        for gid in qualified:
            root=goal_root(gid)
            if not (root/'adaptation/weightsmith/candidates.json').exists():
                run_many(gid+'-adaptation',[command('adaptation/run.py','--goal',gid,'--gpu',0)],timeout=14400)
            if not (root/'adaptation/baselines/candidates.json').exists():
                run_many(gid+'-baselines',[command('prepare_baselines.py','--goal',gid,'--gpu',0)],timeout=7200)
            run_many(gid+'-reset-checks',[
                command('preflight.py','--goal',gid,'--split','development','--gpu',0),
                command('preflight.py','--goal',gid,'--split','heldout','--gpu',1)],timeout=3600)
            run_many(gid+'-freeze-selection',[command('evaluate.py','selection','--goal',gid)],timeout=1800)
            workers(gid,'selection')
        # No goal's test results can affect another goal's selected candidates.
        run_many('freeze-all-heldout',[command('evaluate.py','freeze-all-heldout')],timeout=3600)
        for gid in qualified:
            workers(gid,'heldout')
            run_many(gid+'-analysis',[command('analyze.py','--goal',gid)],timeout=3600)
        run_many('aggregate-analysis',[command('analyze.py','--aggregate')],timeout=3600)
        report=ROOT/'analysis-v1/report.json';record=read(report)
        if record['status']!='complete_and_verified':raise ValueError('Final report not verified')
        update(status='complete_with_verified_report',stage='complete',report=str(report),report_sha256=sha(report),completed_unix=time.time())
    except BaseException as exc:
        update(status='stopped',reason=repr(exc));raise

if __name__=='__main__':main()
