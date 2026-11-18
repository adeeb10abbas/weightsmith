"""Durable preflight -> qualification -> adaptation -> frozen test -> report."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from worker import digest,save

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpu',type=int,choices=(1,),required=True)
    args=ap.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=[*HERE.rglob('*.py'),HERE/'protocol.json',*sorted((HERE/'runtime/layouts').glob('*.json'))]
    identity={str(p):digest(p) for p in files}
    seal=root/'supervisor-source-freeze.json'
    if seal.exists() and json.loads(seal.read_text())!=identity:raise RuntimeError('Supervisor source changed; preserve existing cohort')
    save(seal,identity)
    state={'status':'starting','started_at':time.time(),'gpu':args.gpu,'root':str(root),'completed_stages':[]}
    try:
        stages=[('preflight','preflight.py',root/'preflight/complete.json'),
                ('qualification','qualify.py',root/'qualification-status.json'),
                ('adaptation-and-test','advance.py',root/'advance-status.json')]
        for name,script,receipt in stages:
            if any(digest(p)!=h for p,h in identity.items()):raise RuntimeError('Source changed after study freeze')
            if receipt.exists():
                status=json.loads(receipt.read_text())
                if status.get('status')=='complete':
                    state['completed_stages'].append({'stage':name,'reused_complete':True});continue
                if status.get('status') in ('stopped','failed'):raise RuntimeError('Preserved failed stage '+name)
            state.update(status='running',stage=name);save(root/'study-status.json',state)
            began=time.time();print('START '+name,flush=True)
            subprocess.run([sys.executable,str(HERE/script),'--root',str(root),'--gpu',str(args.gpu)],cwd=REPO,check=True)
            state['completed_stages'].append({'stage':name,'wall_seconds':time.time()-began})
            print('DONE '+name,flush=True)
        state.update(status='complete',report=str(root/'analysis-v1/report.json'))
    except BaseException as exc:
        state.update(status='stopped',reason=repr(exc));raise
    finally:
        state['updated_at']=time.time();save(root/'study-status.json',state)

if __name__=='__main__':main()
