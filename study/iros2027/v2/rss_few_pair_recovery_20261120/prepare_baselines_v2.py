"""Construct one embedding direction and three source-matched residual fields."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import time
import sys
from recovery import HERE,STUDY,RECROOT,AMENDMENT,amendment
from common import REPO,ROOT,goal,goal_root,protocol,read,save,sha,verify

def run(goal_id,gpu):
    from weightsmith import config as C
    g=goal(goal_id);root=goal_root(goal_id);dest=root/'adaptation/baselines-recovery-v2'
    output=root/'adaptation/baselines/candidates.json'
    if output.exists():
        record=read(output);verify(record['artifacts']);return record
    source_status=read(root/'source-status.json')
    if source_status['status']!='qualified': raise ValueError('Nine qualified sources required')
    manifest=Path(source_status['manifest']);source=read(manifest)
    if sha(manifest)!=source_status['manifest_sha256']: raise ValueError('Source manifest changed')
    edits=read(root/'adaptation/weightsmith/candidates.json')
    if edits['source_manifest_sha256']!=sha(manifest): raise ValueError('Different edit source pool')
    verify(edits['artifacts'])
    amendment(); gate=RECROOT/'baseline-v2-qualification/passed.json';cert=read(gate)
    if cert['status']!='pass' or cert['new_policy_episodes']!=0: raise ValueError('Baseline identity gate missing')
    verify(cert['artifacts'])
    dest.mkdir(parents=True,exist_ok=True);costs=[];candidates=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    def execute(label,args):
        command=[str(C.OPENPI_PY),str(HERE/'baselines_v2/prepare.py'),*map(str,args)]
        start=time.time()
        with (dest/(label+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if result.returncode: raise RuntimeError('Baseline construction failed; preserved '+label)
        costs.append({'construction':label,'wall_seconds':time.time()-start})
    families=[('embedding',None)] + [('residual_'+x['group_id'],x) for x in source['groups']]
    for family,group in families:
        base=dest/family/'direction'
        if group is None:
            args=['embedding','--original-prompt',g['original'],'--clarified-prompt',g['teacher'],'--gpu',gpu]
        else:
            args=['residual','--original-capture',edits['captures']['original'],
                  '--teacher-capture',edits['captures']['teacher'],'--indices',*group['indices'],
                  '--layer',protocol()['baselines']['residual_layer']]
        execute(family+'-direction',[*args,'--source-manifest',manifest,'--output',base])
        for strength in protocol()['baselines']['strengths']:
            label=str(strength).replace('.','p');workspace=dest/family/('strength-'+label)
            execute(family+'-'+label,['strength','--source',base,'--strength',strength,'--output',workspace])
            candidates.append({'candidate_id':family+'-s'+label,'family':family,
                'group_id':None if group is None else group['group_id'],
                'method':'embedding' if group is None else 'residual','strength':strength,
                'workspace':str(workspace),'source_pair_ids':[] if group is None else group['pair_ids'],
                'source_manifest':str(manifest),'source_manifest_sha256':sha(manifest)})
    files=[Path(__file__),AMENDMENT,gate,manifest,root/'adaptation/weightsmith/candidates.json',
           *sorted((HERE/'baselines_v2').glob('*.py')),*sorted(dest.glob('**/baseline.json')),
           *sorted(dest.glob('**/direction.npy'))]
    record={'cohort_id':protocol()['cohort_id'],'goal_id':goal_id,'status':'complete',
            'source_manifest':str(manifest),'source_manifest_sha256':sha(manifest),
            'runtime_recovery':'Punctuation-safe instruction boundaries; no changes to strengths, source groups, or selection rule.', 'candidates':candidates,'costs':costs,'artifacts':{str(p.resolve()):sha(p) for p in files},
            'scope':'One source-independent embedding field; three source-matched residual fields. No held-out outcomes used.'}
    save(output,record,exclusive=True);return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--goal',required=True,choices=protocol()['goal_ids'])
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True);args=ap.parse_args()
    run(args.goal,args.gpu)
