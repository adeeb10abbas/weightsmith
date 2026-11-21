"""Own and verify policy-free scene-reset checks for each frozen split."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import time
from common import HERE,goal,goal_root,native_sources,protocol,read,save,sha,verify

def validate_summary(summary,goal_id,layout):
    spec=read(layout);expected=len(spec['layouts'])//16*2
    if (summary.get('status')!='complete' or summary.get('goal_id')!=goal_id or
        summary.get('reset_cycles')!=expected or summary.get('policy_calls')!=0 or
        summary.get('action_calls')!=0 or summary.get('forbidden_modules_loaded')):
        raise ValueError('Policy-free reset validation did not complete')
    ids=[r['id'] for r in spec['layouts']]
    if summary.get('layout_ids')!=ids: raise ValueError('Reset layout set differs')
    observed=[x for row in summary['resets'] for x in row['layout_ids']]
    if sorted(observed)!=sorted(ids*2): raise ValueError('Each layout must pass two resets')
    if summary['source_sha256'].get(str(Path(layout).resolve()))!=sha(layout):
        raise ValueError('Reset certificate does not bind the requested layout file')
    for reset in summary['resets']:
        if reset.get('vertical_displacement_checked') is not True:
            raise ValueError('Missing vertical-reset qualification')
        support=reset.get('table_contact',{})
        if set(support)!=set(spec['objects']) or any(len(x)!=16 or not all(x) for x in support.values()):
            raise ValueError('Missing table-support qualification')
    verify(summary['source_sha256'])

def run(goal_id,split,gpu):
    if split not in ('collection','development','heldout'): raise ValueError('Unknown split')
    from weightsmith import config as C
    from weightsmith.locks import GpuLock
    layout=goal_root(goal_id)/f'design/{split}-layouts.json'
    output=goal_root(goal_id)/f'preflight/{split}'
    receipt=output/'receipt.json'
    if receipt.exists():
        record=read(receipt);verify(record['artifacts'])
        validate_summary(read(output/'summary.json'),goal_id,layout)
        if record['returncode']!=0: raise ValueError('Failed reset process')
        return record
    if output.exists(): raise ValueError(f'Partial preflight preserved; no automatic retry: {output}')
    output.mkdir(parents=True)
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='Y',VK_ICD_FILENAMES=C.VK_ICD,
             PYTHONPATH=str(C.ROBOLAB)+os.pathsep+os.environ.get('PYTHONPATH',''))
    env.pop('OPENAI_API_KEY',None)
    command=[str(C.SIM_PY),str(HERE/'goals/preflight.py'),'--goal',goal_id,
             '--layouts',str(layout),'--output',str(output),'--headless','--device',f'cuda:{gpu}',
             '--/rtx/verifyDriverVersion/enabled=false','--/renderer/multiGpu/enabled=false',
             '--/renderer/multiGpu/autoEnable=false',f'--/renderer/activeGpu={gpu}']
    start=time.time()
    with GpuLock(gpu), (output/'native.log').open('x') as log:
        result=subprocess.run(command,cwd=C.ROBOLAB,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    if result.returncode!=0: raise RuntimeError(f'Reset preflight failed: {output}/native.log')
    validate_summary(read(output/'summary.json'),goal_id,layout)
    paths=[*native_sources(goal_id),Path(__file__),layout,*output.glob('*.json')]
    record={'status':'passed','goal_id':goal_id,'split':split,'returncode':result.returncode,
            'policy_calls':0,'action_calls':0,'wall_seconds':time.time()-start,
            'artifacts':{str(p.resolve()):sha(p) for p in sorted(set(paths))}}
    save(receipt,record,exclusive=True);return record

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--goal',required=True,choices=protocol()['goal_ids'])
    ap.add_argument('--split',required=True,choices=('collection','development','heldout'))
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True)
    args=ap.parse_args();print(run(args.goal,args.split,args.gpu))
