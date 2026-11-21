"""Bounded policy-free reset diagnosis; no editing of the original cohort."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json
import hashlib

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(REPO))
from weightsmith import config as C
from weightsmith.locks import GpuLock

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layouts',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entry=REPO/'study/iros2027/v2/rss_few_pair_validation_20261120/goals/preflight.py'
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='Y',VK_ICD_FILENAMES=C.VK_ICD,
        PYTHONPATH=str(C.ROBOLAB)+os.pathsep+str(REPO),OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env.pop('OPENAI_API_KEY',None);env.pop('WSP_START_LAYOUT_IDS',None)
    env.pop('DISPLAY',None);env.pop('WAYLAND_DISPLAY',None)
    command=[str(C.SIM_PY),'-u',str(HERE/'bootstrap.py'),str(entry),'--goal','mustard_right','--layouts',str(args.layouts.resolve()),
        '--output',str(args.output.resolve()),'--headless','--device','cuda:1',
        '--/rtx/verifyDriverVersion/enabled=false','--/renderer/multiGpu/enabled=false',
        '--/renderer/multiGpu/autoEnable=false','--/renderer/activeGpu=1']
    start=time.time()
    with GpuLock(1),(args.output/'native.log').open('x') as log:
        print('GPU1 policy-free reset diagnostic started',flush=True)
        result=subprocess.run(command,cwd=C.ROBOLAB,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    summary=json.loads((args.output/'summary.json').read_text()) if (args.output/'summary.json').exists() else {}
    report={'returncode':result.returncode,'summary_status':summary.get('status'),'reason':summary.get('reason'),
        'wall_seconds':time.time()-start,'layouts':str(args.layouts.resolve()),'layouts_sha256':sha(args.layouts),
        'entry_sha256':sha(entry),'diagnostic_code_sha256':sha(__file__),'bootstrap_sha256':sha(HERE/'bootstrap.py'),
        'policy_calls':summary.get('policy_calls'),'action_calls':summary.get('action_calls'),
        'passed':result.returncode==0 and summary.get('status')=='complete'}
    (args.output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
