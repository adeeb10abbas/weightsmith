"""Qualify the original poses using the same per-batch process lifecycle as evaluation.

Every requested layout is retained. This only partitions a multi-batch geometry
file into fresh-process fixed16 checks; it never changes a pose or tolerance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ORIGINAL=REPO/'study/iros2027/v2/rss_few_pair_validation_20261120'
sys.path.insert(0,str(ORIGINAL))
from common import read,save,sha,verify
from preflight import validate_summary
import importlib.util

def load_check():
    spec=importlib.util.spec_from_file_location('recovery_scene_check',HERE/'check.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def prepare(layouts,output):
    layouts=Path(layouts).resolve();output=Path(output).resolve();source=read(layouts)
    if len(source['layouts'])%16 or not source['layouts']:
        raise ValueError('Require full batches of sixteen original layouts')
    output.mkdir(parents=True,exist_ok=True)
    record={'original_layouts':str(layouts),'original_layouts_sha256':sha(layouts),
        'goal_id':source['goal_id'],'stage':source['stage'],
        'process_lifecycle':'fresh process per16-layout batch; two resets within each process',
        'geometry_changed':False,'scoring_changed':False,'tolerances_changed':False,
        'parent_failure':str(REPO/'projects/rss-few-pair-validation-20261120/mustard_right/preflight/collection/summary.json'),
        'partitions':[]}
    for group in range(len(source['layouts'])//16):
        subset={**source,'layouts':source['layouts'][16*group:16*(group+1)]}
        subset['preflight_partition']={'source':str(layouts),'source_sha256':sha(layouts),'group':group,
            'pose_change':False,'policy_outcomes_used':False}
        path=output/f'batch-{group:02d}-layouts.json';save(path,subset,exclusive=True)
        record['partitions'].append({'group':group,'path':str(path),'sha256':sha(path),
            'output':str(output/f'batch-{group:02d}'),'layout_ids':[x['id'] for x in subset['layouts']]})
    save(output/'partition.json',record,exclusive=True)
    return record

def certify(layouts,output):
    output=Path(output).resolve();record=read(output/'partition.json');source=read(layouts)
    if record['original_layouts']!=str(Path(layouts).resolve()) or record['original_layouts_sha256']!=sha(layouts):
        raise ValueError('Original geometry changed during qualification')
    artifacts={str(Path(layouts).resolve()):sha(layouts),str(output/'partition.json'):sha(output/'partition.json'),
        str(HERE/'isolated_preflight.py'):sha(HERE/'isolated_preflight.py'),str(HERE/'check.py'):sha(HERE/'check.py'),
        str(HERE/'bootstrap.py'):sha(HERE/'bootstrap.py')}
    observed=[];children=[]
    for p in record['partitions']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Partition changed')
        child=Path(p['output']);diagnostic=read(child/'diagnostic.json');summary=read(child/'summary.json')
        if diagnostic['returncode']!=0 or not diagnostic['passed'] or diagnostic['layouts_sha256']!=p['sha256']:
            raise ValueError('Child process did not pass')
        validate_summary(summary,source['goal_id'],p['path'])
        observed.extend(x for reset in summary['resets'] for x in reset['layout_ids'])
        for path in [Path(p['path']),*sorted(child.glob('*.json'))]:artifacts[str(path.resolve())]=sha(path)
        artifacts.update(summary['source_sha256'])
        children.append({'group':p['group'],'returncode':diagnostic['returncode'],
            'summary':str(child/'summary.json'),'summary_sha256':sha(child/'summary.json'),
            'reset_cycles':summary['reset_cycles'],'policy_calls':summary['policy_calls'],'action_calls':summary['action_calls']})
    if sorted(observed)!=sorted([r['id'] for r in source['layouts']]*2):
        raise ValueError('Each original layout must pass exactly twice')
    verify(artifacts)
    receipt={**record,'status':'passed','children':children,'policy_calls':0,'action_calls':0,
        'reset_cycles':sum(c['reset_cycles'] for c in children),'layout_count':len(source['layouts']),
        'certified_unix':time.time(),'artifacts':artifacts}
    save(output/'receipt.json',receipt,exclusive=True)
    return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layouts',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    record=prepare(args.layouts,args.output);check=load_check()
    for p in record['partitions']:
        output=Path(p['output'])
        if (output/'diagnostic.json').exists():
            d=read(output/'diagnostic.json')
            if not d['passed']:raise ValueError('A failed attempt is retained; no implicit retry')
            continue
        sys.argv=[str(HERE/'check.py'),'--layouts',p['path'],'--output',str(output)]
        check.main()
    result=certify(args.layouts,args.output)
    print(json.dumps({'status':result['status'],'layout_count':result['layout_count'],'receipt':str(args.output/'receipt.json')}),flush=True)

if __name__=='__main__':main()
