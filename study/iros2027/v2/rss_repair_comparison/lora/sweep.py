"""Bounded development-only LoRA learning-rate sweep before closed-loop selection.

Preserves the original 1e-3 fit and trains exactly two additional rates. A root-
provided source freeze is mandatory. No policy episodes or held-out data are read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[3]
sys.path.insert(0,str(REPO))
from train_lora import dump, load_dataset, sha

NEW_RATES = (('lr-3e-5',3e-5),('lr-1e-4',1e-4))
LAYERS = list(range(8,17))
BUDGET_SECONDS = 15*60


def verify_identity(identity):
    for path,expected in identity.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input/source/candidate changed: {path}')


def read_freeze(path):
    payload=json.loads(Path(path).read_text())
    sources=payload.get('sources',payload)
    if not isinstance(sources,dict) or not sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64 for k,v in sources.items()):
        raise ValueError('Source freeze must contain a path-to-SHA256 mapping, directly or under sources')
    sources={str(Path(p).resolve()):digest for p,digest in sources.items()}
    for required in (Path(__file__),HERE/'train_lora.py',STUDY/'protocol.json'):
        if str(required.resolve()) not in sources:
            raise ValueError(f'Source freeze is missing required implementation/protocol: {required}')
    verify_identity(sources)
    return sources


def balanced_action_mse(training):
    """Equal weight to repair and preservation; never select on flow loss."""
    validation=training['final_validation_actions']
    means={}
    counts={}
    for role in ('repair','preservation'):
        values=[row['robot_action_mse'] for row in validation['items'] if row['role']==role]
        if not values or any(not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Finite development action errors for both roles are required')
        mean=sum(values)/len(values)
        reported=validation['by_role'][role]['robot_action_mse']
        if not math.isclose(mean,reported,rel_tol=1e-12,abs_tol=1e-15):
            raise ValueError('Per-role validation summary differs from its items')
        means[role]=mean;counts[role]=len(values)
    return (means['repair']+means['preservation'])/2,means,counts


def validate_recipe(training,dataset_sha,learning_rate,train_source_sha):
    expected={'rank':32,'layers':LAYERS,'steps':200,'learning_rate':learning_rate,
              'weight_decay':0.,'seed':17,'dataset_manifest_sha256':dataset_sha,
              'source_code_sha256':train_source_sha,'train_inputs':24,'validation_inputs':8}
    if any(training.get(key)!=value for key,value in expected.items()):
        raise ValueError('Candidate differs from the frozen data, source, or LoRA recipe')
    if training.get('closed_loop_evaluation') != 'not performed by this training command':
        raise ValueError('Training artifact has an unexpected evaluation boundary')
    allowed={f'language.blocks.{layer:02d}.mlp.linear' for layer in LAYERS}
    # The historical base manifest exposes action weights too. Its inherited
    # include_action flag describes that exposure, not this adapter's edits.
    # Verify the actual edited names here and full tensor paths below.
    if not set(training['workspace_validation']['edited_tensors']).issubset(allowed):
        raise ValueError('LoRA candidate changed a tensor outside the frozen VLM scope')
    _,_,counts=balanced_action_mse(training)
    if counts!={'repair':4,'preservation':4}:
        raise ValueError('Expected four validation inputs for each role')


def workspace_hashes(workspace):
    workspace=Path(workspace).resolve()
    config=json.loads((workspace/'workspace.json').read_text())
    expected={f'language.blocks.{layer:02d}.mlp.linear' for layer in LAYERS}
    if set(config['tensors']) != expected:
        raise ValueError('Expected exactly language down projections in layers 8–16')
    for layer in LAYERS:
        entry=config['tensors'][f'language.blocks.{layer:02d}.mlp.linear']
        if entry.get('model_path')!=['PaliGemma','llm','layers','mlp','linear'] or entry.get('scan_index')!=layer:
            raise ValueError('Workspace tensor definition points outside the declared VLM layer')
    files=[workspace/'workspace.json',*sorted((workspace/'edits').glob('*.npy'))]
    if len(files)!=10 or {p.name for p in files[1:]}!={item['file'] for item in config['tensors'].values()}:
        raise ValueError('Expected workspace metadata and exactly nine selected merged tensors')
    return {str(path):sha(path) for path in files}


def seal_candidate(fit,candidate_id,case,seal_path,source_identity_sha,dataset_sha):
    fit=Path(fit).resolve()
    workspace=workspace_hashes(fit/'workspace')
    artifacts={str(fit/name):sha(fit/name) for name in ('training.json','settings.json','adapters.npz')}
    artifacts.update(workspace)
    value={'candidate_id':candidate_id,'case':case,'workspace_artifacts':workspace,'artifacts':artifacts,
           'source_identity_sha256':source_identity_sha,'dataset_manifest_sha256':dataset_sha,'created_at':time.time()}
    if Path(seal_path).exists():
        raise FileExistsError(seal_path)
    dump(seal_path,value)
    return value


def candidate_record(fit,candidate_id,case,learning_rate,stage,seal_path,source_identity_sha,dataset_sha,train_source_sha,wrapper_seconds):
    fit=Path(fit).resolve();training=json.loads((fit/'training.json').read_text())
    validate_recipe(training,dataset_sha,learning_rate,train_source_sha)
    mse,by_role,counts=balanced_action_mse(training)
    seal=seal_candidate(fit,candidate_id,case,seal_path,source_identity_sha,dataset_sha)
    return {'candidate_id':candidate_id,'learning_rate':learning_rate,'fit_path':str(fit),
            'workspace':str(fit/'workspace'),'training_path':str(fit/'training.json'),
            'training_sha256':sha(fit/'training.json'),'balanced_validation_action_mse':mse,
            'validation_action_mse_by_role':by_role,'validation_counts_by_role':counts,
            'initial_validation_flow_loss':training['initial_validation_flow_loss'],
            'final_validation_flow_loss':training['final_validation_flow_loss'],
            'training_wall_seconds':training['wall_seconds'],'wrapper_wall_seconds':wrapper_seconds,
            'stage':stage,'status':'complete','seal_path':str(Path(seal_path).resolve()),'seal_sha256':sha(seal_path),
            'scope_verified_from':'Actual workspace model_path/scan_index and edited-tensor allowlist; inherited include_action exposure flag is not used as evidence.',
            'workspace_artifacts':seal['workspace_artifacts']}


def nominate(candidates):
    expected={'initial-1e-3',*(name for name,_ in NEW_RATES)}
    if {row['candidate_id'] for row in candidates}!=expected or len(candidates)!=3 or any(row['status']!='complete' for row in candidates):
        raise ValueError('All three complete candidates are required; failed trials cannot be silently omitted')
    new=[row for row in candidates if row['stage']=='new']
    if {row['candidate_id'] for row in new}!={name for name,_ in NEW_RATES}:
        raise ValueError('Only the two new learning rates can be nominated at Stage A')
    if any(not math.isfinite(row['balanced_validation_action_mse']) or row['balanced_validation_action_mse']<0 for row in candidates):
        raise ValueError('Invalid nomination metric')
    selected=min(new,key=lambda row:(row['balanced_validation_action_mse'],row['learning_rate']))
    return selected['candidate_id']


def run(args):
    started=time.time();deadline=time.monotonic()+BUDGET_SECONDS
    adaptation=Path(args.adaptation_root).resolve();freeze_path=Path(args.source_freeze).resolve()
    output=adaptation/'lora-sweep'
    if output.exists():
        raise FileExistsError('Existing sweep is preserved; inspect it before any separately authorized retry')
    sources=read_freeze(freeze_path)
    adaptation_status=json.loads((adaptation/'status.json').read_text())
    if adaptation_status.get('status')!='complete' or adaptation_status.get('case')!=args.case:
        raise ValueError('A complete original case adaptation is required')
    spec_path=adaptation/'spec.json';spec=json.loads(spec_path.read_text())
    if spec.get('case')!=args.case or spec.get('split')!='development':
        raise ValueError('Only the selected case development specification is allowed')
    dataset=adaptation/'lora-dataset';meta,examples=load_dataset(dataset);del examples
    if meta['spec_sha256']!=sha(spec_path):
        raise ValueError('Original dataset and development specification differ')
    dataset_sha=sha(dataset/'dataset.json');train_sha=sha(HERE/'train_lora.py')
    original=adaptation/'lora-fit'
    initial=json.loads((original/'training.json').read_text())
    validate_recipe(initial,dataset_sha,.001,train_sha)
    old_seals=json.loads((adaptation/'candidate-seals.json').read_text())
    initial_workspace=workspace_hashes(original/'workspace')
    if initial_workspace!=old_seals['lora']['artifacts']:
        raise ValueError('Original fitted workspace differs from its immutable candidate seal')
    identity=dict(sources)
    input_paths=[freeze_path,spec_path,dataset/'dataset.json',dataset/'input-spec.json',
                 adaptation/'candidate-seals.json',adaptation/'identity.json',adaptation/'status.json',
                 original/'training.json',original/'settings.json',original/'adapters.npz']
    input_paths.extend(dataset/row['file'] for row in meta['records'])
    for path in input_paths:identity[str(path.resolve())]=sha(path)
    identity.update(initial_workspace)
    output.mkdir()
    dump(output/'identity.json',identity)
    shutil.copyfile(freeze_path,output/'source-freeze.json')
    identity_sha=sha(output/'identity.json')
    for name,digest in sources.items():
        source=Path(name)
        if source.stat().st_size<=2*1024**2:
            target=output/'frozen-sources'/digest/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
    status={'case':args.case,'split':'development','status':'starting','started_at':started,
            'budget_seconds':BUDGET_SECONDS,'new_learning_rates':[rate for _,rate in NEW_RATES],
            'completed_new_trials':[],'failed_attempts':[],'closed_loop_evaluation':'not performed by this sweep'}
    dump(output/'status.json',status)
    candidates=[]
    try:
        wrapper_initial=next(stage['wall_seconds'] for stage in adaptation_status['completed_stages'] if stage['stage']=='lora-train')
        candidates.append(candidate_record(original,'initial-1e-3',args.case,.001,'initial',output/'initial-seal.json',
                                           identity_sha,dataset_sha,train_sha,wrapper_initial))
        from weightsmith import config as C
        env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
                 JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='1')
        for candidate_id,rate in NEW_RATES:
            verify_identity(identity)
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Fifteen-minute case sweep budget exhausted')
            fit=output/candidate_id;log=output/f'{candidate_id}.log'
            command=[C.OPENPI_PY,str(HERE/'train_lora.py'),'train','--dataset',str(dataset),'--output',str(fit),
                     '--gpu',str(args.gpu),'--lr',str(rate),'--rank','32','--steps','200','--seed','17',
                     '--weight-decay','0','--layers',*[str(layer) for layer in LAYERS]]
            trial_start=time.time()
            status.update(status='running',candidate_id=candidate_id,learning_rate=rate,trial_started_at=trial_start)
            dump(output/'status.json',status)
            print(f'START {args.case} {candidate_id}',flush=True)
            try:
                with log.open('w') as stream:
                    subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                   check=True,timeout=remaining)
                verify_identity(identity)
                record=candidate_record(fit,candidate_id,args.case,rate,'new',fit/'seal.json',
                                        identity_sha,dataset_sha,train_sha,time.time()-trial_start)
                record['log_path']=str(log);record['log_sha256']=sha(log)
                candidates.append(record)
            except BaseException as exc:
                status['failed_attempts'].append({'candidate_id':candidate_id,'learning_rate':rate,'reason':repr(exc),
                                                 'wall_seconds':time.time()-trial_start,'log_path':str(log)})
                raise
            status['completed_new_trials'].append(record)
            dump(output/'status.json',status)
            print(f'DONE {args.case} {candidate_id}: balanced validation action MSE={record["balanced_validation_action_mse"]:.9g}',flush=True)
        verify_identity(identity)
        for row in candidates:verify_identity(json.loads(Path(row['seal_path']).read_text())['artifacts'])
        nominated=nominate(candidates)
        nomination={'case':args.case,'status':'complete','split':'development','cohort_id':meta['cohort_id'],
            'dataset_path':str(dataset),'dataset_manifest_sha256':dataset_sha,'spec_path':str(spec_path),'spec_sha256':sha(spec_path),
            'source_freeze_path':str(freeze_path),'source_freeze_sha256':sha(freeze_path),
            'source_identity_path':str(output/'identity.json'),'source_identity_sha256':identity_sha,
            'stage_a_rule':'Minimum mean development-validation physical action MSE, equal repair/preservation weight; exact ties use smaller learning rate; nominate only among the two NEW rates.',
            'selected_new_candidate_id':nominated,'closed_loop_candidate_ids':['initial-1e-3',nominated],
            'closed_loop_selection_status':'pending; root-controlled fresh development episodes required',
            'closed_loop_selection_rule':'Highest sum of strict successes across repair and primary preservation; tie lower balanced validation action MSE, then lower learning rate.',
            'candidates':candidates,'failed_attempts':status['failed_attempts'],
            'costs':{'initial_training_seconds':candidates[0]['training_wall_seconds'],
                     'new_training_seconds':sum(row['training_wall_seconds'] for row in candidates if row['stage']=='new'),
                     'total_training_seconds':sum(row['training_wall_seconds'] for row in candidates),
                     'shared_dataset_preparation_seconds':meta['wall_seconds'],
                     'sweep_wall_seconds':time.time()-started,'all_three_fit_steps':600},
            'independent_policy_episodes_generated':0,'source_code_sha256':sha(__file__)}
        dump(output/'nomination.json',nomination)
        status.update(status='complete',selected_new_candidate_id=nominated,wall_seconds=time.time()-started,
                      nomination_sha256=sha(output/'nomination.json'))
        print(json.dumps({'case':args.case,'nominated':nominated,'closed_loop_candidates':['initial-1e-3',nominated],
                          'wall_seconds':status['wall_seconds']}),flush=True)
    except BaseException as exc:
        status.update(status='stopped',reason=repr(exc),wall_seconds=time.time()-started)
        raise
    finally:
        dump(output/'status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptation-root',type=Path,required=True)
    parser.add_argument('--case',choices=['temporal','spatial'],required=True)
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--source-freeze',type=Path,required=True)
    args=parser.parse_args()
    run(args)


if __name__=='__main__':main()
